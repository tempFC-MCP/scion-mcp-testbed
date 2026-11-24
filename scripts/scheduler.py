"""Epoch scheduler and dependency catalogue. No live consensus or signature validation.

Inputs are accepted, normalized service records. Repeated appearances across
rounds are allowed; duplicate validators inside a round are not.
"""
from itertools import chain, combinations
from time import perf_counter
import numpy as np


def normalize_scion_path(path, accepted_record_ids):
    """Extract dependencies from an assembled path verified by the control plane.

    accepted_record_ids comes from the trusted adapter, not the advertiser.
    Raw PCB signature verification and physical geolocation are not done here.
    """
    if not path.get('as_entries'):
        raise ValueError('Missing assembled AS sequence')
    return dict(dependencies=sorted({'AS:'+str(e['isd_as']) for e in path['as_entries']}),
                accepted=path['record_id'] in accepted_record_ids,
                expires=path['expires'],latency_ms=path['latency_ms'])


def usable_paths(service, epoch_end, latency_ms):
    return [frozenset(p['dependencies']) for p in service
            if p['accepted'] and p['expires'] >= epoch_end
            and p['latency_ms'] <= latency_ms]


def minimal_cuts(paths, max_order=2):
    """Inclusion-minimal hitting sets of size <= max_order; [] => empty cut."""
    cuts = {frozenset()}
    for path in paths:
        candidates = {c if c & path else c | {d}
                      for c in cuts for d in path}
        candidates = {c for c in candidates if len(c) <= max_order}
        cuts = {c for c in candidates if not any(b < c for b in candidates)}
    return cuts


def catalogue(records, epoch_end=1, latency_ms=200, max_order=1):
    """All listed services are required; at least one path per service suffices.

    Cut enumeration is bounded, not an assertion that all joint failures are
    covered. Mandatory dependencies represent endpoint/hosting availability.
    """
    normalized = []
    scenarios = set()
    for record in records:
        services = [usable_paths(s, epoch_end, latency_ms) for s in record['services']]
        if not services or any(not p for p in services):
            raise ValueError('No usable path for a required service')
        mandatory = frozenset(record['mandatory'])
        normalized.append((mandatory, services))
        scenarios.update(frozenset([d]) for d in mandatory)
        for paths in services:
            scenarios.update(frozenset([d]) for p in paths for d in p)
            scenarios.update(minimal_cuts(paths, max_order))
    scenarios = sorted(scenarios, key=lambda s: (len(s), tuple(sorted(s))))
    failures = np.array([[bool(s & m) or any(all(s & p for p in paths)
                           for paths in services) for m, services in normalized]
                         for s in scenarios], dtype=np.int8)
    return scenarios, failures.reshape(len(scenarios), len(normalized))


def quotas(schedule, n):
    return np.bincount(np.asarray(schedule).ravel(), minlength=n)


def appearances(schedule, regions, nr):
    out = np.zeros((len(schedule), nr), dtype=np.int8)
    for t, group in enumerate(schedule):
        out[t, regions[group]] = 1
    return out


def window_misses(a, h):
    if h is None:
        return 0
    represented = np.zeros(a.shape, dtype=bool)
    for offset in range(h):
        np.logical_or(represented, np.roll(a, -offset, axis=0), out=represented)
    return int(np.count_nonzero(~represented))


class InvalidSchedule(ValueError):
    """A proposed assignment violates a required scheduling constraint."""


class NoVerifiedSchedule(InvalidSchedule):
    """None of the supplied schedules passed the current hard constraints."""


def metrics(schedule, instance):
    """Return per-outage maxima in the row order of F and its scenario catalogue."""
    try:
        a = np.asarray(schedule)
    except (TypeError, ValueError) as exc:
        raise InvalidSchedule('Expected a rectangular assignment') from exc
    cost = np.asarray(instance['cost'])
    pair_delay = np.maximum(cost, cost.T)
    n, N = instance['n'], instance['N']
    if a.shape != (instance['T'], N):
        raise InvalidSchedule('Wrong number of rounds or proposers')
    if not np.issubdtype(a.dtype, np.integer) or np.any(a < 0) or np.any(a >= n):
        raise InvalidSchedule('Unknown validator index')
    a = a.astype(np.intp, copy=False)
    if any(len(set(group)) != N for group in a):
        raise InvalidSchedule('A validator appears twice in one round')
    if not np.array_equal(quotas(a, n), instance['q']):
        raise InvalidSchedule('Assignment changes validator quotas')
    if 'eligible' in instance:
        eligible = np.asarray(instance['eligible'], dtype=bool)
        if eligible.shape != (instance['T'], n):
            raise ValueError('Invalid round eligibility matrix')
        if any(not eligible[t, group].all() for t, group in enumerate(a)):
            raise InvalidSchedule('Ineligible proposer in round')
    if not all(pair_delay[u, v] <= instance['tau']
               for group in a for u, v in combinations(group, 2)):
        raise InvalidSchedule('A proposer pair exceeds the communication limit')
    failures = np.asarray(instance['F'])
    if failures.shape == (0,):  # JSON stores an empty matrix as [].
        failures = failures.reshape(0, n)
    if failures.shape != (len(instance['budget']), n):
        raise ValueError('Expected one outage row per budget and one column per validator')
    if 'scenarios' in instance and len(instance['scenarios']) != len(failures):
        raise ValueError('Expected one scenario identifier per outage row')
    exposure = failures[:, a].sum(axis=2)
    outage_maxima = exposure.max(axis=1, initial=0)
    geo = np.array([len(set(instance['regions'][g])) for g in a])
    aa = appearances(a, instance['regions'], instance['nr'])
    misses = window_misses(aa, instance.get('h'))
    gaps = []
    for r in range(instance['nr']):
        pos = np.flatnonzero(aa[:, r])
        gaps.append(int(max(np.diff(np.r_[pos, pos[0] + len(a)])) - 1)
                    if len(pos) else len(a))
    excess = np.maximum(exposure - instance['budget'][:, None], 0)
    deficit = np.maximum(instance['g'] - geo, 0)
    return dict(exposure_max=int(outage_maxima.max(initial=0)),
                exposure_max_by_scenario=outage_maxima.tolist(),
                excess=int(excess.sum()), geo_deficit=int(deficit.sum()),
                mean_regions=float(geo.mean()), window_misses=misses,
                max_cyclic_gap_capped_T=max(gaps),
                mean_pair_ms=float(np.mean([pair_delay[u, v]
                    for g in a for u, v in combinations(g, 2)])) if N > 1 else 0.0,
                compliant=bool(not excess.any() and not deficit.any() and misses == 0))


def verify_recorded_metrics(recorded, recomputed, *, approximate_floats=False):
    """Compare recomputed metrics with a stored run, including older scalar-only reports."""
    for key, value in recomputed.items():
        if key == 'exposure_max_by_scenario' and key not in recorded:
            continue
        if key not in recorded:
            raise ValueError(f'Missing saved metric: {key}')
        if approximate_floats and isinstance(value, float):
            matches = np.isclose(recorded[key], value)
        else:
            matches = recorded[key] == value
        if not matches:
            raise ValueError(f'Saved metric differs from recomputation: {key}')


def weighted_score(stats):
    """Equation 9 with the three weights used in the experiments."""
    return 10 * stats['geo_deficit'] + stats['excess'] + 2 * stats['window_misses']


def adopt_schedule(instance, candidates):
    """Choose the best verified schedule available at the decision deadline.

    Consider the original first, then the candidates in their agreed order.
    Ties retain the earlier valid schedule. A new snapshot may invalidate the
    original. Raise NoVerifiedSchedule if no supplied schedule passes.
    """
    best = None
    best_score = float('inf')
    options = chain([('original', instance['initial'])], candidates.items())
    for name, candidate in options:
        if candidate is None:
            continue
        try:
            candidate_stats = metrics(candidate, instance)
        except InvalidSchedule:
            continue
        score = weighted_score(candidate_stats)
        if score < best_score:
            best = name, candidate, candidate_stats
            best_score = score
    if best is None:
        raise NoVerifiedSchedule('No supplied schedule satisfies the current hard constraints')
    return best


def local_search(instance, mode='full', iterations=3000, seed=0):
    """Quota-preserving swap descent. No approximation or optimality guarantee."""
    a = instance['initial'].copy()
    if len(a) == 1:
        return a, 0.0  # No second round to swap with.
    rng = np.random.default_rng(seed)
    cost = np.asarray(instance['cost'])
    pair_delay = np.maximum(cost, cost.T)
    F = instance['F'] if mode == 'full' else instance['provider_F']
    b = instance['budget'] if mode == 'full' else instance['provider_budget']
    def score(group):
        if mode == 'latency':
            return sum(pair_delay[u, v] for u, v in combinations(group, 2))
        geo = max(0, instance['g'] - len(set(instance['regions'][group])))
        risk = int(np.maximum(F[:, group].sum(axis=1) - b, 0).sum())
        return 10 * geo + (risk if mode in ('full', 'provider') else 0)
    scores = [score(g) for g in a]
    h = instance.get('h') if mode == 'full' else None
    aa = appearances(a, instance['regions'], instance['nr'])
    windows = window_misses(aa, h)
    start = perf_counter()
    for _ in range(iterations):
        t, u = rng.choice(len(a), size=2, replace=False)
        i, j = rng.integers(instance['N'], size=2)
        x, y = a[t, i], a[u, j]
        if 'eligible' in instance and (not instance['eligible'][t][y]
                                      or not instance['eligible'][u][x]):
            continue
        if x in a[u] or y in a[t]:
            continue
        at, au = a[t].copy(), a[u].copy()
        at[i], au[j] = y, x
        if any(pair_delay[p, q] > instance['tau']
               for g in (at, au) for p, q in combinations(g, 2)):
            continue
        st, su = score(at), score(au)
        oldt, oldu = aa[t].copy(), aa[u].copy()
        aa[t] = 0; aa[u] = 0
        aa[t, instance['regions'][at]] = 1
        aa[u, instance['regions'][au]] = 1
        nw = window_misses(aa, h)
        if st + su + 2 * nw < scores[t] + scores[u] + 2 * windows:
            a[t], a[u], scores[t], scores[u], windows = at, au, st, su, nw
        else:
            aa[t], aa[u] = oldt, oldu
    return a, perf_counter() - start


def solve_direct(instance, strict=False, time_limit=5):
    """Sparse validator-round MILP with explicit deficits; pairwise QoS adapter.

    Objective 10*geographic deficits + exposure excess + 2*missed cyclic windows.
    In strict mode deficits are fixed to zero. Exact quotas and QoS stay hard.
    """
    begin = perf_counter()
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
    n, T, N, nr = (instance[k] for k in ('n','T','N','nr'))
    F, budgets = instance['F'], instance['budget']
    ns, h = len(F), instance.get('h')
    # z[n,T], a[nr,T], d[T], e[ns,T], w[nr,T] if enabled
    az, ad = n*T, (n+nr)*T
    ae, aw = ad+T, ad+T+ns*T
    nv = aw + (nr*T if h is not None else 0)
    c = np.zeros(nv); c[ad:ae] = 10; c[ae:aw] = 1; c[aw:] = 2
    ub = np.ones(nv); ub[ad:ae] = instance['g']; ub[ae:aw] = N
    if 'eligible' in instance:
        eligible = np.asarray(instance['eligible'], dtype=bool)
        assert eligible.shape == (T, n)
        ub[:n*T] = eligible.T.ravel()
    if strict: ub[ad:] = 0
    rows, cols, vals, lower, upper = [], [], [], [], []
    def add(entries, lo=-np.inf, hi=np.inf):
        row = len(lower)
        for j,v in entries:
            rows.append(row); cols.append(j); vals.append(v)
        lower.append(lo); upper.append(hi)
    for t in range(T): add(((v*T+t,1) for v in range(n)), N, N)
    for v in range(n): add(((v*T+t,1) for t in range(T)), instance['q'][v], instance['q'][v])
    for r in range(nr):
        members = np.flatnonzero(instance['regions'] == r)
        for t in range(T):
            # a=1 iff at least one selected member, without per-member rows.
            terms = [(v*T+t,1) for v in members]
            add(terms+[(az+r*T+t,-1)], lo=0)
            add(terms+[(az+r*T+t,-N)], hi=0)
    for t in range(T):
        add([(az+r*T+t,1) for r in range(nr)]+[(ad+t,1)], lo=instance['g'])
    for s, failed in enumerate(F):
        for t in range(T):
            add([(v*T+t,1) for v in np.flatnonzero(failed)]+[(ae+s*T+t,-1)], hi=budgets[s])
    for u in range(n):
        for v in range(u+1,n):
            if max(instance['cost'][u,v], instance['cost'][v,u]) > instance['tau']:
                for t in range(T): add([(u*T+t,1),(v*T+t,1)],hi=1)
    if h is not None:
        for r in range(nr):
            for t in range(T):
                add([(az+r*T+(t+j)%T,1) for j in range(h)]+[(aw+r*T+t,1)],lo=1)
    mat = coo_matrix((vals,(rows,cols)), shape=(len(lower),nv)).tocsc()
    start = perf_counter()
    result = milp(c, integrality=np.ones(nv), bounds=Bounds(np.zeros(nv),ub),
                  constraints=LinearConstraint(mat,lower,upper),
                  options={'time_limit':time_limit,'mip_rel_gap':0})
    elapsed = perf_counter()-start
    schedule = None
    if result.x is not None and np.max(np.abs(result.x-np.rint(result.x))) < 1e-5:
        z = np.rint(result.x[:n*T]).reshape(n,T)
        schedule = np.array([np.flatnonzero(z[:,t]) for t in range(T)])
        metrics(schedule, instance)
    metadata = dict(status=int(result.status), message=result.message, solve_seconds=elapsed,
                    construction_seconds=start-begin,total_seconds=perf_counter()-begin,
                    variables=nv, constraints=len(lower),
                    objective=float(result.fun) if result.fun is not None else None,
                    dual_bound=float(result.mip_dual_bound) if getattr(result,'mip_dual_bound',None) is not None else None,
                    gap=float(result.mip_gap) if getattr(result,'mip_gap',None) is not None else None)
    if schedule is not None:
        m=metrics(schedule,instance)
        value=weighted_score(m)
        metadata['verified_objective']=value
        lb=metadata['dual_bound']
        metadata['verified_gap']=max(0.,(value-lb)/max(1,abs(value))) if lb is not None else None
    return schedule, metadata
