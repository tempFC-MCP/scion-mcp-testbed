"""Compare two grouping rules using the paths measured in an expanded SCION run."""
import argparse
import json
from itertools import combinations
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from scheduler import adopt_schedule, local_search, metrics, quotas, solve_direct, weighted_score

class NoEligibleStart(ValueError):
    """The four-round allocation batch cannot be split into delay-valid triples."""


def partition(members, admissible):
    if not members:
        return []
    first = min(members)
    for group in admissible[first]:
        if group <= members:
            rest = partition(members - group, admissible)
            if rest is not None:
                return [sorted(group), *rest]
    return None


def eligible_start(original, cost, limit, seed):
    """Preserve the hub order and both turns. Repartition only incompatible batches."""
    schedule = original.copy()
    rng = np.random.default_rng(seed)
    for start in range(0, len(schedule), 4):
        batch = schedule[start:start + 4]
        members = batch.flatten()
        assert len(set(members)) == 12
        for _ in range(10000):
            if np.all(cost[batch[:, :, None], batch[:, None, :]] <= limit):
                break
            batch[:] = rng.permutation(members).reshape(4, 3)
        else:
            groups = [frozenset(g) for g in combinations(members, 3)
                      if all(cost[u, v] <= limit for u, v in combinations(g, 2))]
            admissible = {v: [group for group in groups if v in group] for v in members}
            exact = partition(set(members), admissible)
            if exact is None:
                raise NoEligibleStart(f"No valid partition of the four rounds starting at {start} within {limit} ms")
            batch[:] = exact
    return schedule


def instance(data, limit):
    regions = np.array(data['regions'])
    cost = np.array(data['cost'])
    original = np.array(data['initial'])
    hubs = regions[original[:, 0]]
    eligible = regions[None, :] == hubs[:, None]
    start = eligible_start(original, cost, limit, data['measurement']['seed'])
    transit = []
    for validator, routes in zip(data['validators'], data['routes'], strict=True):
        services = routes['services']
        if validator['id'] != routes['validator'] or len(services) != 2 or any(len(s) != 1 for s in services):
            raise ValueError('Expected one shared primary route to each observer per validator')
        cores = {s[0]['core'] for s in services}
        if len(cores) != 1:
            raise ValueError('Routes to the two observers use different transit ASes')
        transit.append(cores.pop())
    transit = np.asarray(transit)
    hosting = np.array([v['provider'] for v in data['validators']])
    hosts = np.array([hosting == label for label in np.unique(hosting)])
    transits = np.array([transit == label for label in np.unique(transit)])
    q = quotas(original, data['n'])
    dominant = int(np.argmax(transits @ q))
    failed = np.concatenate((transits, hosts))
    budgets = np.ones(len(failed), dtype=int)
    budgets[dominant] = 2
    model = dict(n=data['n'], T=len(start), N=data['N'], nr=data['nr'], g=1, h=None, q=q,
                 regions=regions, cost=cost, tau=limit, eligible=eligible, initial=start,
                 scenarios=([['AS:' + label] for label in np.unique(transit)]
                            + [['host:' + label] for label in np.unique(hosting)]),
                 F=failed, budget=budgets, provider_F=hosts,
                 provider_budget=np.ones(len(hosts), dtype=int))
    metrics(start, model)
    return model, dominant


def select_policy(model, dominant, seed, policy):
    """Return the adopted schedule and its score under both rules' common catalogue."""
    if policy == 'transit':
        objective = dict(model, F=model['F'][[dominant]], budget=model['budget'][[dominant]],
                         scenarios=[model['scenarios'][dominant]])
    elif policy == 'full':
        objective = model
    else:
        raise ValueError(f'Unknown scheduling rule: {policy}')
    start = perf_counter()
    candidate, _ = local_search(objective, iterations=4000, seed=seed + 100)
    seconds = perf_counter() - start
    selected, chosen, checked = adopt_schedule(objective, {policy: candidate})
    metrics(chosen, model)
    exposure = model['F'][:, chosen].sum(axis=2)
    assert np.array_equal(exposure.sum(axis=1), model['F'] @ model['q'])
    hosted = model['provider_F'][:, chosen].sum(axis=2)
    row = dict(policy=policy, seed=seed, seconds=seconds, adopted=selected,
               initial_score=weighted_score(metrics(model['initial'], objective)),
               adopted_score=weighted_score(checked),
               transit_success=int(np.count_nonzero(exposure[dominant] <= 2)),
               all_hosts_distinct=int(np.count_nonzero(hosted.max(axis=0) == 1)),
               joint_success=int(np.count_nonzero((exposure <= model['budget'][:, None]).all(axis=0))))
    return chosen, row


def run_comparison(model, dominant, seed):
    rows = {}
    measurements = {'transit': [], 'full': []}
    schedules = {}
    for repetition in range(6):
        order = ('transit', 'full') if repetition % 2 == 0 else ('full', 'transit')
        for name in order:
            chosen, stats = select_policy(model, dominant, seed, name)
            if repetition == 0:
                schedules[name], rows[name] = chosen, stats
            else:
                assert np.array_equal(chosen, schedules[name])
                measurements[name].append(stats['seconds'])
    for name, row in rows.items():
        row['seconds'] = median(measurements[name])
    return list(rows.values())


def two_survivor_schedule(model, dominant, distinct_hosts):
    """Maximize rounds with two surviving proposers on the recorded hub sequence."""
    survives = ~model['F'][dominant]
    schedule = np.empty_like(model['initial'])
    hubs = model['regions'][model['initial'][:, 0]]
    for hub in np.unique(hubs):
        members = np.flatnonzero(model['regions'] == hub)
        groups = np.array(list(combinations(members, model['N'])))
        delays = model['cost'][groups[:, :, None], groups[:, None, :]]
        groups = groups[np.all(delays <= model['tau'], axis=(1, 2))]
        if distinct_hosts:
            host_counts = model['provider_F'][:, groups].sum(axis=2)
            groups = groups[(host_counts <= 1).all(axis=0)]
        if not len(groups):
            return dict(status='infeasible', hub=int(hub))
        qualifies = survives[groups].sum(axis=1) >= 2
        # A group may recur. Its multiplicity consumes one turn from each member.
        incidence = (members[:, None, None] == groups[None, :, :]).any(axis=2)
        q = model['q'][members]
        result = milp(-qualifies.astype(float), integrality=np.ones(len(groups)),
                      bounds=Bounds(0, np.inf), constraints=LinearConstraint(incidence, q, q),
                      options={'mip_rel_gap': 0})
        if result.status == 2:
            return dict(status='infeasible', hub=int(hub))
        if not result.success:
            raise RuntimeError(result.message)
        selected = np.repeat(groups, np.rint(result.x).astype(int), axis=0)
        schedule[hubs == hub] = selected
        # Every qualifying round consumes two surviving turns. With distinct
        # hosts, at least one of those turns must be outside the busiest survivor host.
        live_turns = q * survives[members]
        upper = live_turns.sum() // 2
        if distinct_hosts:
            upper = min(upper, live_turns.sum() - (model['provider_F'][:, members] @ live_turns).max())
        assert (survives[selected].sum(axis=1) >= 2).sum() <= upper
    metrics(schedule, model)
    if distinct_hosts:
        assert (model['provider_F'][:, schedule].sum(axis=2) <= 1).all()
    counts = np.bincount(survives[schedule].sum(axis=1), minlength=model['N'] + 1)
    return dict(status='optimal', schedule=schedule.tolist(), rounds_by_surviving_proposers=counts.tolist(),
                qualifying_rounds=int(counts[2:].sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True, help='Batch directory of your expanded SCION run')
    args = parser.parse_args()
    results = []
    threshold_results = []
    unavailable = []
    for limit in (20, 11):
        for seed in range(3):
            case = args.batch / 'cases' / f'p3-shared-s{seed}-rtt20' / 'instance.json'
            if not case.is_file():
                parser.error(f'Missing {case}. Supply the batch directory of your expanded SCION run.')
            data = json.loads(case.read_text())
            if data['measurement']['seed'] != seed or data['measurement']['variant'] != 'shared':
                raise ValueError(f'Unexpected schedule input in {case}')
            try:
                model, dominant = instance(data, limit)
            except NoEligibleStart as exc:
                unavailable.append(dict(limit_ms=limit, seed=seed, reason=str(exc)))
                print(f'{limit} ms, seed {seed}: {exc}')
                continue
            if limit == 20:  # The limit used for the recorded outage replays.
                for distinct_hosts in (False, True):
                    solution = two_survivor_schedule(model, dominant, distinct_hosts)
                    threshold_results.append(dict(seed=seed, distinct_hosts=distinct_hosts, **solution))
            local_pairs = np.triu(model['regions'][:, None] == model['regions'], k=1)
            excluded = int(np.count_nonzero(model['cost'][local_pairs] > limit))
            rows = run_comparison(model, dominant, seed)
            # Separately test whether every cap can be met, as required by strict policy.
            candidate, meta = solve_direct(model, strict=True, time_limit=10)
            if meta['status'] not in (0, 1, 2):
                raise RuntimeError(meta['message'])
            assert candidate is None or metrics(candidate, model)['compliant']
            feasible = True if candidate is not None else False if meta['status'] == 2 else None
            for row in rows:
                row.update(limit_ms=limit, excluded_pairs=excluded, local_pairs=int(local_pairs.sum()),
                           strict_feasible=feasible, strict_solver_status=meta['status'])
            results.extend(rows)
    for limit in (20, 11):
        for name in ('transit', 'full'):
            group = [r for r in results if r['limit_ms'] == limit and r['policy'] == name]
            if not group:
                continue
            print(limit, name, group[0]['excluded_pairs'], '/', group[0]['local_pairs'],
                  {k: sum(r[k] for r in group) for k in ('transit_success', 'all_hosts_distinct', 'joint_success')},
                  'median seconds', round(median(r['seconds'] for r in group), 4),
                  'improved starts', sum(r['adopted'] != 'original' for r in group),
                  'strictly feasible instances', sum(r['strict_feasible'] is True for r in group),
                  'unresolved', sum(r['strict_feasible'] is None for r in group))
    output = args.batch / 'report'
    output.mkdir(exist_ok=True)
    (output / 'transit-comparison.json').write_text(json.dumps(results, indent=2) + '\n')
    (output / 'unavailable-transit-cases.json').write_text(json.dumps(unavailable, indent=2) + '\n')
    (output / 'two-survivor-schedules.json').write_text(json.dumps(threshold_results, indent=2) + '\n')
    for row in threshold_results:
        print('Seed', row['seed'], 'distinct hosts', row['distinct_hosts'],
              'rounds with at least two surviving proposers:', row.get('qualifying_rounds', row['status']))


if __name__ == '__main__':
    main()
