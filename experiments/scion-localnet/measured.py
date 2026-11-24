"""Convert frozen SCION observations into the paper's scheduler input format."""
import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

import lab
from network import CORES, LEAVES, RECEIVERS, sequence_ases

sys.path.insert(0, str(lab.ROOT.parents[1] / 'scripts'))
from scheduler import catalogue, local_search, metrics, normalize_scion_path, quotas, solve_direct


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def write(path, value):
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + '\n')


@dataclass(frozen=True)
class Snapshot:
    directory: Path
    digest: str
    manifest: dict
    paths: dict
    probes: dict


def load_snapshot(directory):
    """Verify a frozen observation directory once per experiment preparation."""
    if isinstance(directory, Snapshot):
        return directory
    manifest_bytes = (directory / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if 'completed_at' not in manifest:
        raise ValueError('Incomplete observation collection')
    lab.verify_files(directory, manifest['files_sha256'])
    selected = json.loads((directory / 'selected-paths.json').read_text())
    paths = {}
    for r in selected:
        ases = sequence_ases(r['path'])
        if len(ases) != 3 or ases[0] != r['source'] or ases[-1] != r['destination'] or ases[1] not in CORES:
            raise ValueError('Unexpected path shape')
        if r['path'].get('status') != 'alive':
            raise ValueError('Unprobed/dead path cannot be accepted')
        key = (r['source'], r['destination'], ases[1])
        if key in paths:
            raise ValueError('Ambiguous selected path')
        paths[key] = r['path']
    probes = {}
    for line in (directory / 'probes.jsonl').read_text().splitlines():
        r = json.loads(line)
        data = r['data']
        path = data['path']
        ases = sequence_ases(path)
        core = ases[1] if len(ases) == 3 else None
        key = (r['source'], r['destination'], core)
        if core is None and r['source'] != r['destination']:
            raise ValueError('Unexpected probe path')
        if core is not None:
            # The pinned showpaths truncates to 16 hex characters; ping emits
            # the full fingerprint. Also compare every AS/interface hop.
            selected_path = paths[key]
            if (not path['fingerprint'].startswith(selected_path['fingerprint'])
                    or selected_path['hops'] != path['hops']
                    or selected_path['sequence'] != path['sequence']):
                raise ValueError('Probe did not use the selected path')
            selected_path['fingerprint'] = path['fingerprint']
        replies = data.get('replies') or []
        unique = {x['scmp_seq']: x for x in replies if x['state'] in ('success', 'Out of Order', 'out of order')}
        valid = (data['statistics']['sent'] == r['count'] == manifest['probe_count']
                 and len(unique) == r['count']
                 and data['statistics']['received'] == r['count'])
        samples = [x['round_trip_time'] for x in unique.values()]
        if any(not np.isfinite(x) or x < 0 for x in samples):
            raise ValueError('Invalid RTT sample')
        if key in probes:
            raise ValueError('Duplicate probe batch')
        probes[key] = dict(valid=valid, p95_ms=float(np.percentile(samples, 95)) if valid else None,
                           fingerprint=path['fingerprint'])
    expected = {(a, b, c) for a in LEAVES for b in LEAVES + RECEIVERS if a != b for c in CORES}
    expected |= {(a, a, None) for a in LEAVES}
    if set(probes) != expected or set(paths) != expected - {(a, a, None) for a in LEAVES}:
        raise ValueError('Incomplete path/probe coverage')
    return Snapshot(directory.resolve(), hashlib.sha256(manifest_bytes).hexdigest(),
                    manifest, paths, probes)


def read_snapshot(directory):
    snapshot = load_snapshot(directory)
    return snapshot.manifest, snapshot.paths, snapshot.probes


def fastest_peer_path(probes, source, destination):
    cores = (None,) if source == destination else CORES
    keys = ((source, destination, core) for core in cores)
    return min((k for k in keys if probes[k]['valid']),
               key=lambda k: (probes[k]['p95_ms'], str(k)), default=None)


def proposer_costs(validators, probes, missing_cost=None):
    """Admission uses the slower of the two fastest directional p95 RTTs."""
    ases = {v['as_id'] for v in validators}
    directional = {}
    for source in ases:
        for destination in ases:
            key = fastest_peer_path(probes, source, destination)
            if key is None and missing_cost is None:
                raise ValueError(f'No measured path for {source} to {destination}')
            directional[source, destination] = None if key is None else probes[key]['p95_ms']

    costs = np.zeros((len(validators), len(validators)))
    for i, u in enumerate(validators):
        for j in range(i):
            a, b = u['as_id'], validators[j]['as_id']
            forward, reverse = directional[a, b], directional[b, a]
            if forward is None or reverse is None:
                costs[i, j] = costs[j, i] = missing_cost
            else:
                costs[i, j] = costs[j, i] = max(forward, reverse)
    return costs


def build_instance(directory, variant='shared', seed=0, N=4, tau=180,
                   allocation_tau=50, service_rtt_limit=180, T=64, h=None):
    if variant not in ('balanced', 'shared', 'backup') or not 2 <= N <= 24 or T < 1:
        raise ValueError('Invalid variant or schedule dimensions')
    if not 0 < allocation_tau <= tau or service_rtt_limit <= 0 or (h is not None and not 1 <= h <= T):
        raise ValueError('Invalid service limits')
    snapshot = load_snapshot(directory)
    manifest, paths, probes = snapshot.manifest, snapshot.paths, snapshot.probes
    validators = manifest['validators']
    n = len(validators)
    if n != 24 or len({v['id'] for v in validators}) != n:
        raise ValueError('Expected the declared 24 distinct validator identities')
    # The pilot encodes an excluded pair as a finite value in the saved matrix.
    excluded = 1e9
    cost = proposer_costs(validators, probes, missing_cost=excluded)
    rng = np.random.default_rng(seed)
    weights = rng.lognormal(0, 1, n)
    initial = []
    for _ in range(T):
        chosen = []
        for _ in range(N):
            candidates = [v for v in range(n) if v not in chosen
                          and all(cost[v, u] <= allocation_tau for u in chosen)]
            if not candidates:
                raise ValueError('Declared allocation limit cannot seed a group; no hidden resampling')
            probability = weights[candidates] / weights[candidates].sum()
            chosen.append(int(rng.choice(candidates, p=probability)))
        initial.append(chosen)
    initial = np.array(initial)
    permutation = np.random.default_rng(seed + 9000).permutation(n)
    assignments = np.repeat(np.arange(3), [8, 8, 8] if variant == 'balanced' else [12, 6, 6])
    primary = np.empty(n, dtype=int)
    primary[permutation] = assignments
    epoch_end = time.time() + T + 120
    records, routes = [], []
    for i, v in enumerate(validators):
        services, selected = [], []
        core_indices = [primary[i]]
        if variant == 'backup':
            core_indices.append((primary[i] + 1) % len(CORES))
        for dest in RECEIVERS:
            service, entries = [], []
            for index in core_indices:
                key = (v['as_id'], dest, CORES[index])
                p, observation = paths[key], probes[key]
                expires = datetime.fromisoformat(p['expiry'].replace('Z', '+00:00')).timestamp()
                if not observation['valid'] or observation['p95_ms'] > service_rtt_limit:
                    raise ValueError(f'Ineligible declared service path {key}: RTT={observation["p95_ms"]}, limit={service_rtt_limit}')
                if expires < epoch_end:
                    raise ValueError('Snapshot paths expire before the execution horizon')
                record_id = f'{v["as_id"]}/{dest}/{p["fingerprint"]}'
                normalized = normalize_scion_path(dict(record_id=record_id,
                    as_entries=[dict(isd_as=a) for a in sequence_ases(p)],
                    expires=expires, latency_ms=observation['p95_ms']), {record_id})
                normalized['record_id'] = record_id
                service.append(normalized)
                entries.append(dict(receiver=dest, core=CORES[index], fingerprint=p['fingerprint'],
                                    sequence=p['sequence'], expiry=p['expiry'],
                                    probe_p95_ms=observation['p95_ms']))
            services.append(service)
            selected.append(entries)
        records.append(dict(mandatory=['host:' + v['provider']], services=services))
        routes.append(dict(validator=v['id'], services=selected))
    scenarios, F = catalogue(records, epoch_end=epoch_end,
                             latency_ms=service_rtt_limit, max_order=1)
    scenario_lists = [sorted(s) for s in scenarios]
    provider_mask = np.array([next(iter(s)).startswith('host:') for s in scenarios])
    receiver_mask = np.array([next(iter(s)) in {'AS:' + a for a in RECEIVERS} for s in scenarios])
    budgets = np.where(receiver_mask, N, max(1, N // 3))
    regions = np.array([v['region'] for v in validators])
    return dict(n=n, N=N, T=T, regions=regions, nr=3, g=min(N, 3),
                F=F, budget=budgets, provider_F=F[provider_mask],
                provider_budget=budgets[provider_mask], cost=cost, tau=tau,
                initial=initial, q=quotas(initial, n), h=h, records=records,
                scenarios=scenario_lists, validators=validators, routes=routes,
                measurement=dict(source=str(snapshot.directory), variant=variant, seed=seed,
                    snapshot_completed_at=manifest['completed_at'], epoch_end=epoch_end,
                    allocation_tau_rtt_ms=allocation_tau, service_rtt_limit_ms=service_rtt_limit,
                    admission_policy='max of directional minimum p95 single-core probe RTT',
                    excluded_pair_sentinel=excluded, snapshot_hash=snapshot.digest))


def schedule(instance):
    rows = []
    for mode in ['random', 'latency', 'geo', 'provider', 'full', 'MILP']:
        meta = {}
        if mode == 'random':
            result, elapsed = instance['initial'], 0.
        elif mode == 'MILP':
            result, meta = solve_direct(instance, time_limit=5)
            elapsed = meta['total_seconds']
        else:
            result, elapsed = local_search(instance, mode, iterations=4000,
                                           seed=instance['measurement']['seed'] + 100)
        row = dict(method=mode, seconds=elapsed, solver=meta)
        if result is not None:
            row.update(schedule=result.tolist(), **metrics(result, instance))
            if not np.array_equal(instance['F'][:, result].sum(axis=(1, 2)), instance['F'] @ instance['q']):
                raise AssertionError('Exposure accounting identity violated')
        rows.append(row)
    return rows


def load_instance(path):
    data = json.loads(path.read_text())
    for name in ('regions', 'F', 'budget', 'provider_F', 'provider_budget', 'cost', 'initial', 'q'):
        data[name] = np.array(data[name])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', choices=['balanced', 'shared', 'backup'], default='shared')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--proposers', type=int, default=4)
    parser.add_argument('--service-rtt-limit', type=float, default=180)
    parser.add_argument('--tau', type=float, default=180)
    parser.add_argument('--allocation-tau', type=float, default=50)
    parser.add_argument('--window', type=int)
    args = parser.parse_args()
    instance = build_instance(args.snapshot, args.variant, args.seed, args.proposers,
                              service_rtt_limit=args.service_rtt_limit, tau=args.tau,
                              allocation_tau=args.allocation_tau, h=args.window)
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'instance.json', instance)
    rows = schedule(instance)
    write(args.out / 'schedules.json', rows)
    print(json.dumps([{k: r.get(k) for k in ('method', 'excess', 'mean_regions', 'seconds')} for r in rows], indent=2))


if __name__ == '__main__':
    main()
