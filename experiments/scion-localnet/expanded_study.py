"""Reduced, paired evaluation of local and distributed infrastructure-aware schedules."""
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import time

import numpy as np

import lab
import network
import measured
import coordination
import replay
import batch
import peer_exchange
from scheduler import catalogue, metrics, quotas


def instance(snapshot, seed, providers, variant, limit):
    snapshot = measured.load_snapshot(snapshot)
    paths, probes = snapshot.paths, snapshot.probes
    hosts = network.addresses()
    validators = []
    for i in range(36):
        region, local = divmod(i, 12)
        provider = local // (12 // providers)
        ia = network.LEAVES[3 * region + provider]
        validators.append(dict(id=f'v{i:02d}', as_id=ia, region=region,
                               provider=f'H{provider}', local_ip=hosts[ia]))
    cost = measured.proposer_costs(validators, probes)
    rng = np.random.default_rng(seed)
    initial = []
    for _ in range(2):
        for r in range(3):
            initial.extend(rng.permutation(np.arange(12*r, 12*(r+1))).reshape(4, 3).tolist())
    initial = np.array(initial)
    assert np.all(quotas(initial, 36) == 2)
    primary = np.zeros(36, dtype=int)
    route_rng = np.random.default_rng(seed + 500)
    for r in range(3):
        order = route_rng.permutation(np.arange(12*r, 12*(r+1)))
        primary[order] = np.repeat(np.arange(6), 2) if variant == 'separate' else [0]*8 + [1, 2, 3, 4]
    records, routes = [], []
    until = time.time() + 180
    for i, v in enumerate(validators):
        indices = [int(primary[i])]
        if variant == 'backup':
            indices.append(int(primary[i]) ^ 1)
        services, service_routes = [], []
        for receiver in network.RECEIVERS:
            alternatives, entries = [], []
            for index in indices:
                key = v['as_id'], receiver, network.CORES[index]
                path, probe = paths[key], probes[key]
                expiry = datetime.fromisoformat(path['expiry'].replace('Z', '+00:00')).timestamp()
                if not probe['valid'] or probe['p95_ms'] > 500 or expiry <= until:
                    raise ValueError('Declared delivery path fails measurement or lifetime requirement')
                alternatives.append(dict(dependencies=['AS:' + a for a in network.sequence_ases(path)],
                                         accepted=True, expires=expiry, latency_ms=probe['p95_ms']))
                entries.append(dict(receiver=receiver, core=network.CORES[index],
                                    fingerprint=path['fingerprint'], sequence=path['sequence'],
                                    expiry=path['expiry'], probe_p95_ms=probe['p95_ms']))
            services.append(alternatives)
            service_routes.append(entries)
        records.append(dict(mandatory=['host:' + v['provider']], services=services))
        routes.append(dict(validator=v['id'], services=service_routes))
    scenarios, F = catalogue(records, epoch_end=until, latency_ms=500, max_order=1)
    receiver_names = {'AS:' + a for a in network.RECEIVERS}
    budgets = np.array([3 if set(s) & receiver_names else 1 for s in scenarios])
    provider_mask = np.array([any(d.startswith('host:') for d in s) for s in scenarios])
    return dict(n=36, N=3, T=24, nr=3, g=1 if limit == 20 else 3, h=None,
                regions=np.array([v['region'] for v in validators]), validators=validators,
                F=F, budget=budgets, provider_F=F[provider_mask], provider_budget=budgets[provider_mask],
                cost=cost, tau=limit, initial=initial, q=quotas(initial, 36), records=records,
                routes=routes, scenarios=[sorted(s) for s in scenarios],
                measurement=dict(source=str(snapshot.directory), variant=variant, seed=seed,
                                 providers_per_region=providers, epoch_end=until,
                                 admission_policy='max of directional minimum p95 SCMP RTT',
                                 snapshot_hash=snapshot.digest))


def schedules(x, seed):
    if x['tau'] == 20:
        base = dict(method='base', schedule=x['initial'].tolist(), **metrics(x['initial'], x))
    else:
        base = coordination.solve_groups(x, 'geo', seed)
        base['method'] = 'base'
    full = coordination.solve_groups(x, 'full', seed)
    full['method'] = 'infra'
    if x['tau'] == 20:
        # Preserve the base hub sequence as well as individual allocations.
        by_region = {r: [] for r in range(3)}
        for group in full['schedule']:
            regions = {int(x['regions'][v]) for v in group}
            assert len(regions) == 1
            by_region[regions.pop()].append(group)
        full['schedule'] = [by_region[int(x['regions'][group[0]])].pop() for group in x['initial']]
        full.update(metrics(full['schedule'], x))
    return [base, full]


def prepare(snapshot, out):
    out.mkdir(parents=True, exist_ok=False)
    (out/'cases').mkdir()
    observations = measured.load_snapshot(snapshot)
    rows, tasks, peers = [], [], []
    settings = [(2, 'shared', 20), (3, 'shared', 20), (3, 'shared', 100),
                (3, 'shared', 180), (3, 'shared', 260),
                (3, 'separate', 20), (3, 'backup', 20)]
    for seed in range(3):
        for providers, variant, limit in settings:
            name = f'p{providers}-{variant}-s{seed}-rtt{limit}'
            case = out/'cases'/name
            case.mkdir()
            x = instance(observations, seed, providers, variant, limit)
            selected = schedules(x, seed)
            if seed == 0 and (providers, variant, limit) == (3, 'shared', 20):
                hidden = deepcopy(x)
                hidden['F'], hidden['budget'] = x['provider_F'], x['provider_budget']
                hidden['scenarios'] = [s for s in x['scenarios']
                                       if any(dep.startswith('host:') for dep in s)]
                row = coordination.solve_groups(hidden, 'full', seed)
                by_region = {r: [] for r in range(3)}
                for group in row['schedule']:
                    by_region[int(x['regions'][group[0]])].append(group)
                row['schedule'] = [by_region[int(x['regions'][g[0]])].pop() for g in x['initial']]
                row.update(method='hosting', **metrics(row['schedule'], x))
                selected.append(row)
            measured.write(case/'instance.json', x)
            measured.write(case/'schedules.json', selected)
            for s in selected:
                exposure = x['F'][:, np.array(s['schedule'])].sum(axis=2)
                hosting = x['provider_F'][:, np.array(s['schedule'])].sum(axis=2)
                counts = [len({x['validators'][v]['provider'] for v in g}) for g in s['schedule']]
                rows.append(dict(case=name, seed=seed, providers=providers, variant=variant,
                    rtt_limit=limit, method=s['method'], mean_regions=s['mean_regions'],
                    excess=s['excess'], hosting_worst=int(hosting.max()),
                    single_provider_rounds=counts.count(1),
                    optimal=s.get('solver', {}).get('optimal'),
                    all_limits_rounds=int((exposure <= x['budget'][:,None]).all(axis=0).sum())))
                # Repeat the central local/distributed comparison across all seeds.
                main = providers == 3 and variant == 'shared' and limit in (20, 260)
                supporting = seed == 0 and limit == 20 and (providers == 2 or variant in ('separate', 'backup'))
                if main or supporting or s['method'] == 'hosting':
                    tasks.append(dict(id=f'exp-{name}-{s["method"]}-outage', case=name,
                                      method=s['method'], fault=network.CORES[0], routing='surviving', deadline_ms=750))
                if seed == 0 and s['method'] == 'infra' and (main or supporting):
                    tasks.append(dict(id=f'exp-{name}-infra-healthy', case=name,
                                      method='infra', fault='', routing='surviving', deadline_ms=750))
                if providers == 3 and variant == 'shared' and limit in (20,100,260) and s['method'] == 'infra':
                    peers.append(dict(id=f'exp-peer-{name}', case=name, method='infra'))
            print('Prepared', name, flush=True)
    names = ['expanded_study.py','coordination.py','peer_exchange.py','network.py','lab.py',
             'measured.py','replay.py','batch.py','cmd/mcp-traffic/main.go','../../scripts/scheduler.py','resume_timing.py']
    measured.write(out/'plan.json', dict(tasks=tasks, snapshot=str(snapshot.resolve()),
        interval_ms=1000, warmup_payloads=1, payload_bytes=262144,
        code_sha256=lab.code_hashes(names)))
    measured.write(out/'peer-plan.json',dict(tasks=peers,payload_bytes=512,interval_ms=500,
        deadline_ms=400,start_skew_limit_ms=5))
    measured.write(out/'sweep.json',rows)
    print('Prepared',len(tasks),'payload runs and',len(peers),'peer runs',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','install','payloads','peers'])
    p.add_argument('--snapshot',type=Path)
    p.add_argument('--out',type=Path)
    args = p.parse_args()
    lab.select_profile('expanded')
    if args.action == 'prepare': prepare(args.snapshot,args.out)
    elif args.action == 'install': replay.install()
    elif args.action == 'payloads': batch.run(args.out)
    else: peer_exchange.run(args.out)
