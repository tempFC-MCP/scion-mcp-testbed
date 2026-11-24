"""Fixed-allocation delay sweep and paired SCION deployment workloads."""
import argparse
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

import lab
import measured

LIMITS = [20, 60, 100, 140, 180, 220, 260]
PEER_LIMITS = [20, 100, 260]


def solve_groups(instance, policy, seed=0, time_limit=30):
    """Small-instance reference for Equation group-allocation, not a large-n solver."""
    if policy not in ('geo', 'full'):
        raise ValueError('Unknown policy')
    n, N, T = (instance[k] for k in ('n', 'N', 'T'))
    groups = np.array([g for g in combinations(range(n), N)
                       if all(instance['cost'][u, v] <= instance['tau']
                              for u, v in combinations(g, 2))], dtype=int)
    if not len(groups):
        raise ValueError('No communication-eligible groups')
    # Enumerate only in these small deployment references. Larger experiments
    # elsewhere use the direct validator-round formulation and swap heuristic.
    geo = np.array([max(0, instance['g'] - len(set(instance['regions'][g])))
                    for g in groups])
    risk = np.maximum(instance['F'][:, groups].sum(axis=2)
                      - instance['budget'][:, None], 0).sum(axis=0)
    objective = 10 * geo + (risk if policy == 'full' else 0)
    matrix = csc_matrix((np.ones(groups.size),
                        (groups.ravel(), np.repeat(np.arange(len(groups)), N))),
                       shape=(n, len(groups)))
    start = time.perf_counter()
    result = milp(objective.astype(float), integrality=np.ones(len(groups)),
                  bounds=Bounds(np.zeros(len(groups)), np.full(len(groups), T)),
                  constraints=LinearConstraint(matrix, instance['q'], instance['q']),
                  options={'time_limit': time_limit, 'mip_rel_gap': 0})
    if result.x is None or np.max(np.abs(result.x - np.rint(result.x))) > 1e-5:
        raise RuntimeError(f'No integral schedule: {result.message}')
    counts = np.rint(result.x).astype(int)
    schedule = np.repeat(groups, counts, axis=0)
    np.random.default_rng(seed).shuffle(schedule)
    stats = measured.metrics(schedule, instance)
    value = 10 * stats['geo_deficit'] + (stats['excess'] if policy == 'full' else 0)
    if abs(value - result.fun) > 1e-5:
        raise AssertionError('Expanded objective disagrees with solver')
    return dict(method=policy, schedule=schedule.tolist(), **stats,
                seconds=time.perf_counter() - start,
                solver=dict(status=int(result.status), message=result.message,
                            optimal=result.status == 0, objective=value,
                            dual_bound=float(result.mip_dual_bound), gap=float(result.mip_gap),
                            eligible_groups=len(groups)))


def code_hashes():
    names = ['coordination.py', 'peer_exchange.py', 'lab.py', 'network.py',
             'measured.py', 'replay.py', 'batch.py', 'cmd/mcp-traffic/main.go',
             '../../scripts/scheduler.py']
    return lab.code_hashes(names)


def prepare(snapshot, out):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'cases').mkdir()
    observations = measured.load_snapshot(snapshot)
    summary, tasks, peer_tasks = [], [], []
    for seed in (0, 1, 2):
        base = measured.build_instance(observations, variant='shared', seed=seed, N=3,
                                       tau=20, allocation_tau=20,
                                       service_rtt_limit=500, T=32)
        for limit in LIMITS:
            instance = deepcopy(base)
            instance['tau'] = limit
            name = f'seed-{seed}-rtt-{limit}'
            case = out / 'cases' / name
            case.mkdir()
            measured.write(case / 'instance.json', instance)
            rows = [solve_groups(instance, policy, seed) for policy in ('geo', 'full')]
            measured.write(case / 'schedules.json', rows)
            for row in rows:
                a = np.array(row['schedule'])
                regions = [len(set(instance['regions'][g])) for g in a]
                exposure = instance['F'][:, a].sum(axis=2)
                core = instance['scenarios'].index(['AS:1-ff00:0:110'])
                receivers = {'AS:1-ff00:0:301', 'AS:1-ff00:0:302'}
                relevant = [i for i, s in enumerate(instance['scenarios'])
                            if not set(s) & receivers]
                summary.append(dict(seed=seed, rtt_limit_ms=limit, policy=row['method'],
                    mean_regions=row['mean_regions'], excess=row['excess'],
                    full_coverage_rounds=regions.count(3), rounds=instance['T'],
                    eligible_groups=row['solver']['eligible_groups'],
                    objective=row['solver']['objective'], optimal=row['solver']['optimal'],
                    dual_bound=row['solver']['dual_bound'],
                    max_nonreceiver_exposure=int(exposure[relevant].max()),
                    core110_loss_histogram=np.bincount(exposure[core], minlength=4).tolist(),
                    regional_allocations=[int(instance['q'][instance['regions'] == r].sum())
                                          for r in range(3)]))
                if limit in PEER_LIMITS:
                    peer_tasks.append(dict(id=f'peer-{name}-{row["method"]}',
                                           case=name, method=row['method']))
                if seed == 0 and limit in PEER_LIMITS:
                    for fault, label in [('', 'healthy'), ('1-ff00:0:110', 'outage')]:
                        tasks.append(dict(id=f'payload-{name}-{row["method"]}-{label}',
                                          case=name, method=row['method'], fault=fault,
                                          routing='surviving', deadline_ms=750))
            print(f'Prepared {name}: coverage {[round(r["mean_regions"],3) for r in rows]}', flush=True)
    measured.write(out / 'sweep.json', summary)
    measured.write(out / 'peer-plan.json', dict(tasks=peer_tasks, payload_bytes=512,
                   interval_ms=500, deadline_ms=400, start_skew_limit_ms=5,
                   scope='all-to-all transport measurements, not consensus execution'))
    measured.write(out / 'plan.json', dict(tasks=tasks, code_sha256=code_hashes(),
                   interval_ms=1000, warmup_payloads=1, payload_bytes=262144,
                   snapshot=str(snapshot.resolve()),
                   design='Fixed quotas; measured RTT admission sweep; 3 proposers; 3 allocation seeds'))


def check(out):
    """Reconstruct counts, pair eligibility, and dependency exposure from saved inputs."""
    summaries = json.loads((out / 'sweep.json').read_text())
    counts = {}
    for row in summaries:
        case = out / 'cases' / f'seed-{row["seed"]}-rtt-{row["rtt_limit_ms"]}'
        x = measured.load_instance(case / 'instance.json')
        selected = next(r for r in json.loads((case / 'schedules.json').read_text())
                        if r['method'] == row['policy'])
        q = [0] * x['n']
        penalty, coverage = 0, []
        hist = [0] * 4
        failed = x['scenarios'].index(['AS:1-ff00:0:110'])
        assert len(selected['schedule']) == x['T']
        for group in selected['schedule']:
            assert len(group) == len(set(group)) == x['N']
            for v in group:
                q[v] += 1
            assert all(x['cost'][u, v] <= x['tau'] for u in group for v in group)
            coverage.append(len({int(x['regions'][v]) for v in group}))
            for s, budget in enumerate(x['budget']):
                penalty += max(0, sum(bool(x['F'][s, v]) for v in group) - int(budget))
            hist[sum(bool(x['F'][failed, v]) for v in group)] += 1
        assert q == x['q'].tolist()
        assert counts.setdefault(row['seed'], q) == q
        assert penalty == row['excess']
        assert sum(coverage) / len(coverage) == row['mean_regions']
        assert hist == row['core110_loss_histogram']
        assert coverage.count(3) == row['full_coverage_rounds']
    print(f'Independently checked {len(summaries)} schedules with fixed quotas.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check'])
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    lab.select_profile('research')
    if args.action == 'prepare':
        if args.snapshot is None:
            parser.error('prepare requires --snapshot')
        prepare(args.snapshot, args.out)
    check(args.out)


if __name__ == '__main__':
    main()
