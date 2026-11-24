"""Sequential deployment comparisons with validation gates and resumable progress."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time

import numpy as np

import lab
import measured
import network
import replay


def recovery_check(core):
    """Require forwarding, not merely interface-up, before the next replay."""
    hosts = network.addresses()
    jobs = [(a, b) for a in network.LEAVES for b in network.RECEIVERS]
    until = time.monotonic() + 20
    while True:
        def probe(pair):
            a, b = pair
            try:
                r = network.ping(a, b, hosts, f'{a} {core} {b}', 1)
                return r['data']['statistics']['received'] == 1
            except (RuntimeError, TimeoutError):
                return False
        with ThreadPoolExecutor(max_workers=4) as pool:
            ok = all(list(pool.map(probe, jobs)))
        if ok:
            return
        if time.monotonic() > until:
            raise RuntimeError('Forwarding did not recover on all required service paths')


def prepare(snapshot, out, seeds, repeats, extras):
    out.mkdir(parents=True, exist_ok=False)
    cases = out / 'cases'
    cases.mkdir()
    observations = measured.load_snapshot(snapshot)
    tasks, offline = [], []
    def case(variant, seed, N=4, tau=180, h=None):
        name = f'{variant}-s{seed}-N{N}-tau{tau}-h{h}'
        directory = cases / name
        if directory.exists():
            return name
        ins = measured.build_instance(observations, variant, seed, N,
                                      tau=tau, service_rtt_limit=300, h=h)
        directory.mkdir()
        measured.write(directory / 'instance.json', ins)
        rows = measured.schedule(ins)
        if any('schedule' not in row for row in rows):
            raise RuntimeError('Missing schedule; inspect solver result before replay')
        measured.write(directory / 'schedules.json', rows)
        offline.append(dict(case=name, variant=variant, seed=seed, N=N, tau=tau, h=h,
                            results=[{k:r[k] for k in ('method','seconds','excess','geo_deficit',
                                      'mean_regions','window_misses')} for r in rows]))
        print(f'Prepared {name}', flush=True)
        return name
    for seed in seeds:
        comparison = [case(v, seed) for v in ('balanced','shared','backup')]
        instances = [measured.load_instance(cases / name / 'instance.json') for name in comparison]
        for ins in instances[1:]:
            for k in ('q','initial','cost','regions','provider_F'):
                if not np.array_equal(ins[k], instances[0][k]):
                    raise ValueError(f'Paired {k} differs between routing conditions')
        for repeat in range(repeats):
            block = [dict(case=name, method=method, fault=fault, deadline_ms=500,
                          routing='surviving', seed=seed, repeat=repeat, phase='comparison')
                     for name in comparison for method in ('geo','provider','full')
                     for fault in ('',network.CORES[0])]
            random.Random(7000+100*seed+repeat).shuffle(block)
            tasks.extend(block)
    if extras:
        seed = seeds[0]
        shared, backup = case('shared',seed), case('backup',seed)
        tasks += [dict(case=shared,method='full',fault='',deadline_ms=250,routing='surviving',phase='deadline')]
        tasks += [dict(case=backup,method='full',fault=network.CORES[0],deadline_ms=d,
                       routing='retry',phase='retry') for d in (250,500)]
        for N in (4,8):
            for tau in (50,100,180):
                case('shared',seed,N,tau)
        for h in (1,2,4):
            case('shared',seed,h=h)
        tasks += [dict(case=case('shared',seed,h=4),method='full',fault='',deadline_ms=500,
                       routing='surviving',phase='regional-window')]
        tasks += [dict(case=case('shared',seed,N=8),method='full',fault=f,deadline_ms=500,
                       routing='surviving',phase='eight-proposers') for f in ('',network.CORES[0])]
        tasks += [dict(case=case('shared',seed,N=2),method='full',fault='',deadline_ms=500,
                       routing='surviving',phase='two-proposers-control')]
    for i, task in enumerate(tasks):
        task['id'] = f'{out.name}-{i:03d}'
    plan = dict(snapshot=str(snapshot.resolve()), seeds=seeds, repetitions=repeats,
                extras=extras, interval_ms=1000, warmup_payloads=1,
                tasks=tasks, code_sha256=lab.code_hashes(('batch.py', 'lab.py', 'network.py', 'measured.py',
                                                'replay.py', 'cmd/mcp-traffic/main.go')))
    measured.write(out/'plan.json',plan)
    measured.write(out/'offline-scheduling.json',offline)
    return plan


def run(out):
    plan=json.loads((out/'plan.json').read_text())
    lab.verify_files(lab.ROOT, plan['code_sha256'])
    completed=[]
    for task in plan['tasks']:
        target=out/'runs'/task['id']
        if (target/'summary.json').exists():
            lab.verify_recording(target)
            summary=json.loads((target/'summary.json').read_text())
            if not summary['accepted']:
                raise ValueError('Previously rejected replay needs investigation')
        else:
            if target.exists():
                raise ValueError(f'Incomplete replay {target}; retain its evidence and investigate before restarting')
            if 'link_condition' in task:
                condition = plan['link_conditions'][task['link_condition']]
                actual = json.loads((lab.GEN/'link-profile.json').read_text())
                if any(actual.get(k) != v for k,v in condition.items()):
                    network.shape(**condition)
                    calibration = []
                    hosts = network.addresses()
                    for a,b,c in [(network.LEAVES[0],network.LEAVES[2],network.CORES[0]),
                                  (network.LEAVES[2],network.LEAVES[4],network.CORES[1]),
                                  (network.LEAVES[0],network.LEAVES[4],network.CORES[0])]:
                        sample = network.ping(a,b,hosts,f'{a} {c} {b}',100)
                        if sample['data']['statistics']['received'] != 100:
                            raise RuntimeError('Link calibration lost probes; preserve evidence and investigate')
                        calibration.append(sample)
                    measured.write(out/f'calibration-{task["id"]}.json',calibration)
            print(f'RUN {len(completed)+1}/{len(plan["tasks"])} {task}',flush=True)
            measured.write(out/'progress.json',dict(status='running',completed=len(completed),
                           total=len(plan['tasks']),current=task,updated_at=time.time()))
            summary=replay.execute(out/'cases'/task['case'],task['method'],target,task['fault'],
                                   task['routing'],plan['interval_ms'],task['deadline_ms'],plan['warmup_payloads'],
                                   task.get('payload_bytes',plan.get('payload_bytes',262144)))
            if not summary['accepted']:
                raise RuntimeError(f'Start-skew gate failed for {task["id"]}; run preserved')
            if summary['prediction_confusion']['predicted_True_delivered_True']:
                raise RuntimeError('A structurally disconnected proposer delivered; inspect the routing/catalogue')
            if task['fault']:
                recovery_check(task['fault'])
        completed.append(dict(task=task,summary=summary))
        measured.write(out/'results.json',completed)
        print(f'PASS {len(completed)}/{len(plan["tasks"])} delivered={summary["proposals_delivered"]}/'
              f'{summary["proposal_attempts"]} predicted_excess={summary["predicted_excess"]} '
              f'observed_excess={summary["observed_excess"]}',flush=True)
    measured.write(out/'progress.json',dict(status='complete',completed=len(completed),
                   total=len(plan['tasks']),updated_at=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seeds',default='0')
    parser.add_argument('--repetitions',type=int,default=1)
    parser.add_argument('--no-extras',action='store_true')
    args=parser.parse_args();lab.select_profile('research')
    if args.action=='prepare':
        if args.snapshot is None or args.repetitions<1:parser.error('prepare needs snapshot and positive repetitions')
        seeds=[int(s) for s in args.seeds.split(',')]
        if len(set(seeds))!=len(seeds):parser.error('seeds must be distinct')
        plan=prepare(args.snapshot,args.out,seeds,args.repetitions,not args.no_extras)
        print(f'Prepared {len(plan["tasks"])} replays. Epoch time alone: {len(plan["tasks"])*64/60:.1f} minutes; warmup adds time.')
    else:
        try:run(args.out)
        except Exception as exc:
            measured.write(args.out/'failure.json',dict(error=str(exc),at=time.time()))
            raise


if __name__=='__main__':main()
