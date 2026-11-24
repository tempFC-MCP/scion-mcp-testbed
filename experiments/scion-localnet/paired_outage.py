"""Replay matched local schedules against individual provider outages and one transit outage."""
import argparse
import fcntl
import json
from pathlib import Path

import numpy as np

import batch
import expanded_study
import lab
import measured
import network
import replay
import transit_comparison
from scheduler import metrics


def prepare_case(source, directory, seed):
    model, dominant = transit_comparison.instance(source, limit=20)
    x = dict(source)  # Keep SCION endpoints and selected paths with the schedule.
    x.update(model)
    x['original_draw'] = source['initial']
    case = directory / f'seed-{seed}'
    case.mkdir()
    measured.write(case / 'instance.json', x)
    policies = {}
    for method in ('transit', 'full'):
        chosen, score = transit_comparison.select_policy(model, dominant, seed, method)
        policies[method] = dict(method=method, schedule=chosen.tolist(), comparison=score,
                                **metrics(chosen, model))
    measured.write(case / 'schedules.json', list(policies.values()))

    dominant_label, = model['scenarios'][dominant]
    core = dominant_label.removeprefix('AS:')
    providers = sorted({v['provider'] for v in x['validators']})
    if providers != ['H0', 'H1', 'H2'] or core != network.CORES[0]:
        raise ValueError('Unexpected shared-route or hosting assignment in the expanded testbed')
    for provider in providers:
        members = [v for v in x['validators'] if v['provider'] == provider]
        sites = {v['as_id'] for v in members}
        if (len(sites) != 3 or len({v['region'] for v in members}) != 3 or
                any((v['as_id'] in sites) != (v['provider'] == provider) for v in x['validators'])):
            raise ValueError(f'{provider} must occupy one dedicated proposer network in each hub')
    # Rotate the faults across seeds and alternate which policy goes first within each pair.
    faults = [('provider', p) for p in providers] + [('transit', core)]
    tasks = []
    for offset in range(len(faults)):
        index = (offset + seed) % len(faults)
        kind, fault = faults[index]
        label = ('host:' if kind == 'provider' else 'AS:') + fault
        scenario = model['scenarios'].index([label])
        names = ('transit', 'full') if (seed + index) % 2 == 0 else ('full', 'transit')
        for method in names:
            schedule = np.asarray(policies[method]['schedule'])
            survivors = x['N'] - model['F'][scenario][schedule].sum(axis=1)
            token = fault.split(':')[-1]
            tasks.append(dict(id=f's{seed}-{kind}-{token}-{method}', seed=seed, case=case.name,
                              method=method, fault_kind=kind, fault=fault,
                              modeled_survivors=survivors.tolist()))
    return tasks


def prepare(snapshot, out):
    observed = measured.load_snapshot(snapshot)
    out.mkdir(parents=True, exist_ok=False)
    cases = out / 'cases'
    cases.mkdir()
    tasks = []
    for seed in range(3):
        source = expanded_study.instance(observed, seed, 3, 'shared', 20)
        tasks.extend(prepare_case(source, cases, seed))
    files = {str(p.relative_to(out)): lab.file_sha256(p) for p in sorted(cases.glob('*/*.json'))}
    code = ('paired_outage.py', 'expanded_study.py', 'replay.py', 'batch.py', 'lab.py', 'network.py',
            'measured.py', '../../scripts/transit_comparison.py', '../../scripts/scheduler.py',
            'cmd/mcp-traffic/main.go', 'cmd/mcp-traffic/BUILD.bazel', 'versions.json')
    plan = dict(tasks=tasks, snapshot_sha256=observed.digest, case_sha256=files,
                code_sha256=lab.code_hashes(code), payload_bytes=256 * 1024,
                interval_ms=1000, deadline_ms=750, warmup_payloads=1, rtt_limit_ms=20,
                cutoffs_ms=[220, 500, 750], start_skew_limit_ms=5, max_timed_attempts=2)
    measured.write(out / 'plan.json', plan)
    print(f'Prepared {len(tasks)} physical outage replays in {out}', flush=True)


def run(out):
    plan = json.loads((out / 'plan.json').read_text())
    lab.verify_files(lab.ROOT, plan['code_sha256'])
    lab.verify_files(out, plan['case_sha256'])
    runs = out / 'runs'
    runs.mkdir(exist_ok=True)
    results = []
    for number, task in enumerate(plan['tasks'], 1):
        complete = 0
        attempt = 0
        while complete < plan['max_timed_attempts']:
            attempt += 1
            dest = runs / f"{task['id']}-a{attempt}"
            if dest.exists() and not (dest / 'checksums.json').exists():
                print(f'Preserving interrupted attempt {dest.name}', flush=True)
                continue
            if not dest.exists():
                # Every proposer site must be reachable through the restored busiest transit.
                # Replay also warms each actual receiver path with a payload before injecting the fault.
                batch.recovery_check(network.CORES[0])
                print(f"Running {number}/{len(plan['tasks'])}: {dest.name}", flush=True)
                provider = task['fault_kind'] == 'provider'
                replay.execute(out / 'cases' / task['case'], task['method'], dest,
                               failed_core='' if provider else task['fault'],
                               failed_provider=task['fault'] if provider else '',
                               deadline_ms=plan['deadline_ms'], interval_ms=plan['interval_ms'],
                               warmup_payloads=plan['warmup_payloads'], payload_bytes=plan['payload_bytes'])
            lab.verify_recording(dest)
            if not (dest / 'summary.json').exists():
                print(f'Preserving incomplete attempt {dest.name}', flush=True)
                continue
            summary = json.loads((dest / 'summary.json').read_text())
            wanted = [3 - n for n in task['modeled_survivors']]
            if summary['predicted_unavailable'] != wanted:
                raise ValueError(f'Predicted outage differs from frozen plan for {dest.name}')
            complete += 1
            result = dict(task_id=task['id'], attempt=attempt, run_dir=str(dest.relative_to(out)),
                          run=summary['run'], accepted=summary['accepted'],
                          max_start_skew_ms=summary['max_start_skew_ms'])
            results.append(result)
            temporary = out / 'results.tmp'
            measured.write(temporary, results)
            temporary.replace(out / 'results.json')
            print(f"{dest.name}: {summary['proposals_delivered']}/72 delivered; "
                  f"start skew {summary['max_start_skew_ms']:.2f} ms", flush=True)
            if result['accepted']:
                break
        else:
            raise RuntimeError(f"Both completed attempts exceeded the start-time gate for {task['id']}")
    batch.recovery_check(network.CORES[0])
    print(f"Completed {len(plan['tasks'])} paired-policy replays.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    lab.select_profile('expanded')
    out = args.out.resolve()
    if args.action == 'prepare':
        if not args.snapshot:
            parser.error('prepare needs --snapshot from an expanded deployment')
        prepare(args.snapshot.resolve(), out)
    else:
        with (out / '.run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(out)


if __name__ == '__main__':
    main()
