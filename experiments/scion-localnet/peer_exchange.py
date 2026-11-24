"""Measure small messages between scheduled proposers over frozen SCION paths."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

import lab
import measured
import network


def routes(snapshot):
    _, paths, probes = measured.read_snapshot(snapshot)
    hosts = network.addresses()
    result = {}
    for a in network.LEAVES:
        for b in network.LEAVES:
            key = measured.fastest_peer_path(probes, a, b)
            if key is None:
                raise ValueError(f'No measured path for {a} to {b}')
            p = paths[key] if a != b else None
            result[a, b] = dict(remote=f'{b},{hosts[b]}:4445',
                               sequence=p['sequence'] if p else '',
                               fingerprint=probes[key]['fingerprint'], core=key[2] or '',
                               probe_p95_rtt_ms=probes[key]['p95_ms'],
                               expiry=p['expiry'] if p else None)
    return result


def install_servers():
    hosts = network.addresses()
    pattern = '^/app/mcp-agent --mode serve --daemon .* --listen .*:4445 '
    for ia in network.LEAVES:
        if network.docker(ia, 'pgrep', '-f', pattern, check=False).returncode == 0:
            continue
        lab.run(['docker', 'exec', '-d', network.container(ia), '/app/mcp-agent',
                 '--mode', 'serve', '--daemon', hosts[ia] + ':30255',
                 '--listen', hosts[ia] + ':4445', '--log', '/etc/scion/peer-receipts.jsonl'])


def verify(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    jobs = json.loads((out / 'jobs.json').read_text())
    senders = [json.loads(line) for line in (out / 'senders.jsonl').read_text().splitlines()]
    receipts = [json.loads(line) for line in (out / 'receipts.jsonl').read_text().splitlines()]
    expected = {(j['round'], j['validator']): j for j in jobs}
    starts = [r for r in senders if r['kind'] == 'start']
    if len(starts) != len(expected) or {(r['round'], r['validator']) for r in starts} != set(expected):
        raise AssertionError('Missing, duplicate or unexpected message starts')
    digest = hashlib.sha256(bytes([0xa5]) * manifest['payload_bytes']).hexdigest()
    earliest, starts_by_key = {}, {}
    skew = 0
    for r in starts:
        origin = manifest['start_ns'] + r['round'] * manifest['interval_ns']
        assert r['planned_ns'] == origin and r['actual_ns'] >= origin and r['run'] == manifest['run']
        starts_by_key[r['round'], r['validator']] = r['actual_ns']
        skew = max(skew, (r['actual_ns'] - origin) / 1e6)
    for r in receipts:
        key = int(r['round']), r['validator']
        assert key in expected and r['run'] == manifest['run']
        job = expected[key]
        route = job['targets'][0][0]
        assert r['receiver'] == job['destination_as'] and r['path'] == route['fingerprint']
        if not r['verified']:
            continue
        assert r['bytes'] == manifest['payload_bytes'] and r['hash'] == digest
        assert r['complete_ns'] >= starts_by_key[key]
        earliest[key] = min(earliest.get(key, r['complete_ns']), r['complete_ns'])
    attempts = [r for r in senders if r['kind'] == 'attempt']
    assert len(attempts) == len(expected)
    assert {(r['round'], r['validator']) for r in attempts} == set(expected)
    for r in attempts:
        route = expected[r['round'], r['validator']]['targets'][0][0]
        assert r['path'] == route['fingerprint'] and r['receiver'] == route['remote']
        assert r['application_bytes_offered'] == manifest['payload_bytes']
    completion, messages_by_own_start = [], []
    deadline_ns = manifest['deadline_ns']
    for t in range(manifest['rounds']):
        keys = [key for key in expected if key[0] == t]
        origin = manifest['start_ns'] + t * manifest['interval_ns']
        if all(k in earliest and earliest[k] <= origin + deadline_ns for k in keys):
            completion.append(max((earliest[k] - origin) / 1e6 for k in keys))
        else:
            completion.append(None)
    for key, when in earliest.items():
        messages_by_own_start.append((when - starts_by_key[key]) / 1e6)
    by_deadline = {str(d): sum(x is not None and x <= d for x in completion)
                   for d in [10, 20, 50, 100, 150, 200, 250, 300, 400]}
    values = [x for x in completion if x is not None]
    return dict(run=manifest['run'], rounds=manifest['rounds'], expected_messages=len(expected),
                received_messages=len(earliest), all_message_rounds=len(values),
                complete_rounds_by_ms=by_deadline, group_completion_ms=completion,
                message_p95_ms=float(np.percentile(messages_by_own_start, 95)) if messages_by_own_start else None,
                group_p95_ms=float(np.percentile(values, 95)) if values else None,
                max_start_skew_ms=skew, accepted=skew <= manifest['start_skew_limit_ms'],
                sender_acknowledgements=sum(bool(r['acknowledged']) for r in attempts))


def execute(case, method, out, selected_routes, plan):
    x = measured.load_instance(case / 'instance.json')
    selected = next(r for r in json.loads((case / 'schedules.json').read_text()) if r['method'] == method)
    measured.metrics(selected['schedule'], x)
    out.mkdir(parents=True, exist_ok=False)
    run = lab.new_run_id(out.name)
    hosts = network.addresses()
    jobs = []
    for t, group in enumerate(selected['schedule']):
        for u in group:
            for v in group:
                if u == v:
                    continue
                a, b = x['validators'][u], x['validators'][v]
                route = selected_routes[a['as_id'], b['as_id']]
                if route['probe_p95_rtt_ms'] > x['tau']:
                    raise ValueError('Selected path does not satisfy admission limit')
                if route['expiry'] and datetime.fromisoformat(route['expiry'].replace('Z', '+00:00')).timestamp() < time.time() + 180:
                    raise ValueError('Peer path expires before run finishes')
                jobs.append(dict(round=t, validator=f'{a["id"]}-to-{b["id"]}',
                                 source_as=a['as_id'], destination_as=b['as_id'], targets=[[route]]))
    manifest = dict(run=run, task_id=out.name, method=method, case=str(case.resolve()),
                    rounds=x['T'], proposers=x['N'], payload_bytes=plan['payload_bytes'],
                    interval_ns=int(plan['interval_ms'] * 1e6),
                    deadline_ns=int(plan['deadline_ms'] * 1e6),
                    start_skew_limit_ms=plan['start_skew_limit_ms'],
                    started_at=datetime.now(timezone.utc).isoformat(),
                    admission_rtt_ms=x['tau'], routing='surviving', failed_core='',
                    warmup_payloads=1, primary_attempt_ns=150_000_000,
                    agent=json.loads((lab.GEN / 'agent-install.json').read_text()),
                    runtime_link_profile=json.loads((lab.GEN / 'link-profile.json').read_text()),
                    case_sha256=lab.file_sha256(case / 'instance.json'),
                    schedule_sha256=lab.file_sha256(case / 'schedules.json'))
    measured.write(out / 'jobs.json', jobs)
    configs, processes = [], []
    try:
        for ia in network.LEAVES:
            local = network.host_config(ia).parent / 'runs' / run
            local.mkdir(parents=True, exist_ok=False)
            remote = '/etc/scion/runs/' + run
            cfg = dict(manifest, daemon=hosts[ia] + ':30255',
                       jobs=[j for j in jobs if j['source_as'] == ia],
                       start_file=remote + '/start.json', output=remote + '/sender.jsonl',
                       ready=remote + '/ready.json')
            measured.write(local / 'agent.json', cfg)
            stream = (out / f'agent-{ia.split(":")[-1]}.log').open('w')
            proc = subprocess.Popen(['docker', 'exec', network.container(ia), '/app/mcp-agent',
                                     '--mode', 'replay', '--config', remote + '/agent.json'],
                                    stdout=stream, stderr=stream)
            configs.append(local)
            processes.append((ia, proc, stream))
        until = time.monotonic() + 90
        while not all((p / 'ready.json').exists() for p in configs):
            if any(p.poll() is not None and p.returncode != 0 for _, p, _ in processes):
                raise RuntimeError('Peer warmup failed; inspect agent logs')
            if time.monotonic() > until:
                raise TimeoutError('Peer agents not ready')
            time.sleep(.1)
        manifest['start_ns'] = time.monotonic_ns() + 3_000_000_000
        measured.write(out / 'manifest.json', manifest)
        for local in configs:
            measured.write(local / 'start.json', dict(start_ns=manifest['start_ns']))
        for _, proc, stream in processes:
            if proc.wait(timeout=x['T'] * plan['interval_ms'] / 1000 + 30):
                raise RuntimeError('Peer replay failed')
            stream.close()
        logs = [json.loads(line) for p in configs for line in (p / 'sender.jsonl').read_text().splitlines()]
        receipts = [r for ia in network.LEAVES
                    for r in map(json.loads, (network.host_config(ia).parent / 'peer-receipts.jsonl').read_text().splitlines())
                    if r['run'] == run]
        (out / 'senders.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in logs))
        (out / 'receipts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in receipts))
        summary = verify(out)
        measured.write(out / 'summary.json', summary)
        return summary
    finally:
        for ia, proc, stream in processes:
            if proc.poll() is None:
                pattern = '^/app/mcp-agent --mode replay --config /etc/scion/runs/' + run + '/agent.json$'
                network.docker(ia, 'pkill', '-f', pattern, check=False)
                proc.terminate()
                proc.wait(timeout=10)
            stream.close()
        measured.write(out / 'checksums.json', {p.name: lab.file_sha256(p)
                       for p in out.iterdir() if p.is_file() and p.name != 'checksums.json'})


def run(out):
    overall = json.loads((out / 'plan.json').read_text())
    lab.verify_files(lab.ROOT, overall['code_sha256'])
    plan = json.loads((out / 'peer-plan.json').read_text())
    selected_routes = routes(Path(overall['snapshot']))
    install_servers()
    completed = []
    for task in plan['tasks']:
        dest = out / 'peers' / task['id']
        print(f'PEER {len(completed)+1}/{len(plan["tasks"])} {task["id"]}', flush=True)
        if (dest / 'summary.json').exists():
            lab.verify_recording(dest)
            summary = verify(dest)
        else:
            summary = execute(out / 'cases' / task['case'], task['method'], dest, selected_routes, plan)
        if not summary['accepted']:
            raise RuntimeError('Peer timing gate failed; retained measurement needs investigation')
        completed.append(dict(task=task, summary=summary))
        measured.write(out / 'peer-results.json', completed)
        print(f'PASS messages={summary["received_messages"]}/{summary["expected_messages"]} '
              f'group-p95={summary["group_p95_ms"]:.3f}ms', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['run', 'verify'])
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    lab.select_profile('research')
    if args.action == 'run':
        run(args.out)
    else:
        print(json.dumps(verify(args.out), indent=2))
