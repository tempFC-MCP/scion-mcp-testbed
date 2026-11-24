"""Execute and verify fixed schedules using HTTP/3 proposal transfers over SCION."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

import lab
import measured
import network


def build_agent():
    lab.verify_source()
    dest = lab.SOURCE / 'local_mcp_agent'
    dest.mkdir(exist_ok=True)
    for name in ['main.go', 'BUILD.bazel']:
        shutil.copyfile(lab.ROOT / 'cmd' / 'mcp-traffic' / name, dest / name)
    shutil.copyfile(lab.SOURCE / 'testing/clab/cmd/e2e_http/scion.go', dest / 'scion.go')
    lab.bazel('build', '//local_mcp_agent:agent')


def install():
    binary = lab.SOURCE / 'bazel-bin/local_mcp_agent/agent_/agent'
    hosts = network.addresses()
    if any(network.docker(ia, 'pgrep', '-f', '^/app/mcp-agent --mode replay ', check=False).returncode == 0
           for ia in network.LEAVES):
        raise RuntimeError('Refusing to replace agents while a proposer process is running')
    # A previous peer-timing experiment can leave listeners on the proposer hosts.
    for ia in network.LEAVES + network.RECEIVERS:
        network.docker(ia, 'pkill', '-f', '^/app/mcp-agent --mode serve ', check=False)
    clocks = []
    for ia in network.LEAVES + network.RECEIVERS:
        lab.run(['docker', 'cp', binary, network.container(ia) + ':/app/mcp-agent'])
        before = time.monotonic_ns()
        reading = int(network.docker(ia, '/app/mcp-agent', '--mode', 'clock').stdout)
        after = time.monotonic_ns()
        if not before <= reading <= after:
            raise ValueError('Host/container monotonic clocks are not comparable')
        clocks.append(dict(as_id=ia, before_ns=before, container_ns=reading, after_ns=after))
    for ia in network.RECEIVERS:
        existing = network.docker(ia, 'pgrep', '-x', 'mcp-agent', check=False)
        if existing.returncode:
            lab.run(['docker', 'exec', '-d', network.container(ia), '/app/mcp-agent',
                     '--mode', 'serve', '--daemon', hosts[ia] + ':30255',
                     '--listen', hosts[ia] + ':4444', '--log', '/etc/scion/proposal-receipts.jsonl'])
    measured.write(lab.GEN / 'agent-install.json', dict(
        binary_sha256=lab.file_sha256(binary), clocks=clocks,
        transport=json.loads(network.docker(network.LEAVES[0], '/app/mcp-agent',
                                           '--mode', 'settings').stdout)))


def fault_links(instance, profile, core, provider):
    """Resolve one modeled failure onto the data interfaces of its emulated ASes."""
    if core and provider:
        raise ValueError('Choose one transit or hosting outage per replay')
    if core:
        if core not in network.CORES:
            raise ValueError(f'Unknown transit AS: {core}')
        ases = [core]
    elif provider:
        validators = instance['validators']
        ases = sorted({v['as_id'] for v in validators if v['provider'] == provider})
        if not ases or not set(ases).issubset(network.LEAVES):
            raise ValueError(f'No declared proposer networks for {provider}')
        if any(v['as_id'] in ases and v['provider'] != provider for v in validators):
            raise ValueError('A selected AS also hosts validators assigned to another provider')
        scenario = instance['scenarios'].index(['host:' + provider])
        if list(instance['F'][scenario]) != [v['as_id'] in ases for v in validators]:
            raise ValueError('Hosting outage would disconnect a different set than the model predicts')
    else:
        return []

    links = []
    for ia in ases:
        rows = [r for r in profile['interfaces'] if r['as_id'] == ia]
        peers = set(network.CORES)
        if core:
            peers = (peers - {core}) | set(network.LEAVES) | set(network.RECEIVERS)
        devices = {r['device'] for r in rows}
        if (len(rows) != len(peers) or {r['remote_as'] for r in rows} != peers or
                len(devices) != len(rows) or any(not re.fullmatch('eth[1-9][0-9]*', d) for d in devices)):
            raise ValueError(f'Expected one distinct data interface per peer at {ia}')
        links.extend((ia, r['device']) for r in rows)
    return links


def set_fault_link(link, down, events, core, provider):
    ia, device = link
    requested = time.monotonic_ns()
    network.docker(ia, 'ip', 'link', 'set', device, 'down' if down else 'up')
    state = json.loads(network.docker(ia, 'ip', '-j', 'link', 'show', device).stdout)[0]
    if state['ifname'] != device or ('UP' in state['flags']) == down:
        raise RuntimeError(f'Unexpected link state after fault operation on {ia} {device}')
    event = dict(as_id=ia, interface=device, down=down, requested_ns=requested,
                 confirmed_ns=time.monotonic_ns())
    event.update(dict(core=core) if core else dict(provider=provider))
    events.append(event)


def restore_fault_links(touched, events, core, provider):
    failures = []
    for link in reversed(touched):
        try:
            set_fault_link(link, False, events, core, provider)
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError, KeyError, IndexError) as exc:
            failures.append(f'{link}: {exc}')
    if failures:
        raise RuntimeError('Could not restore every testbed link: ' + '; '.join(failures))


def resources(stop, output):
    with output.open('w') as f:
        while not stop.is_set():
            row = dict(monotonic_ns=time.monotonic_ns(),
                       cpu=Path('/proc/stat').read_text().splitlines()[0],
                       load=Path('/proc/loadavg').read_text().strip(),
                       memory=Path('/proc/meminfo').read_text().splitlines()[:3])
            f.write(json.dumps(row) + '\n'); f.flush()
            stop.wait(1)


def summarize(out, instance, selected, manifest, logs, receipts):
    N, T = instance['N'], instance['T']
    schedule = selected['schedule']
    expected = {(t, instance['validators'][v]['id']) for t, group in enumerate(schedule) for v in group}
    starts = [r for r in logs if r['kind'] == 'start']
    keys = [(r['round'], r['validator']) for r in starts]
    if len(keys) != N * T or set(keys) != expected:
        raise ValueError('Missing or duplicate proposal starts')
    if any(r.get('run') != manifest['run'] or
           r['planned_ns'] != manifest['start_ns'] + r['round'] * manifest['interval_ns']
           for r in starts):
        raise ValueError('Proposal start does not match the run and round origin')
    skew = max((r['actual_ns'] - r['planned_ns']) / 1e6 for r in starts)
    if any(r['actual_ns'] < r['planned_ns'] for r in starts):
        raise ValueError('Proposal started before its scheduled time')
    successful, earliest = set(), {}
    for r in receipts:
        t = int(r['round'])
        if (t, r['validator']) not in expected or r['receiver'] not in network.RECEIVERS or r['run'] != manifest['run']:
            raise ValueError('Unexpected receipt identity')
        origin = manifest['start_ns'] + t * manifest['interval_ns']
        if r['verified']:
            if r['bytes'] != manifest['payload_bytes']:
                raise ValueError('Verified receipt has the wrong payload size')
            if r['complete_ns'] < origin:
                raise ValueError('Receiver clock inconsistent with scheduled start')
            key = (t, r['validator'], r['receiver'])
            earliest[key] = min(earliest.get(key, float('inf')), (r['complete_ns'] - origin) / 1e6)
            if r['complete_ns'] <= origin + manifest['deadline_ns']:
                successful.add((t, r['validator'], r['receiver']))
    fault = manifest['failed_core']
    provider = manifest.get('failed_provider')
    if fault or provider:
        s = instance['scenarios'].index(['AS:' + fault if fault else 'host:' + provider])
        predicted_mask = instance['F'][s]
        budget = int(instance['budget'][s])
    else:
        predicted_mask = [False] * instance['n']
        budget = max(1, N // 3)
    failures, regions, outcomes, delivered_sets = [], [], [], []
    for t, group in enumerate(schedule):
        delivered = [v for v in group if all((t, instance['validators'][v]['id'], r) in successful for r in network.RECEIVERS)]
        failures.append(N - len(delivered))
        regions.append(len({instance['validators'][v]['region'] for v in delivered}))
        delivered_sets.append({instance['validators'][v]['region'] for v in delivered})
        outcomes.extend(dict(round=t, validator=instance['validators'][v]['id'],
                             predicted_affected=bool(predicted_mask[v]), delivered=v in delivered)
                        for v in group)
    predicted = [sum(bool(predicted_mask[v]) for v in group) for group in schedule]
    confusion = {f'predicted_{p}_delivered_{d}':sum(r['predicted_affected']==p and r['delivered']==d for r in outcomes)
                 for p in [False,True] for d in [False,True]}
    measured.write(out/'proposal-outcomes.json',outcomes)
    planned_sets = [{instance['validators'][v]['region'] for v in group} for group in schedule]
    region_labels = {v['region'] for v in instance['validators']}
    windows = {str(h): {name:sum(not any(r in sets[(t+j)%T] for j in range(h))
                            for r in region_labels for t in range(T))
                       for name, sets in [('scheduled_misses',planned_sets),('delivered_misses',delivered_sets)]}
               for h in (1,2,4) if h<=T}
    summary = dict(scope='single-host SCION deployment pilot, not consensus performance',
        run=manifest['run'], method=selected['method'], variant=instance['measurement']['variant'],
        routing=manifest['routing'], failed_core=fault, rounds=T, proposers=N,
        proposal_attempts=N*T, proposals_delivered=sum(N-f for f in failures),
        max_start_skew_ms=skew, accepted=skew<=5,
        rejection_reason=None if skew<=5 else 'start skew exceeds preregistered 5 ms tolerance',
        predicted_unavailable=predicted, observed_unsuccessful=failures,
        predicted_excess=sum(max(0,x-budget) for x in predicted),
        observed_excess=sum(max(0,x-budget) for x in failures),
        mean_scheduled_regions=selected['mean_regions'], mean_delivered_regions=sum(regions)/T,
        prediction_confusion=confusion,
        cyclic_regional_windows=windows,
        delivered_regions=regions, verified_receipt_latencies_ms=list(earliest.values()),
        application_bytes_offered=sum(r['application_bytes_offered'] for r in logs if r['kind']=='attempt'),
        sender_acknowledgements=sum(bool(r['acknowledged']) for r in logs if r['kind']=='attempt'))
    if provider:
        summary.update(failed_provider=provider, failed_provider_ases=manifest['failed_provider_ases'])
    measured.write(out / 'summary.json', summary)
    return summary


def execute(case, method, out, failed_core='', routing='surviving', interval_ms=1000, deadline_ms=250,
            warmup_payloads=1, payload_bytes=262144, *, failed_provider=None):
    if not 0 < deadline_ms < interval_ms:
        raise ValueError('Deadline must be positive and shorter than the round interval')
    if not isinstance(payload_bytes,int) or not 0<payload_bytes<=4*1024*1024:
        raise ValueError('Payload must be an integer from 1 byte to 4 MiB')
    instance = measured.load_instance(case / 'instance.json')
    selected = next(r for r in json.loads((case / 'schedules.json').read_text()) if r['method'] == method)
    if 'schedule' not in selected:
        raise ValueError('No valid schedule returned for this method')
    measured.metrics(selected['schedule'], instance)
    minimum_expiry = min(datetime.fromisoformat(r['expiry'].replace('Z','+00:00')).timestamp()
                         for v in instance['routes'] for rs in v['services'] for r in rs)
    if minimum_expiry < time.time() + instance['T'] * interval_ms / 1000 + 120:
        raise ValueError('Frozen paths expire before this replay ends')
    out.mkdir(parents=True, exist_ok=False)
    run_id = lab.new_run_id(out.name)
    hosts = network.addresses()
    manifest = dict(run=run_id, task_id=out.name, method=method, failed_core=failed_core, routing=routing,
        interval_ns=int(interval_ms*1e6), deadline_ns=int(deadline_ms*1e6), payload_bytes=payload_bytes,
        primary_attempt_ns=150_000_000, warmup_payloads=warmup_payloads, case=str(case.resolve()),
        agent=json.loads((lab.GEN / 'agent-install.json').read_text()),
        runtime_link_profile=json.loads((lab.GEN / 'link-profile.json').read_text()),
        **lab.source_provenance(),
        code_sha256=lab.code_hashes(['lab.py', 'network.py', 'measured.py', 'replay.py',
                                     'cmd/mcp-traffic/main.go', 'cmd/mcp-traffic/BUILD.bazel', 'versions.json']),
        instance_sha256=lab.file_sha256(case / 'instance.json'))
    links = fault_links(instance, manifest['runtime_link_profile'], failed_core, failed_provider)
    if failed_provider:
        manifest.update(failed_provider=failed_provider, failed_provider_ases=sorted({ia for ia, _ in links}))
    configs, processes, faults, touched = [], [], [], []
    stop = threading.Event()
    monitor = threading.Thread(target=resources,args=(stop,out/'resources.jsonl'))
    monitor.start()
    try:
        for ia in network.LEAVES:
            local = network.host_config(ia).parent / 'runs' / run_id
            local.mkdir(parents=True, exist_ok=False)
            remote = '/etc/scion/runs/' + run_id
            jobs = []
            for t, group in enumerate(selected['schedule']):
                for v in group:
                    validator = instance['validators'][v]
                    if validator['as_id'] != ia:
                        continue
                    targets = [[dict(remote=f'{r["receiver"]},{hosts[r["receiver"]]}:4444',
                                      sequence=r['sequence'], fingerprint=r['fingerprint'], core=r['core'])
                                for r in rs] for rs in instance['routes'][v]['services']]
                    jobs.append(dict(round=t, validator=validator['id'], targets=targets))
            cfg = dict(manifest, daemon=hosts[ia]+':30255', jobs=jobs, rounds=instance['T'],
                       start_file=remote+'/start.json',output=remote+'/sender.jsonl',ready=remote+'/ready.json')
            measured.write(local/'agent.json', cfg)
            stream = (out / f'agent-{ia.split(":")[-1]}.log').open('w')
            processes.append((subprocess.Popen(['docker','exec',network.container(ia),'/app/mcp-agent',
                        '--mode','replay','--config',remote+'/agent.json'],stdout=stream,stderr=stream),stream))
            configs.append(local)
        until = time.monotonic()+90
        while not all((p/'ready.json').exists() for p in configs):
            if any(p.poll() is not None and p.returncode != 0 for p,_ in processes):
                raise RuntimeError('Agent warmup failed; inspect per-agent logs')
            if time.monotonic()>until:
                raise TimeoutError('Agents did not become ready')
            time.sleep(.1)
        before = {
            ia:json.loads(network.docker(ia,'ip','-s','-j','link').stdout)
            for ia in network.CORES+network.LEAVES+network.RECEIVERS}
        measured.write(out/'interfaces-before.json', before)
        active = {(ia, d['ifname']) for ia, devices in before.items() for d in devices if 'UP' in d['flags']}
        if not set(links).issubset(active):
            raise RuntimeError('An interface selected for the outage was already down or missing')
        for link in links:
            # A command may set the interface down and then time out during verification.
            touched.append(link)
            set_fault_link(link, True, faults, failed_core, failed_provider)
        start = time.monotonic_ns()+3_000_000_000
        manifest['start_ns']=start
        manifest['started_at_utc']=datetime.now().astimezone().isoformat()
        measured.write(out/'manifest.json', manifest)
        measured.write(out/'schedule.json', selected)
        for directory in configs:
            measured.write(directory/'start.json', dict(start_ns=start))
        for process,stream in processes:
            result = process.wait(timeout=instance['T']*interval_ms/1000+30)
            stream.close()
            if result:
                raise RuntimeError('Replay agent failed; inspect per-agent logs')
        logs = [json.loads(line) for directory in configs
                for line in (directory/'sender.jsonl').read_text().splitlines()]
        receipts = [r for ia in network.RECEIVERS
                    for r in map(json.loads,(network.host_config(ia).parent/'proposal-receipts.jsonl').read_text().splitlines())
                    if r['run']==run_id]
        measured.write(out/'interfaces-after.json', {
            ia:json.loads(network.docker(ia,'ip','-s','-j','link').stdout)
            for ia in network.CORES+network.LEAVES+network.RECEIVERS})
        (out/'senders.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in logs))
        (out/'receipts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in receipts))
        return summarize(out,instance,selected,manifest,logs,receipts)
    finally:
        try:
            restore_fault_links(touched, faults, failed_core, failed_provider)
        finally:
            measured.write(out/'faults.json', faults)
            for ia,(process,stream) in zip(network.LEAVES,processes):
                if process.poll() is None:
                    # Terminating docker exec alone leaves the container process alive.
                    pattern = '^/app/mcp-agent --mode replay --config /etc/scion/runs/' + run_id + '/agent.json$'
                    network.docker(ia,'pkill','-f',pattern,check=False)
                    process.terminate()
                    process.wait(timeout=10)
                stream.close()
            stop.set();monitor.join()
            measured.write(out/'checksums.json',{p.name:lab.file_sha256(p)
                           for p in out.iterdir() if p.is_file() and p.name!='checksums.json'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build','install','run'])
    parser.add_argument('--profile', choices=['research', 'expanded'], default='research')
    parser.add_argument('--case',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--method',default='full')
    fault = parser.add_mutually_exclusive_group()
    fault.add_argument('--failed-core',default='')
    fault.add_argument('--failed-provider', help='Hosting provider label from the instance, such as H0')
    parser.add_argument('--routing',choices=['surviving','retry'],default='surviving')
    parser.add_argument('--interval-ms',type=float,default=1000)
    parser.add_argument('--deadline-ms',type=float,default=250)
    parser.add_argument('--warmup-payloads',type=int,default=1)
    parser.add_argument('--payload-kib',type=int,default=256)
    args=parser.parse_args();lab.select_profile(args.profile)
    if args.action=='build':build_agent()
    elif args.action=='install':install()
    else:
        if args.case is None or args.out is None:parser.error('run needs --case and --out')
        summary=execute(args.case,args.method,args.out,args.failed_core,args.routing,args.interval_ms,args.deadline_ms,
                        args.warmup_payloads,args.payload_kib*1024,failed_provider=args.failed_provider)
        print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,list)},indent=2))
        if not summary['accepted']:raise SystemExit('Run failed the start-skew gate')


if __name__=='__main__':main()
