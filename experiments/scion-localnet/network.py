"""Research topology, controlled link profiles, and live SCION observations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import subprocess
import time

import lab

CORES = [f'1-ff00:0:{i}' for i in (110, 111, 112)]
LEAVES = [f'1-ff00:0:{i}' for i in range(201, 207)]
RECEIVERS = [f'1-ff00:0:{i}' for i in (301, 302)]
REGIONS = dict(zip(CORES + LEAVES + RECEIVERS,
                   [0, 1, 2, 0, 0, 1, 1, 2, 2, 0, 2]))
PROVIDERS = [0, 1, 1, 2, 2, 0]


def configure(expanded=False):
    """Mutate lists in place so existing adapters retain the selected profile."""
    CORES[:] = [f'1-ff00:0:{i}' for i in range(110, 116 if expanded else 113)]
    LEAVES[:] = [f'1-ff00:0:{i}' for i in range(201, 210 if expanded else 207)]
    PROVIDERS[:] = [0, 1, 2] * 3 if expanded else [0, 1, 1, 2, 2, 0]
    REGIONS.clear()
    REGIONS.update(zip(CORES + LEAVES + RECEIVERS,
        ([0, 0, 1, 1, 2, 2] + [0]*3 + [1]*3 + [2]*3 + [0, 2])
        if expanded else [0, 1, 2, 0, 0, 1, 1, 2, 2, 0, 2]))


def topology():
    ases = {c: dict(core=True, voting=True, authoritative=True, issuing=True)
            for c in CORES}
    ases.update({a: dict(cert_issuer=CORES[0]) for a in LEAVES + RECEIVERS})
    counters = dict.fromkeys(ases, 0)
    links = []
    pairs = [(a, b, 'CORE') for a, b in itertools.combinations(CORES, 2)]
    pairs += [(c, a, 'CHILD') for a in LEAVES + RECEIVERS for c in CORES]
    for a, b, kind in pairs:
        counters[a] += 1
        counters[b] += 1
        links.append(dict(a=f'{a}-A#{counters[a]}', b=f'{b}-A#{counters[b]}',
                          linkAtoB=kind))
    return {'ASes': ases, 'links': links}


def host_config(ia):
    suffix = ia.split('-')[1].replace(':', '_')
    candidates = list((lab.GEN / f'AS{suffix}').glob('host-*/topology.json'))
    if len(candidates) != 1:
        raise ValueError(f'Expected one host for {ia}, found {len(candidates)}')
    return candidates[0]


def container(ia):
    return f'clab-{lab.LAB}-{ia.replace(":", "_")}-{host_config(ia).parent.name}'


def docker(ia, *args, timeout=30, check=True):
    result = subprocess.run(['docker', 'exec', container(ia), *map(str, args)],
                            capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f'{ia}: {result.stderr or result.stdout}')
    return result


def addresses():
    return json.loads((lab.GEN / 'sciond_addresses.json').read_text())


def inventory():
    hosts = addresses()
    if set(hosts) != set(CORES + LEAVES + RECEIVERS):
        raise ValueError('Generated topology has unexpected ASes')
    return [dict(id=f'v{i:02d}', as_id=LEAVES[i // 4], region=REGIONS[LEAVES[i // 4]],
                 provider=f'H{PROVIDERS[i // 4]}', local_ip=hosts[LEAVES[i // 4]])
            for i in range(4 * len(LEAVES))]


def interfaces():
    """Resolve interfaces from generated topology and running addresses, not guesses."""
    rows = []
    for ia in CORES + LEAVES + RECEIVERS:
        config = json.loads(host_config(ia).read_text())
        if len(config['border_routers']) != 1:
            raise ValueError('Expected one border router per AS')
        by_addr = {a['local']: i['ifname']
                   for i in json.loads(docker(ia, 'ip', '-j', 'address').stdout)
                   for a in i.get('addr_info', [])}
        router = next(iter(config['border_routers'].values()))
        for ifid, data in router['interfaces'].items():
            local = data['underlay']['local'].rsplit(':', 1)[0]
            dev = by_addr[local]
            if dev == 'eth0':
                raise ValueError('Refusing to impair management interface')
            rows.append(dict(as_id=ia, interface_id=int(ifid), device=dev,
                             remote_as=data['isd_as'], local_ip=local))
    expected = 2 * len(topology()['links'])
    if len(rows) != expected:
        raise ValueError(f'Expected {expected} directed interfaces; found {len(rows)}')
    return rows


def shape(profile, rate_mbit=None, jitter_ms=0):
    if profile not in ('wan', 'continental', 'unshaped'):
        raise ValueError('Unknown link profile')
    rate_mbit = rate_mbit if rate_mbit is not None else (100 if profile == 'wan' else 1000)
    if rate_mbit <= 0 or not 0 <= jitter_ms <= 1 or (profile == 'unshaped' and jitter_ms):
        raise ValueError('Invalid rate or jitter (supported jitter range: 0 to 1 ms)')
    rows = interfaces()
    for row in rows:
        ra, rb = sorted((REGIONS[row['as_id']], REGIONS[row['remote_as']]))
        matrix = ({(0, 1): 38, (1, 2): 82, (0, 2): 110} if profile == 'continental'
                  else {(0, 1): 15, (1, 2): 35, (0, 2): 55})
        delay = 2 if ra == rb else matrix[(ra, rb)]
        row['delay_ms'] = delay if profile != 'unshaped' else 0
        row['rate_mbit'] = rate_mbit
        row['jitter_ms'] = jitter_ms
        args = ['tc', 'qdisc', 'replace', 'dev', row['device'], 'root', 'netem',
                'delay', f'{row["delay_ms"]}ms']
        if jitter_ms:
            # netem's built-in jitter is uniform. An explicit distribution
            # name would request a .dist file absent from the minimal image.
            args += [f'{jitter_ms}ms']
        args += ['rate', f'{rate_mbit}mbit', 'limit', '10000']
        row['command'] = args
        docker(row['as_id'], *args)
        row['actual_qdisc'] = json.loads(docker(row['as_id'], 'tc', '-j',
                                               'qdisc', 'show', 'dev', row['device']).stdout)
    record = dict(profile=profile, rate_mbit=rate_mbit, jitter_ms=jitter_ms,
                  jitter_distribution='uniform, independent per egress packet' if jitter_ms else 'none',
                  jitter_seed='not exposed by the installed tc; kernel randomness',
                  applied_at=datetime.now(timezone.utc).isoformat(),
                  interfaces=rows, delay_semantics='one directed egress, includes control traffic')
    (lab.GEN / 'link-profile.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Applied {profile}, {rate_mbit} Mbit/s, +/-{jitter_ms} ms jitter to {len(rows)} directed data interfaces.', flush=True)


def sequence_ases(path):
    result = []
    for hop in path.get('hops') or []:
        if not result or hop['isd_as'] != result[-1]:
            result.append(hop['isd_as'])
    return result


def discover(src, dst, hosts):
    started = time.time()
    args = ['/app/scion', 'showpaths', dst, '--sciond', hosts[src] + ':30255',
            '--format', 'json', '--maxpaths', '0', '--timeout', '15s',
            '--sequence', f'{src} 0 {dst}', '--refresh']
    attempts = []
    # Fresh daemons can answer before asynchronously requested segments arrive.
    # Record initialization attempts and wait for the declared topology, before
    # collecting latency samples. Refresh incomplete cached segment responses;
    # filter to the declared single-core routes before probing. Do not retry
    # or discard measured ping losses.
    until = time.monotonic() + 40
    while True:
        result = docker(src, *args)
        data = json.loads(result.stdout)
        attempts.append(dict(observed_at=time.time(), data=data))
        present = {sequence_ases(p)[1] for p in data.get('paths', [])
                   if len(sequence_ases(p)) == 3 and p.get('status') == 'alive'}
        if set(CORES) <= present or time.monotonic() >= until:
            break
        time.sleep(1)
    if data['local_isd_as'] != src or data['destination'] != dst:
        raise ValueError('Path query endpoint mismatch')
    return dict(source=src, destination=dst, observed_at=started, data=data,
                initialization_attempts=attempts)


def ping(src, dst, hosts, sequence, count):
    args = ['/app/scion', 'ping', f'{dst},{hosts[dst]}', '--sciond', hosts[src] + ':30255',
            '--format', 'json', '-c', str(count), '--interval', '10ms', '--timeout', '1s']
    if sequence:
        args += ['--sequence', sequence]
    result = docker(src, *args, timeout=max(30, count * .02 + 10))
    data = json.loads(result.stdout)
    return dict(source=src, destination=dst, observed_at=time.time(), count=count,
                requested_sequence=sequence, data=data)


def collect(out, count):
    out.mkdir(parents=True, exist_ok=False)
    hosts = addresses()
    manifest = dict(schema=1, scion_commit=lab.LOCK['scion_commit'],
                    started_at=time.time(), validators=inventory(), receivers=RECEIVERS,
                    cores=CORES, probe_count=count, probe_interval_ms=10,
                    link_profile=json.loads((lab.GEN / 'link-profile.json').read_text()),
                    source_revision=lab.paper_revision(),
                    topology_sha256=lab.file_sha256(lab.TOPO))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    requests = [(a, b) for a in LEAVES for b in LEAVES + RECEIVERS if a != b]
    selected = []
    with ThreadPoolExecutor(max_workers=4) as pool, (out / 'paths.jsonl').open('w') as f:
        for rec in pool.map(lambda pair: discover(*pair, hosts), requests):
            f.write(json.dumps(rec) + '\n')
            f.flush()
            for core in CORES:
                candidates = [p for p in rec['data'].get('paths', [])
                              if sequence_ases(p) == [rec['source'], core, rec['destination']]
                              and p.get('status') == 'alive']
                if not candidates:
                    raise ValueError(f'No alive single-core path: {rec["source"]} {core} {rec["destination"]}')
                p = min(candidates, key=lambda p: p['fingerprint'])
                selected.append((rec['source'], rec['destination'], p))
    jobs = [(a, b, p['sequence']) for a, b, p in selected]
    jobs += [(a, a, None) for a in LEAVES]
    with ThreadPoolExecutor(max_workers=4) as pool, (out / 'probes.jsonl').open('w') as f:
        for index, rec in enumerate(pool.map(lambda job: ping(job[0], job[1], hosts, job[2], count), jobs)):
            f.write(json.dumps(rec) + '\n')
            f.flush()
            if index % 12 == 0:
                print(f'Collected {index + 1}/{len(jobs)} probe batches', flush=True)
    (out / 'selected-paths.json').write_text(json.dumps([
        dict(source=a, destination=b, path=p) for a, b, p in selected], indent=2) + '\n')
    manifest['completed_at'] = time.time()
    manifest['files_sha256'] = {p.name: lab.file_sha256(p)
                                for p in out.iterdir() if p.name != 'manifest.json'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {len(selected)} selected paths and {len(jobs)} probe batches to {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['shape', 'collect'])
    parser.add_argument('--profile', choices=['wan', 'continental', 'unshaped'], default='wan')
    parser.add_argument('--rate-mbit', type=float)
    parser.add_argument('--jitter-ms', type=float, default=0)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--lab-profile', choices=['research', 'expanded'], default='research')
    args = parser.parse_args()
    lab.select_profile(args.lab_profile)
    configure(args.lab_profile == 'expanded')
    if args.action == 'shape':
        shape(args.profile, args.rate_mbit, args.jitter_ms)
    else:
        if args.out is None or args.count < 1:
            parser.error('collect requires --out and a positive --count')
        collect(args.out, args.count)


if __name__ == '__main__':
    main()
