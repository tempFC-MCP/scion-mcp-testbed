#!/usr/bin/env python3
"""Pinned upstream SCION smoke lab. Does not run paper experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tarfile
import time
import urllib.request
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
STATE = ROOT / '.state'
TOOLS = STATE / 'tools'
SOURCE = STATE / 'scion'
GEN = STATE / 'gen'
LAB = 'mcp-scion-smoke'
LOCK = json.loads((ROOT / 'versions.json').read_text())
TOPO = GEN / f'{LAB}.clab.yml'


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def code_hashes(names):
    return {name: file_sha256(ROOT / name) for name in names}


def verify_files(directory, expected):
    for name, digest in expected.items():
        path = directory / name
        if file_sha256(path) != digest:
            raise ValueError(f'Checksum mismatch: {path}')


def verify_recording(directory):
    verify_files(directory, json.loads((directory / 'checksums.json').read_text()))


def select_profile(profile):
    global GEN, LAB, TOPO
    if profile == 'research':
        GEN = STATE / 'research-gen'
        LAB = 'mcp-scion-research'
    elif profile == 'expanded':
        GEN = STATE / 'expanded-gen'
        LAB = 'mcp-scion-expanded'
    elif profile == 'smoke':
        GEN = STATE / 'gen'
        LAB = 'mcp-scion-smoke'
    else:
        raise ValueError(profile)
    TOPO = GEN / f'{LAB}.clab.yml'
    if profile != 'smoke':
        import network
        network.configure(profile == 'expanded')


def run(args, cwd=ROOT, env=None):
    print('+ ' + shlex.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)


def output(args, cwd=ROOT):
    return subprocess.check_output(list(map(str, args)), cwd=cwd, text=True).strip()


def env():
    result = os.environ.copy()
    result['BAZELISK_HOME'] = str(STATE / 'bazelisk')
    return result


def bazel(command, *args):
    run([TOOLS / 'bazel', '--output_user_root=' + str(STATE / 'bazel'), command,
         '--remote_cache=', '--experimental_remote_downloader=', '--jobs=8',
         *args], cwd=SOURCE, env=env())


def paper_revision():
    """An unpacked source archive has no Git revision. Run plans still hash the code."""
    checkout = ROOT.parents[1]
    if not (checkout / '.git').exists():
        return None
    return output(['git', 'rev-parse', 'HEAD'], checkout)


def source_provenance():
    revision = paper_revision()
    digest = None
    if revision is not None:
        changes = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT.parents[1])
        digest = hashlib.sha256(changes).hexdigest()
    return dict(source_revision=revision, dirty_diff_sha256=digest)


def new_run_id(task_id):
    """Give each replay separate container files and receipt IDs across batches."""
    if not task_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Run name must be alphanumeric with hyphens/underscores')
    return f'{task_id}-{uuid.uuid4().hex}'


def verify_source():
    revision = output(['git', 'rev-parse', 'HEAD'], SOURCE)
    if revision != LOCK['scion_commit']:
        raise RuntimeError('SCION checkout differs from versions.json')
    if output(['git', 'status', '--porcelain', '--untracked-files=no'], SOURCE):
        raise RuntimeError('SCION tracked source has local modifications')


def download(name):
    spec = LOCK[name]
    target = TOOLS / ('bazel' if name == 'bazelisk' else name)
    archive = TOOLS / (name + '.download')
    if not archive.exists():
        temporary = archive.with_suffix('.part')
        with urllib.request.urlopen(spec['url'], timeout=60) as response:
            with temporary.open('wb') as stream:
                shutil.copyfileobj(response, stream)
        temporary.replace(archive)
    digest = file_sha256(archive)
    if digest != spec['sha256']:
        raise RuntimeError(f'{name}: download checksum mismatch')
    if name == 'containerlab':
        with tarfile.open(archive) as tar:
            matches = [m for m in tar.getmembers()
                       if Path(m.name).name == name and m.isfile()]
            if len(matches) != 1:
                raise RuntimeError('Expected one containerlab executable')
            with tar.extractfile(matches[0]) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
    else:
        shutil.copyfile(archive, target)
    target.chmod(0o755)


def bootstrap():
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('This lockfile currently supports Linux x86-64 only')
    TOOLS.mkdir(parents=True, exist_ok=True)
    for name in ['bazelisk', 'containerlab']:
        download(name)
    if not SOURCE.exists():
        SOURCE.mkdir()
        run(['git', 'init', SOURCE])
        run(['git', 'remote', 'add', 'origin', LOCK['scion_repository']], SOURCE)
        run(['git', 'fetch', '--depth', '1', 'origin', LOCK['scion_commit']], SOURCE)
        run(['git', 'checkout', '--detach', 'FETCH_HEAD'], SOURCE)
    verify_source()
    if (SOURCE / '.bazelversion').read_text().strip() != LOCK['bazel_version']:
        raise RuntimeError('Bazel version does not match the source pin')
    print('Pinned tools and source prepared. No system installation changed.')


def build():
    verify_source()
    bazel('run', '//testing/clab:clab_node.load')
    bazel('build', '//testing/clab/cmd/testgen',
          '//testing/clab/cmd/await_connectivity',
          '//testing/clab/cmd/e2e_scion', '//testing/clab/cmd/e2e_http')
    image = output(['docker', 'image', 'inspect', 'scion/clab-node:latest'])
    (STATE / 'scion-image.json').write_text(image + '\n')


def binary(name):
    return SOURCE / 'bazel-bin' / 'testing/clab/cmd' / name / (name + '_') / name


def generate():
    verify_source()
    if GEN.exists():
        raise RuntimeError('Generated directory already exists; preserve it or move it '
                           'after destroying the lab before regenerating')
    research = LAB in ('mcp-scion-research', 'mcp-scion-expanded')
    expanded = LAB == 'mcp-scion-expanded'
    if research:
        from network import topology
        spec = STATE / ('expanded.topo' if expanded else 'research.topo')
        spec.write_text(json.dumps(topology(), indent=2) + '\n')
    else:
        spec = SOURCE / 'topology/tiny.topo'
    run([binary('testgen'), '--topo', spec,
         '--out', GEN, '--name', LAB,
         '--network', '10.66.0.0/16' if expanded else ('10.65.0.0/16' if research else '10.64.0.0/16'),
         '--network-v6', 'fd00:209::/48' if expanded else ('fd00:208::/48' if research else 'fd00:207::/48')])


def clab(command, containerized):
    if not TOPO.exists():
        raise RuntimeError('Generate the smoke topology first')
    args = [command, '-t', TOPO]
    if command == 'deploy':
        expected = json.loads((STATE / 'scion-image.json').read_text())[0]['Id']
        actual = json.loads(output(['docker', 'image', 'inspect',
                                    'scion/clab-node:latest']))[0]['Id']
        if actual != expected:
            raise RuntimeError('SCION image tag no longer matches the built image')
    if containerized:
        # Official containerlab mode. Requires privileged Docker authorization.
        # Mount only lab state and the runtime resources needed for namespaces.
        prefix = ['docker', 'run', '--rm', '--privileged', '--network', 'host',
                  '--entrypoint', 'containerlab',
                  '--pid', 'host', '-v', '/var/run/docker.sock:/var/run/docker.sock',
                  '-v', '/var/run/netns:/var/run/netns',
                  '-v', '/var/lib/docker/containers:/var/lib/docker/containers',
                  '-v', f'{STATE}:{STATE}', '-w', GEN,
                  LOCK['containerlab_image']]
    else:
        prefix = [TOOLS / 'containerlab']
    run(prefix + args, cwd=GEN)


def smoke():
    (STATE / 'smoke-success.json').unlink(missing_ok=True)
    run([binary('await_connectivity'), '-gen', GEN, '-timeout', '60s'])
    run([binary('e2e_scion'), '-gen', GEN, '-lab', LAB, '-parallel', '4'])
    run([binary('e2e_http'), 'run', '--gen', GEN, '--lab', LAB, '--parallel', '4'])
    (STATE / 'smoke-success.json').write_text(json.dumps({
        'scion_commit': LOCK['scion_commit'], 'lab': LAB,
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'upstream connectivity and HTTP/3 smoke test only',
        'topology_sha256': file_sha256(TOPO)
    }, indent=2) + '\n')


def cut_check():
    """Tiny-topology negative control, with unconditional link restoration."""
    if LAB != 'mcp-scion-smoke':
        raise RuntimeError('cut-check is specific to the smoke topology')
    (STATE / 'cut-check-success.json').unlink(missing_ok=True)
    addresses = json.loads((GEN / 'sciond_addresses.json').read_text())
    src = '1-ff00:0:111'
    dst = '1-ff00:0:110'
    container = f'clab-{LAB}-1-ff00_0_111-host-1'
    daemon = addresses[src] + ':30255'
    ping = ['docker', 'exec', container, '/app/scion', 'ping', '-c', '1',
            '--timeout', '2s', '--sciond', daemon, f'{dst},{addresses[dst]}']
    run(ping)
    try:
        run(['docker', 'exec', container, 'ip', 'link', 'set', 'eth1', 'down'])
        # A daemon query over the intact management interface must still work.
        run(['docker', 'exec', container, '/app/scion', 'showpaths', dst,
             '--sciond', daemon, '--no-probe'])
        failed = subprocess.run(ping, text=True, capture_output=True, timeout=15)
        print(failed.stdout, end='')
        print(failed.stderr, end='')
        if failed.returncode == 0:
            raise RuntimeError('SCION ping unexpectedly succeeded across the cut')
        if '100% packet loss' not in failed.stdout + failed.stderr:
            raise RuntimeError('Ping failed for a reason other than measured packet loss')
    finally:
        run(['docker', 'exec', container, 'ip', 'link', 'set', 'eth1', 'up'])
    restored_at = time.monotonic()
    restored = False
    attempts = 0
    # Interface-up is not equivalent to forwarding recovery (e.g. BFD state).
    while time.monotonic() - restored_at < 20:
        attempts += 1
        probe = subprocess.run(ping, text=True, capture_output=True, timeout=10)
        print(probe.stdout, end='')
        print(probe.stderr, end='')
        if probe.returncode == 0:
            restored = True
            break
    if not restored:
        raise RuntimeError('Link restored but SCION ping did not recover within 20 seconds')
    (STATE / 'cut-check-success.json').write_text(json.dumps({
        'scion_commit': LOCK['scion_commit'],
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'tiny topology ping blocked by data-link cut; daemon query remains available',
        'restored_ping': True,
        'recovery_probe_attempts': attempts,
        'recovery_observation_seconds': time.monotonic() - restored_at,
    }, indent=2) + '\n')


def doctor():
    failures = []
    for name, args in [('Docker', ['docker', 'version']),
                       ('containerlab', [TOOLS / 'containerlab', 'version']),
                       ('Python', ['python3', '--version'])]:
        try:
            run(args)
        except (OSError, subprocess.CalledProcessError) as exc:
            failures.append(name)
            print(f'{name} unavailable: {exc}')
    print('Host:', platform.platform(), 'CPUs:', os.cpu_count())
    print('Free disk GiB:', round(shutil.disk_usage(ROOT).free / 2**30, 1))
    print('Effective UID:', os.geteuid())
    print('Network namespace privileges must be checked during deployment.')
    if failures:
        raise RuntimeError('Unavailable prerequisites: ' + ', '.join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['doctor', 'bootstrap', 'build', 'generate',
                                          'up', 'smoke', 'cut-check', 'down'])
    parser.add_argument('--containerized-clab', action='store_true',
                        help='Use the official privileged containerlab container')
    parser.add_argument('--profile', choices=['smoke', 'research', 'expanded'], default='smoke')
    args = parser.parse_args()
    select_profile(args.profile)
    if args.action in ('up', 'down'):
        clab('deploy' if args.action == 'up' else 'destroy', args.containerized_clab)
    else:
        globals()[args.action.replace('-', '_')]()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
