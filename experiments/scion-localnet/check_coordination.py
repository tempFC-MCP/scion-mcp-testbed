"""Exhaustive small-case solver check and independent peer-delivery reconstruction."""
import argparse
import hashlib
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np

import lab
import coordination


def small_cases():
    checked = 0
    for limit in (1., 3.):
        for quota in ([2, 1, 1, 0], [1, 1, 1, 1], [0, 2, 1, 1]):
            x = dict(n=4, N=2, T=2, nr=2, g=2, regions=np.array([0, 0, 1, 1]),
                     q=np.array(quota), cost=np.array([[0, 1, 3, 3], [1, 0, 3, 3],
                                                      [3, 3, 0, 1], [3, 3, 1, 0]]),
                     tau=limit, F=np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=bool),
                     budget=np.array([1, 1]), h=None)
            groups = [g for g in combinations(range(4), 2) if x['cost'][g[0], g[1]] <= limit]
            schedules = [a for a in product(groups, repeat=2)
                         if np.array_equal(np.bincount(np.array(a).ravel(), minlength=4), quota)]
            if not schedules:
                continue
            for policy in ('geo', 'full'):
                def score(schedule):
                    return sum(10 * (2 - len(set(x['regions'][list(g)])))
                               + (sum(max(0, sum(int(f[v]) for v in g) - 1) for f in x['F'])
                                  if policy == 'full' else 0) for g in schedule)
                result = coordination.solve_groups(x, policy)
                assert result['solver']['optimal']
                assert result['solver']['objective'] == min(map(score, schedules))
                checked += 1
    print(f'Group solver matches exhaustive enumeration in {checked} cases.')


def peer_records(out):
    records = json.loads((out / 'peer-results.json').read_text())
    expected_runs = json.loads((out / 'peer-plan.json').read_text())['tasks']
    assert len(records) == len(expected_runs), 'Peer suite is incomplete'
    messages = 0
    for item in records:
        task, saved = item['task'], item['summary']
        dest = out / 'peers' / task['id']
        manifest = json.loads((dest / 'manifest.json').read_text())
        assert manifest.get('task_id', manifest['run']) == task['id']
        case = out / 'cases' / task['case']
        for name, key in [('instance.json', 'case_sha256'), ('schedules.json', 'schedule_sha256')]:
            assert lab.file_sha256(case / name) == manifest[key]
        x = json.loads((case / 'instance.json').read_text())
        a = next(r['schedule'] for r in json.loads((case / 'schedules.json').read_text()) if r['method'] == task['method'])
        expected = {(t, f'{x["validators"][u]["id"]}-to-{x["validators"][v]["id"]}'):
                    x['validators'][v]['as_id']
                    for t, group in enumerate(a) for u in group for v in group if u != v}
        assert len(expected) == x['T'] * x['N'] * (x['N'] - 1)
        jobs = json.loads((dest / 'jobs.json').read_text())
        assert len(jobs) == len(expected)
        assert {(j['round'], j['validator']): j['destination_as'] for j in jobs} == expected
        starts = [r for r in map(json.loads, (dest / 'senders.jsonl').read_text().splitlines()) if r['kind'] == 'start']
        assert len(starts) == len(expected)
        assert {(r['round'], r['validator']) for r in starts} == set(expected)
        receipts = list(map(json.loads, (dest / 'receipts.jsonl').read_text().splitlines()))
        delivered = {}
        digest = hashlib.sha256(bytes([0xa5]) * manifest['payload_bytes']).hexdigest()
        for r in receipts:
            key = int(r['round']), r['validator']
            assert expected[key] == r['receiver'] and r['run'] == manifest['run']
            if r['verified']:
                assert r['hash'] == digest and r['bytes'] == manifest['payload_bytes']
                delivered[key] = min(delivered.get(key, r['complete_ns']), r['complete_ns'])
        complete = []
        for t in range(x['T']):
            origin = manifest['start_ns'] + t * manifest['interval_ns']
            deadline = origin + manifest['deadline_ns']
            keys = [k for k in expected if k[0] == t]
            complete.append(max((delivered[k] - origin) / 1e6 for k in keys)
                            if all(k in delivered and delivered[k] <= deadline for k in keys) else None)
        assert complete == saved['group_completion_ms']
        for deadline, count in saved['complete_rounds_by_ms'].items():
            assert count == sum(v is not None and v <= int(deadline) for v in complete)
        assert saved['received_messages'] == len(delivered)
        messages += len(expected)
    print(f'Independently reconstructed {messages} directed messages in {len(records)} peer runs.')


def payload_records(out):
    plan = json.loads((out / 'plan.json').read_text())
    opportunities = 0
    for task in plan['tasks']:
        dest = out / 'runs' / task['id']
        m = json.loads((dest / 'manifest.json').read_text())
        assert m.get('task_id', m['run']) == task['id']
        saved = json.loads((dest / 'summary.json').read_text())
        lab.verify_recording(dest)
        source = out / 'cases' / task['case'] / 'instance.json'
        assert lab.file_sha256(source) == m['instance_sha256']
        x = json.loads(source.read_text())
        selected = json.loads((dest / 'schedule.json').read_text())
        a = selected['schedule']
        # Rebuild failure sets from the concrete selected paths, not stored F.
        masks = []
        for scenario in x['scenarios']:
            failed = set(scenario)
            masks.append([('host:' + v['provider'] in failed)
                or any(all(bool({'AS:' + v['as_id'], 'AS:' + path['core'],
                                 'AS:' + path['receiver']} & failed) for path in service)
                       for service in routes['services'])
                for v, routes in zip(x['validators'], x['routes'])])
        assert masks == [[bool(v) for v in mask] for mask in x['F']]
        expected = {(t, x['validators'][v]['id']) for t, group in enumerate(a) for v in group}
        senders = list(map(json.loads, (dest / 'senders.jsonl').read_text().splitlines()))
        starts = [r for r in senders if r['kind'] == 'start']
        assert len(starts) == len(expected) and {(r['round'], r['validator']) for r in starts} == expected
        for r in starts:
            assert r['planned_ns'] == m['start_ns'] + r['round'] * m['interval_ns']
            assert r['actual_ns'] >= r['planned_ns']
        skew = max((r['actual_ns'] - r['planned_ns']) / 1e6 for r in starts)
        assert skew == saved['max_start_skew_ms'] and saved['accepted'] and skew <= 5
        received = {key: set() for key in expected}
        digest = hashlib.sha256(bytes([0xa5]) * m['payload_bytes']).hexdigest()
        for r in map(json.loads, (dest / 'receipts.jsonl').read_text().splitlines()):
            key = int(r['round']), r['validator']
            assert key in expected and r['run'] == m['run']
            assert r['receiver'] in ('1-ff00:0:301', '1-ff00:0:302')
            if r['verified']:
                assert r['bytes'] == m['payload_bytes'] and r['hash'] == digest
                if r['complete_ns'] <= m['start_ns'] + key[0] * m['interval_ns'] + m['deadline_ns']:
                    received[key].add(r['receiver'])
        delivered = {key for key, receivers in received.items() if len(receivers) == 2}
        missing = [x['N'] - sum((t, x['validators'][v]['id']) in delivered for v in group)
                   for t, group in enumerate(a)]
        regions = [len({x['validators'][v]['region'] for v in group
                        if (t, x['validators'][v]['id']) in delivered}) for t, group in enumerate(a)]
        assert missing == saved['observed_unsuccessful'] and regions == saved['delivered_regions']
        assert len(delivered) == saved['proposals_delivered']
        mask = masks[x['scenarios'].index(['AS:' + m['failed_core']])] if m['failed_core'] else [False] * x['n']
        assert [sum(mask[v] for v in group) for group in a] == saved['predicted_unavailable']
        opportunities += len(expected)
    print(f'Independently reconstructed {opportunities} proposal deliveries in {len(plan["tasks"])} payload runs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--payloads', action='store_true', help='Also require the completed payload suite')
    args = parser.parse_args()
    small_cases()
    if args.out:
        coordination.check(args.out)
        peer_records(args.out)
        if args.payloads:
            payload_records(args.out)
