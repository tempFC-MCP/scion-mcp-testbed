"""Verify the paired outage replays and count timely delivery to both observers."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import combinations, product
import json
from pathlib import Path

import lab
from recorded_delivery import CUTOFFS_MS, RECEIVERS

PROVIDERS = ('H0', 'H1', 'H2')
TRANSIT = '1-ff00:0:110'
POLICIES = ('transit', 'full')
PAYLOAD_BYTES = 256 * 1024
PAYLOAD_HASH = hashlib.sha256(bytes([0xa5]) * PAYLOAD_BYTES).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def counts(per_round):
    return sum(per_round), sum(n >= 1 for n in per_round), sum(n >= 2 for n in per_round)


def check_fault(run, manifest, task, case, last_send):
    """Check isolation through the last send and subsequent restoration of each data link."""
    fault = task['fault']
    if task['fault_kind'] == 'provider':
        targets = {v['as_id'] for v in case['validators'] if v['provider'] == fault}
        require(set(manifest['failed_provider_ases']) == targets
                and manifest['failed_provider'] == fault and not manifest['failed_core'],
                'Provider outage differs from case')
    else:
        targets = {fault}
        require(manifest['failed_core'] == fault and not manifest.get('failed_provider'),
                'Transit outage differs from case')
    links = {(r['as_id'], r['device']) for r in manifest['runtime_link_profile']['interfaces']
             if r['as_id'] in targets}
    require(links and not any(dev in ('lo', 'eth0') for _, dev in links),
            'Missing data links or management link selected')
    events = read(run / 'faults.json')
    for down in (True, False):
        changes = [e for e in events if e['down'] is down]
        affected = [(e['as_id'], e['interface']) for e in changes]
        require(len(affected) == len(links) and set(affected) == links, 'Incomplete link changes')
        require(all(e['requested_ns'] <= e['confirmed_ns'] for e in changes), 'Invalid link event timing')
        if down:
            require(max(e['confirmed_ns'] for e in changes) <= manifest['start_ns'],
                    'Outage began after the round clock started')
        else:
            require(min(e['requested_ns'] for e in changes) >= last_send,
                    'Outage ended before the senders finished')
    for file, during in [('interfaces-before.json', False), ('interfaces-after.json', True)]:
        inventory = read(run / file)
        target_links = {(as_id, dev['ifname']) for as_id in targets for dev in inventory[as_id]
                        if dev['ifname'] not in ('eth0', 'lo')}
        require(target_links == links, 'Selected outage omits data interfaces in the live topology')
        for as_id, devices in inventory.items():
            for device in devices:
                port = (as_id, device['ifname'])
                if port in links or device['ifname'] == 'eth0':
                    expected_up = not (during and port in links)
                    require(('UP' in device['flags']) == expected_up, f'Unexpected link state on {port}')


def per_turn(case, schedule, task):
    """Independently reconstruct hub membership, quota preservation and usable routes."""
    require((case['n'], case['N'], case['T'], case['tau']) == (36, 3, 24, 20), 'Unexpected study design')
    require(len(schedule) == case['T'], 'Wrong number of rounds')
    allocated = Counter(v for group in case['original_draw'] for v in group)
    require(Counter(v for group in schedule for v in group) == allocated and
            [allocated[i] for i in range(case['n'])] == case['q'], 'Validator turns changed')
    by_turn = {}
    modeled = []
    for t, group in enumerate(schedule):
        hub = case['regions'][case['original_draw'][t][0]]
        require(len(group) == len(set(group)) == case['N'] and all(case['regions'][v] == hub for v in group),
                f'Invalid group for round {t}')
        require(all(case['cost'][v][u] <= case['tau'] for v, u in product(group, repeat=2)),
                f'Group exceeds delay limit in round {t}')
        available = 0
        for v in group:
            validator, routes = case['validators'][v], case['routes'][v]
            require(routes['validator'] == validator['id'] and len(routes['services']) == 2
                    and all(len(paths) == 1 for paths in routes['services']), 'Invalid shared-route case')
            paths = {paths[0]['receiver']: paths[0] for paths in routes['services']}
            require(set(paths) == set(RECEIVERS), 'Wrong destinations in shared-route case')
            by_turn[t, validator['id']] = {r: path['fingerprint'] for r, path in paths.items()}
            available += (validator['provider'] != task['fault'] if task['fault_kind'] == 'provider'
                          else all(path['core'] != task['fault'] for path in paths.values()))
        modeled.append(available)
    require(modeled == task['modeled_survivors'], 'Frozen predictions disagree with assigned paths')
    return by_turn, modeled


def score_attempt(run, case_path, task, entry):
    lab.verify_recording(run)
    m, s = read(run / 'manifest.json'), read(run / 'summary.json')
    case = read(case_path / 'instance.json')
    selected, = [p for p in read(case_path / 'schedules.json') if p['method'] == task['method']]
    require(lab.file_sha256(case_path / 'instance.json') == m['instance_sha256']
            and read(run / 'schedule.json') == selected, 'Case or schedule changed between runs')
    require(m['method'] == task['method'] and m['run'] == entry['run'] == s['run']
            and m['task_id'] == run.name and case['measurement']['seed'] == task['seed'],
            'Mismatched run or seed')
    require((m['interval_ns'], m['deadline_ns'], m['payload_bytes'], m['warmup_payloads'])
            == (1_000_000_000, 750_000_000, PAYLOAD_BYTES, 1), 'Unexpected timing or payload settings')
    by_turn, modeled = per_turn(case, selected['schedule'], task)
    origin = {key: m['start_ns'] + key[0] * m['interval_ns'] for key in by_turn}
    sends = read_lines(run / 'senders.jsonl')
    starts = [e for e in sends if e['kind'] == 'start']
    require(all(e['run'] == m['run'] for e in sends) and
            Counter((e['round'], e['validator']) for e in starts) == Counter(by_turn.keys()),
            'Sender starts differ from scheduled turns')
    skews = []
    for e in starts:
        planned = origin[e['round'], e['validator']]
        require(e['planned_ns'] == planned and e['actual_ns'] >= planned, 'Invalid sender clock')
        skews.append(e['actual_ns'] - planned)
    skew_ms = max(skews) / 1e6
    passed_gate = skew_ms <= 5
    require(passed_gate == entry['accepted'] == s['accepted']
            and skew_ms == entry['max_start_skew_ms'] == s['max_start_skew_ms'], 'Incorrect timing gate')
    attempted = set()
    last_send = max(e['actual_ns'] for e in starts)
    for e in (e for e in sends if e['kind'] == 'attempt'):
        turn = (e['round'], e['validator'])
        receiver = e['receiver'].split(',')[0]
        require(turn in by_turn and by_turn[turn].get(receiver) == e['path'] and e['attempt'] == 0
                and e['application_bytes_offered'] == PAYLOAD_BYTES
                and origin[turn] <= e['start_ns'] <= e['end_ns'], 'Invalid send attempt')
        require((turn, receiver) not in attempted, 'Duplicate single-route send')
        attempted.add((turn, receiver))
        last_send = max(last_send, e['end_ns'])
    if passed_gate:
        require(len(attempted) == len(by_turn) * len(RECEIVERS), 'Missing send to a receiver')
    check_fault(run, m, task, case, last_send)
    received = defaultdict(dict)
    for e in read_lines(run / 'receipts.jsonl'):
        turn, receiver = (int(e['round']), e['validator']), e['receiver']
        require(e['kind'] == 'receipt' and e['run'] == m['run'] and (turn, receiver) in attempted
                and by_turn[turn][receiver] == e['path'] and int(e['attempt']) == 0,
                'Receipt does not belong to a scheduled send')
        if e['verified']:
            require(e['hash'] == PAYLOAD_HASH and e['bytes'] == PAYLOAD_BYTES
                    and origin[turn] <= e['handler_ns'] <= e['first_byte_ns'] <= e['complete_ns'],
                    'Invalid verified payload or receiver time')
            offset = e['complete_ns'] - origin[turn]
            received[turn][receiver] = min(received[turn].get(receiver, offset), offset)
    rows = []
    for cutoff in CUTOFFS_MS:
        delivered = [0] * case['T']
        for turn, offsets in received.items():
            if len(offsets) == len(RECEIVERS) and max(offsets.values()) <= cutoff * 1_000_000:
                delivered[turn[0]] += 1
        totals, one, two = counts(delivered)
        available, model_one, model_two = counts(modeled)
        rows.append(dict(seed=task['seed'], outage=task['fault'], policy=task['method'],
                         cutoff_ms=cutoff, rounds=case['T'], proposals=len(by_turn), delivered=totals,
                         rounds_ge1=one, rounds_ge2=two, available=available,
                         modeled_rounds_ge1=model_one, modeled_rounds_ge2=model_two))
    require(s['proposals_delivered'] == totals and s['observed_unsuccessful'] == [3 - n for n in delivered]
            and s['predicted_unavailable'] == [3 - n for n in modeled], 'Replay and independent counts disagree')
    return rows


def totals_by_outage(rows):
    result = []
    for outage, policy, cutoff in product((*PROVIDERS, 'providers_pooled', TRANSIT), POLICIES, CUTOFFS_MS):
        faults = PROVIDERS if outage == 'providers_pooled' else (outage,)
        sample = [r for r in rows if r['outage'] in faults and r['policy'] == policy and r['cutoff_ms'] == cutoff]
        require(len(sample) == 3 * len(faults), 'Missing or duplicate seed')
        row = dict(outage=outage, policy=policy, cutoff_ms=cutoff)
        for key in ('rounds', 'proposals', 'delivered', 'rounds_ge1', 'rounds_ge2',
                    'available', 'modeled_rounds_ge1', 'modeled_rounds_ge2'):
            row[key] = sum(r[key] for r in sample)
        result.append(row)
    return result


def write_latex(path, rows):
    by_key = {(r['outage'], r['cutoff_ms'], r['policy']): r for r in rows}
    lines = [r'{\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.18}',
             r'\begin{tabular}{|l|r|rr|rr|}\hline',
             r'\textbf{Outage} & \textbf{Cutoff} & \multicolumn{2}{c|}{\textbf{At least one proposal}} & '
             r'\multicolumn{2}{c|}{\textbf{At least two proposals}} \\',
             r' & (ms) & Transit only & Joint & Transit only & Joint \\ \hline']
    for outage, label in [('providers_pooled', 'Hosting'), (TRANSIT, 'Transit')]:
        rounds = by_key[outage, CUTOFFS_MS[0], 'transit']['rounds']
        for i, cutoff in enumerate(CUTOFFS_MS):
            first = rf'\multirow{{3}}{{*}}{{\shortstack[l]{{{label}\\{rounds} rounds}}}}' if i == 0 else ''
            transit, joint = (by_key[outage, cutoff, p] for p in POLICIES)
            lines.append(f'{first} & {cutoff} & {transit["rounds_ge1"]} & {joint["rounds_ge1"]} & '
                         f'{transit["rounds_ge2"]} & {joint["rounds_ge2"]}' + r' \\')
        lines.append(r'\hline')
    lines.append(r'\end{tabular}}')
    path.write_text('\n'.join(lines) + '\n')


def write_hosting_distribution(path, rows):
    hosting = {r['policy']: r for r in rows
               if r['outage'] == 'providers_pooled' and r['cutoff_ms'] == 750}
    replays = hosting['transit']['rounds']
    require(hosting['full']['rounds'] == replays, 'Expected paired hosting-outage replays')
    lines = [
        '{',
        r'\setlength{\tabcolsep}{5pt}',
        r'\renewcommand{\arraystretch}{1.22}',
        r'\renewcommand{\tabularxcolumn}[1]{m{#1}}',
        r'\begin{tabularx}{\linewidth}{|>{\raggedright\arraybackslash}X|'
        r'*{4}{>{\centering\arraybackslash}m{0.09\linewidth}|}}',
        r'\hline',
        r'\multirow{2}{*}{\shortstack[l]{\textbf{Grouping rule}\\'
        + f'{replays} round replays per rule' + '}}',
        r'  & \multicolumn{4}{c|}{\textbf{Proposals delivered per round}} \\',
        r'\cline{2-5}',
        r'  & 0 of 3 & 1 of 3 & 2 of 3 & 3 of 3 \\',
        r'\hline',
    ]
    for policy, label in [('transit', 'Balance use of the busiest transit network'),
                          ('full', 'Balance use of each hosting provider and transit network')]:
        row = hosting[policy]
        rounds, one, two = row['rounds'], row['rounds_ge1'], row['rounds_ge2']
        require(row['proposals'] == 3 * rounds, 'Expected three scheduled proposals per round')
        # The three threshold counts sum to the number of delivered proposals.
        three = row['delivered'] - one - two
        require(0 <= three <= two <= one <= rounds, 'Inconsistent three-proposer delivery totals')
        lines.extend([label, f'  & {rounds - one} & {one - two} & {two - three} & {three} ' + r'\\',
                      r'\hline'])
    lines.extend([r'\end{tabularx}', '}'])
    path.write_text('\n'.join(lines) + '\n')


def tighter_delay_screen(case, limit_ms=11):
    """Test a necessary quota condition at a tighter RTT limit, keeping hubs and turns fixed."""
    hubs = defaultdict(list)
    for index, turns in enumerate(case['q']):
        if turns:
            hubs[case['regions'][index]].append(index)
    rows = []
    width, costs = case['N'], case['cost']
    providers = [v['provider'] for v in case['validators']]
    for hub, members in sorted(hubs.items()):
        pairs = set(combinations(members, 2))
        excluded = {(u, v) for u, v in pairs if max(costs[u][v], costs[v][u]) > limit_ms}
        cross = {(u, v) for u, v in pairs if providers[u] != providers[v]}
        groups = [g for g in combinations(members, width) if excluded.isdisjoint(combinations(g, 2))]
        single_provider = bool(groups) and all(len({providers[v] for v in g}) == 1 for g in groups)
        turns = Counter()
        for v in members:
            turns[providers[v]] += case['q'][v]
        remainders = {p: n % width for p, n in sorted(turns.items())}
        reason = None
        if not groups:
            reason = 'This hub has allocated turns and no eligible group.'
        elif single_provider and any(remainders.values()):
            reason = 'Every eligible group uses one provider. At least one provider has a turn count that is not divisible by the group size.'
        rows.append(dict(hub=hub, pairs=len(pairs), excluded_pairs=len(excluded), cross_provider_pairs=len(cross),
                         excluded_cross_provider_pairs=len(cross & excluded), eligible_groups=len(groups),
                         all_eligible_groups_single_provider=single_provider if groups else None,
                         provider_turns=dict(sorted(turns.items())), turns_modulo_group_size=remainders,
                         status='infeasible' if reason else 'inconclusive', reason=reason))
    return dict(seed=case['measurement']['seed'], rtt_limit_ms=limit_ms, group_size=width, hubs=rows,
                status='infeasible' if any(row['status'] == 'infeasible' for row in rows) else 'inconclusive')


def report(out):
    plan = read(out / 'plan.json')
    lab.verify_files(out, plan['case_sha256'])
    require((plan['rtt_limit_ms'], plan['deadline_ms'], plan['cutoffs_ms'], plan['start_skew_limit_ms'])
            == (20, 750, list(CUTOFFS_MS), 5), 'Input plan differs from the declared experiment')
    tasks = plan['tasks']
    expected = set(product(range(3), (*PROVIDERS, TRANSIT), POLICIES))
    require(len(tasks) == len(expected) and {(t['seed'], t['fault'], t['method']) for t in tasks} == expected,
            'Plan is missing a policy, outage or seed')
    table, attempts, indexed = [], [], set()
    entries = read(out / 'results.json')
    require(len({e['run'] for e in entries}) == len(entries), 'Duplicate run identifier')
    accepted_ids = [e['task_id'] for e in entries if e['accepted']]
    require(len({t['id'] for t in tasks}) == len(tasks)
            and accepted_ids == [t['id'] for t in tasks], 'Missing or reordered accepted replays')
    link_settings = None
    for task in tasks:
        require(task['fault_kind'] == ('provider' if task['fault'] in PROVIDERS else 'transit')
                and task['case'] == f"seed-{task['seed']}", 'Invalid condition')
        runs = [e for e in entries if e['task_id'] == task['id']]
        require(1 <= len(runs) <= plan['max_timed_attempts'], 'Invalid number of completed attempts')
        require(runs[-1]['accepted'] and not any(e['accepted'] for e in runs[:-1]) and
                [e['attempt'] for e in runs] == sorted({e['attempt'] for e in runs}),
                'Selection differs from the first eligible attempt')
        for e in runs:
            dest = out / e['run_dir']
            require(dest.parent == out / 'runs' and dest.name == f"{task['id']}-a{e['attempt']}"
                    and dest not in indexed, 'Mismatched attempt directory')
            indexed.add(dest)
            try:
                rows = score_attempt(dest, out / 'cases' / task['case'], task, e)
            except (KeyError, ValueError, FileNotFoundError) as exc:
                raise ValueError(f'{dest.name}: {exc}') from exc
            config = read(dest / 'manifest.json')['runtime_link_profile']
            setting = sorted((r['as_id'], r['device'], r['remote_as'], r['rate_mbit'], r['delay_ms'], r['jitter_ms'])
                             for r in config['interfaces'])
            if link_settings is None:
                link_settings = setting
            require(setting == link_settings, 'Link shaping changed between replays')
            attempts.append(dict(task=task['id'], attempt=e['attempt'], run=e['run'],
                                 recorded=e['run_dir'], accepted=e['accepted'], max_start_skew_ms=e['max_start_skew_ms']))
            if e['accepted']:
                table.extend(rows)
    require(len(entries) == len(attempts), 'Unknown attempt outside the plan')
    require([r['run'] for r in attempts] == [e['run'] for e in entries],
            'Attempt order differs from frozen plan')
    unfinished = []
    for p in sorted((out / 'runs').iterdir()):
        if p.is_dir() and p not in indexed:
            require(not (p / 'summary.json').exists(), f'Unindexed result in {p}')
            unfinished.append(p.name)
    pooled = totals_by_outage(table)
    dest = out / 'report'
    dest.mkdir(exist_ok=True)
    with (dest / 'paired-outage.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pooled[0]))
        writer.writeheader()
        writer.writerows(pooled)
    write_latex(dest / 'paired-outage-delivery.tex', pooled)
    write_hosting_distribution(dest / 'paired-hosting-distribution.tex', pooled)
    result = dict(design='Three 24-round runs, one per seed, for each policy and outage. '
                  'A complete proposal must arrive at both observers by the cutoff. Pooled hosting counts '
                  'sum the same schedules over three separate outages.', attempts=attempts,
                  interrupted_attempts=unfinished, by_seed=table, totals=pooled,
                  tighter_delay_screen=[tighter_delay_screen(read(out / 'cases' / f'seed-{s}' / 'instance.json'))
                                        for s in range(3)])
    (dest / 'paired-outage.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='Paired replay batch directory')
    args = parser.parse_args()
    result = report(args.out.resolve())
    print(f'Verified {len(result["attempts"])} completed attempts. Report: {args.out.resolve() / "report"}')
