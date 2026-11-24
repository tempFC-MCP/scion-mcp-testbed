"""Calculate the delivery tables from both observers' verified receipt times."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path

RECEIVERS = ('1-ff00:0:301', '1-ff00:0:302')
COLUMNS = ('routes', 'group', 'schedule', 'seed', 'round', 'validator',
           'receiver_301_ns', 'receiver_302_ns')
DEADLINE_NS = 750_000_000
CUTOFFS_MS = (220, 500, 750)
METHODS = {'base': 'Baseline', 'infra': 'Best effort', 'strict': 'Strict'}
SHARED_TRANSIT_RULES = {
    'base': ('Original', 'Keep the groups drawn', 'within each hub'),
    'infra': ('Best effort', 'Aim for at most 1 of 3', 'on the failed transit network'),
    'strict': ('Strict', 'Allow at most 2 of 3', 'on the failed transit network'),
}
ROUTE_EXAMPLES = {'Spread routes': ('separate', 'infra'), 'Shared: base': ('shared', 'base'),
           'Shared: best effort': ('shared', 'infra'), 'Shared: strict limit 2': ('shared', 'strict'),
           'Backup routes': ('backup', 'infra')}
POOLED_POLICIES = [('Local', 'base'), ('Local', 'infra'), ('Local', 'strict'),
           ('Distributed', 'base'), ('Distributed', 'infra')]


def export(batch, strict_batch, destination):
    """Project original accepted runs onto the two receipt timestamps for each scheduled turn."""
    digest = hashlib.sha256(bytes([0xa5]) * 262144).hexdigest()
    output = []
    for directory in (batch, strict_batch):
        cases = {(r['case'], r['method']): r for r in json.loads((directory / 'sweep.json').read_text())}
        for run in json.loads((directory / 'results.json').read_text()):
            task = run['task']
            case = cases[task['case'], task['method']]
            variant, method, limit = case['variant'], case['method'], case['rtt_limit']
            if not task['fault'] or case['providers'] != 3 or method not in ('infra', 'base', 'strict'):
                continue
            if variant != 'shared' and (case['seed'] != 0 or limit != 20 or method != 'infra'):
                continue
            if limit not in (20, 260):
                continue
            dest = directory / 'runs' / task['id']
            manifest = json.loads((dest / 'manifest.json').read_text())
            if (not run['summary']['accepted'] or manifest['deadline_ns'] != DEADLINE_NS
                    or manifest['payload_bytes'] != 262144):
                raise ValueError(f'Unexpected timing or payload settings in {dest.name}')
            instance = json.loads((directory / 'cases' / task['case'] / 'instance.json').read_text())
            schedule = json.loads((dest / 'schedule.json').read_text())['schedule']
            times = {(t, instance['validators'][v]['id']): {} for t, group in enumerate(schedule) for v in group}
            for line in (dest / 'receipts.jsonl').read_text().splitlines():
                receipt = json.loads(line)
                if not receipt['verified']:
                    continue
                t, v, receiver = int(receipt['round']), receipt['validator'], receipt['receiver']
                offset = receipt['complete_ns'] - manifest['start_ns'] - t * manifest['interval_ns']
                if (receipt['hash'] != digest or receipt['bytes'] != 262144 or offset < 0
                        or receiver not in RECEIVERS or receipt['run'] != manifest['run']):
                    raise ValueError(f'Invalid verified receipt in {dest.name}')
                prev = times[t, v].get(receiver)
                times[t, v][receiver] = offset if prev is None else min(prev, offset)
            label = 'Local' if limit == 20 else 'Distributed'
            for (t, v), received in times.items():
                output.append((variant, label, method, case['seed'], t, v,
                               *(received.get(receiver, '') for receiver in RECEIVERS)))
    with destination.open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(COLUMNS)
        writer.writerows(sorted(output))


def recount(path, cutoff_ms=750):
    """Count an entire proposal only after both observers have received it."""
    cutoff_ns = cutoff_ms * 1_000_000
    if not 0 < cutoff_ns <= DEADLINE_NS:
        raise ValueError('Receipt cutoff must fall within the 750 ms recording window')
    turns = defaultdict(dict)
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError('Unexpected receipt CSV columns')
        for row in reader:
            key = (row['routes'], row['group'], row['schedule'], int(row['seed']))
            t, v = int(row['round']), row['validator']
            if (t, v) in turns[key]:
                raise ValueError(f'Duplicate turn: {key}, {t}, {v}')
            times = [int(row[c]) for c in COLUMNS[-2:] if row[c]]
            if any(x < 0 for x in times):
                raise ValueError(f'Negative completion time: {key}, {t}, {v}')
            turns[key][t, v] = len(times) == 2 and max(times) <= cutoff_ns
    histograms = {}
    for key, observations in turns.items():
        by_round = defaultdict(list)
        for (t, _), delivered in observations.items():
            by_round[t].append(delivered)
        if set(by_round) != set(range(24)) or any(len(items) != 3 for items in by_round.values()):
            raise ValueError(f'Expected 24 rounds of three distinct proposers: {key}')
        missing = [items.count(False) for items in by_round.values()]
        histograms[key] = [missing.count(i) for i in range(4)]
    expected = {(r, 'Local', m, 0) for r, m in ROUTE_EXAMPLES.values()}
    expected.update(('shared', g, m, s) for g, m in POOLED_POLICIES for s in range(3))
    if set(histograms) != expected:
        raise ValueError('Unexpected set of runs for the delivery tables')
    local = {label: histograms[r, 'Local', m, 0] for label, (r, m) in ROUTE_EXAMPLES.items()}
    pooled = []
    for group, method in POOLED_POLICIES:
        runs = [histograms['shared', group, method, seed] for seed in range(3)]
        hist = list(map(sum, zip(*runs)))
        pooled.append(dict(group=group, method=method, rounds=sum(hist), loss_histogram=hist,
                           delivered=sum((3 - i) * count for i, count in enumerate(hist))))
    return local, pooled


def cutoff_results(path):
    rows = []
    for cutoff in CUTOFFS_MS:
        routes, pooled = recount(path, cutoff)
        for result in pooled:
            if result['group'] == 'Local':
                three, two, one, _ = result['loss_histogram']
                rows.append(dict(cutoff_ms=cutoff, method=result['method'],
                                 delivered=result['delivered'], at_least_one=one + two + three,
                                 at_least_two=two + three))
    # CUTOFFS_MS ends at the configured recording deadline used for the appendix.
    return rows, routes, pooled


def write_cutoff_table(report, rows):
    lines = [r'{\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.13}',
             r'\begin{tabular}{|l|c|c|c|c|}', r'\hline',
             r'\shortstack[l]{\textbf{Grouping rule}\\for three proposers} & '
             r'\shortstack{\textbf{Time from}\\\textbf{round start}\\(ms)} & '
             r'\shortstack{\textbf{Proposals}\\\textbf{delivered}\\out of 216} & '
             r'\multicolumn{2}{c|}{\shortstack{\textbf{Rounds out of 72}\\with at least this many\\proposals delivered}} \\',
             r'\cline{4-5}', r' & & & 1 of 3 & 2 of 3 \\', r'\hline']
    for method, (name, *description) in SHARED_TRANSIT_RULES.items():
        rule = r'\\'.join([rf'\textbf{{{name}}}', *description])
        group = [row for row in rows if row['method'] == method]
        for index, row in enumerate(group):
            label = rf'\multirow{{{len(group)}}}{{*}}{{\shortstack[l]{{{rule}}}}}' if index == 0 else ''
            lines.append(f"{label} & {row['cutoff_ms']} & {row['delivered']} & "
                         f"{row['at_least_one']} & {row['at_least_two']} " + r'\\')
            lines.append(r'\hline' if index == len(group) - 1 else r'\cline{2-5}')
    lines.append(r'\end{tabular}}')
    (report / 'scion-local-delivery.tex').write_text('\n'.join(lines) + '\n')


def empty_rounds_rows(pooled):
    for row in pooled:
        counts = row['loss_histogram']
        missed = sum(i * count for i, count in enumerate(counts))
        if len(counts) != 4 or sum(counts) != 72 or missed + row['delivered'] != 216:
            raise ValueError('Expected 72 rounds of three proposals for each pooled delivery result')
    empty = {(row['group'], row['method']): row['loss_histogram'][3] for row in pooled}
    return [(label, empty[group, 'base'], empty[group, 'infra'])
            for group, label in [('Local', 'Local'), ('Distributed', 'Geographic')]]


def write_empty_rounds_table(report, pooled):
    lines = [r'{\setlength{\tabcolsep}{8pt}\renewcommand{\arraystretch}{1.2}',
             r'\begin{tabular}{|l|c|c|}\hline',
             r'\textbf{Proposer group} & \multicolumn{2}{c|}{\textbf{Empty rounds out of 72}} \\',
             r'\cline{2-3} & Original grouping & Best effort \\ \hline']
    for group, original, best_effort in empty_rounds_rows(pooled):
        lines.append(f'{group} & {original} & {best_effort} ' + r'\\ \hline')
    lines.append(r'\end{tabular}}')
    (report / 'scion-loss-distribution.tex').write_text('\n'.join(lines) + '\n')


def supplementary_rows(routes, pooled):
    route_rows = []
    for label, (group, method) in ROUTE_EXAMPLES.items():
        counts = routes[label]
        delivered = sum((3 - i) * n for i, n in enumerate(counts))
        name = {'separate': 'Spread', 'shared': 'Shared', 'backup': 'Backup'}[group]
        route_rows.append((name, METHODS[method], counts, delivered))
    pooled_rows = [('Local' if item['group'] == 'Local' else 'Geographic', METHODS[item['method']],
                    item['loss_histogram'], item['delivered']) for item in pooled]
    return route_rows, pooled_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path,
                        help='Path to batch/report/recorded-delivery.csv from your expanded experiment')
    parser.add_argument('--tex-dir', type=Path, help='Write Tables 1 and 6 here')
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f'No receipt file at {args.csv}. Use the report from a completed expanded experiment.')
    rows, routes, pooled = cutoff_results(args.csv)

    route_rows, pooled_rows = supplementary_rows(routes, pooled)
    if args.tex_dir:
        args.tex_dir.mkdir(parents=True, exist_ok=True)
        write_cutoff_table(args.tex_dir, rows)
        write_empty_rounds_table(args.tex_dir, pooled)

    print('Table 1. Local shared-transit outage, same 72 rounds at each receipt cutoff')
    for name, *description in SHARED_TRANSIT_RULES.values():
        print(f"{name}: {' '.join(description)}.")
    print(f'{"From round start (ms)":>21}  {"Grouping rule":13} {"Proposals / 216":>15} {"Rounds with >=1 of 3":>20} {"Rounds with >=2 of 3":>20}')
    for row in rows:
        name = SHARED_TRANSIT_RULES[row['method']][0]
        print(f"{row['cutoff_ms']:21}  {name:13} {row['delivered']:15}"
              f" {row['at_least_one']:20} {row['at_least_two']:20}")
    print('\nTable 6. Empty rounds out of 72 at 750 ms')
    print(f'{"Proposer group":16} {"Original grouping":>17} {"Best effort":>12}')
    for group, original, best_effort in empty_rounds_rows(pooled):
        print(f'{group:16} {original:17} {best_effort:12}')
    for title, cases in [('Pooled shared-transit comparison (72 rounds)', pooled_rows),
                         ('Single-seed route comparison (24 rounds)', route_rows)]:
        print(f'\nSupplementary results. {title} at 750 ms')
        print(f'{"Group and schedule":24} {"Empty rounds":>12} {"Missed proposals":>17}')
        for group, policy, counts, _ in cases:
            label = f'{group} {policy}'
            print(f'{label:24} {counts[3]:12} {sum(i * n for i, n in enumerate(counts)):17}')
