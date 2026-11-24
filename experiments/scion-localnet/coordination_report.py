"""Geographic reach figure and independently checked measurements for the delay sweep."""
import argparse
import csv
import json
from pathlib import Path

import altair as alt
import numpy as np

import lab
import coordination
import measured
import peer_exchange

TEAL = '#087f8c'
BLUE = '#4766bc'
CORAL = '#d5684a'
INK = '#24364b'


def style(chart):
    return (chart.configure(font='DejaVu Sans', background='white')
            .configure_view(stroke=None)
            .configure_axis(labelFontSize=12, titleFontSize=13, titleFontWeight='normal',
                            labelColor=INK, titleColor=INK, domain=False, tickColor='#d5dce3',
                            gridColor='#edf0f4', titlePadding=14, labelPadding=7)
            .configure_legend(labelFontSize=12, titleFontSize=12, orient='top',
                              title=None, symbolStrokeWidth=3, padding=14))


def save(chart, directory, name):
    chart = style(chart)
    for suffix in ('svg', 'pdf', 'png'):
        chart.save(directory / f'{name}.{suffix}', **({'scale_factor': 2} if suffix == 'png' else {}))
    chart.save(directory / f'{name}.html', inline=True)


def table(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(out):
    coordination.check(out)
    directory = out / 'report'
    directory.mkdir(exist_ok=True)
    rows = json.loads((out / 'sweep.json').read_text())
    enriched = []
    for row in rows:
        case = out / 'cases' / f'seed-{row["seed"]}-rtt-{row["rtt_limit_ms"]}'
        instance = measured.load_instance(case / 'instance.json')
        selected = next(r for r in json.loads((case / 'schedules.json').read_text()) if r['method'] == row['policy'])
        schedule = np.array(selected['schedule'])
        exposure = instance['F'][:, schedule].sum(axis=2)
        compliant = (exposure <= instance['budget'][:, None]).all(axis=0)
        provider_counts = [len({instance['validators'][v]['provider'] for v in group})
                           for group in schedule]
        forced_local = None
        if all(len(set(instance['regions'][g])) == 1 for g in schedule):
            forced_local = 0
            for region in range(instance['nr']):
                totals = {}
                for v, quota in zip(instance['validators'], instance['q']):
                    if v['region'] == region:
                        totals[v['provider']] = totals.get(v['provider'], 0) + int(quota)
                total = sum(totals.values())
                assert total % instance['N'] == 0
                rounds = total // instance['N']
                forced_local += sum(max(0, q - (instance['N'] - 1) * rounds) for q in totals.values())
        enriched.append(dict(row, infrastructure_compliant_rounds=int(compliant.sum()),
                             single_provider_rounds=provider_counts.count(1),
                             mean_hosting_providers=float(np.mean(provider_counts)),
                             local_single_provider_lower_bound=forced_local))
    table(directory / 'scheduling.csv', [{k: v for k, v in row.items() if not isinstance(v, list)} for row in enriched])

    coverage, diversity = [], []
    for limit in coordination.LIMITS:
        a = [r for r in enriched if r['rtt_limit_ms'] == limit and r['policy'] == 'geo']
        b = [r for r in enriched if r['rtt_limit_ms'] == limit and r['policy'] == 'full']
        assert sorted(r['seed'] for r in a) == sorted(r['seed'] for r in b) == [0, 1, 2]
        for policy, group in [('Geography', a), ('Infrastructure-aware', b)]:
            vals = [r['mean_regions'] for r in group]
            coverage.append(dict(limit=limit, policy=policy, mean=float(np.mean(vals)),
                                 low=min(vals), high=max(vals), label=f'{np.mean(vals):.2f}'))
        g, f = (sum(r['infrastructure_compliant_rounds'] for r in group) / sum(r['rounds'] for r in group) * 100
                for group in (a, b))
        diversity.append(dict(limit=f'{limit} ms', geo=g, full=f,
                              geo_label=f'{g:.1f}%', full_label=f'{f:.1f}%'))
    table(directory / 'infrastructure-summary.csv', diversity)

    # Draw a single curve only when both policies attain the same coverage in every seed.
    by_policy = {p: sorted((r['rtt_limit_ms'], r['seed'], r['mean_regions'])
                           for r in enriched if r['policy'] == p) for p in ('geo', 'full')}
    shared_curve = by_policy['geo'] == by_policy['full']
    plotted = [r for r in coverage if r['policy'] == 'Geography'] if shared_curve else coverage
    if shared_curve:
        plotted = [{k: v for k, v in r.items() if k != 'policy'} for r in plotted]
    table(directory / 'coverage-summary.csv', plotted)
    base = alt.Chart(alt.Data(values=plotted)).encode(
        x=alt.X('limit:Q', title='Permitted pairwise probe RTT (ms)',
                scale=alt.Scale(domain=[10, 275]), axis=alt.Axis(values=coordination.LIMITS, grid=False)))
    band = base.mark_area(color=TEAL, opacity=.10).encode(
        y=alt.Y('low:Q', title='Regions represented per round', scale=alt.Scale(domain=[.85, 3.05]),
                axis=alt.Axis(values=[1, 1.5, 2, 2.5, 3])), y2='high:Q')
    spread = base.mark_rule(color=TEAL, strokeWidth=2, opacity=.45).encode(y='low:Q', y2='high:Q')
    line = base.mark_line(color=TEAL, strokeWidth=3).encode(y='mean:Q')
    points = base.mark_point(color=TEAL, filled=True, size=85, stroke='white', strokeWidth=2).encode(
        y='mean:Q', tooltip=[alt.Tooltip('limit:Q', title='RTT limit (ms)'),
                            alt.Tooltip('mean:Q', title='Mean regions', format='.3f'),
                            alt.Tooltip('low:Q', title='Smallest seed result'),
                            alt.Tooltip('high:Q', title='Largest seed result')])
    labels = base.mark_text(dy=-16, color=INK, fontSize=12).encode(y='high:Q', text='label:N')
    chart = (band + spread + line + points + labels).properties(width=620, height=290)
    if not shared_curve:
        chart = chart.facet(row=alt.Row('policy:N', title=None))
    save(chart, directory, 'geographic-reach')

    peers = []
    if (out / 'peer-results.json').exists():
        for record in json.loads((out / 'peer-results.json').read_text()):
            dest = out / 'peers' / record['task']['id']
            lab.verify_recording(dest)
            summary = peer_exchange.verify(dest)
            if summary != record['summary'] or not summary['accepted']:
                raise ValueError('Peer summary mismatch or invalid timing')
            case = record['task']['case'].split('-')
            peers.append(dict(seed=int(case[1]), rtt_limit_ms=int(case[3]),
                              policy=record['task']['method'], **summary))
        table(directory / 'peer-summary.csv', [{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in peers])
    payloads = []
    if (out / 'results.json').exists():
        from check_coordination import payload_records
        results = json.loads((out / 'results.json').read_text())
        if len(results) == len(json.loads((out / 'plan.json').read_text())['tasks']):
            payload_records(out)
            for r in results:
                summary, task = r['summary'], r['task']
                case = task['case'].split('-')
                losses = summary['observed_unsuccessful']
                payloads.append(dict(rtt_limit_ms=int(case[3]), policy=task['method'],
                    failed_core=task['fault'], delivered=summary['proposals_delivered'],
                    scheduled=summary['proposal_attempts'],
                    no_proposals_lost=losses.count(0), one_proposal_lost=losses.count(1),
                    two_proposals_lost=losses.count(2), all_proposals_lost=losses.count(3),
                    mean_delivered_regions=summary['mean_delivered_regions'],
                    max_start_skew_ms=summary['max_start_skew_ms']))
            table(directory / 'payload-summary.csv', payloads)
    measured.write(directory / 'report-metadata.json', dict(
        schedules=len(enriched), peer_runs=len(peers), payload_runs=len(payloads),
        coverage_band='minimum and maximum across three allocation seeds; not a confidence interval',
        geographic_curve='means at sampled limits; connecting lines do not locate exact transition thresholds',
        infrastructure_metric='each catalogue scenario affects at most its budget of selected proposers',
        peer_metric='all N(N-1) directed 512-byte messages received; denominator includes incomplete rounds',
        catalogue_scope='proposal delivery to two fixed receivers; peer paths are measured separately',
        baseline='one optimizer-selected maximum-coverage schedule per seed and limit; not a random-scheduler average'))
    print(f'Wrote the geographic reach figure and verified data to {directory}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
