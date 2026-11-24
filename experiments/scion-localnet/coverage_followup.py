"""Allocation sensitivity and a separately recorded missing peer-timing case."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import altair as alt
import numpy as np

import coordination
import lab
import measured
from resume_timing import run_peers
import expanded_study
from scheduler import quotas, verify_recorded_metrics
from coordination_report import save, TEAL


def prepare_allocations(source, out):
    out.mkdir(parents=True, exist_ok=False)
    original = source/'cases'/'p3-shared-s0-rtt20'/'instance.json'
    base = measured.load_instance(original)
    rows = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        x = deepcopy(base)
        weights = rng.lognormal(0, 1, x['n'])
        regions = x['regions']
        regional = np.array([weights[regions == r].sum() for r in range(x['nr'])])
        initial = []
        hubs = rng.choice(x['nr'], size=x['T'], p=regional/regional.sum())
        for hub in hubs:
            members = np.flatnonzero(regions == hub)
            p = weights[members]/weights[members].sum()
            initial.append(rng.choice(members, size=x['N'], replace=False, p=p))
        x['initial'] = np.array(initial)
        x['q'] = quotas(x['initial'], x['n'])
        x['g'] = 3
        measured.write(out/f'allocation-{seed}.json',
                       dict(seed=seed, weights=weights, initial=initial, quotas=x['q'],
                            regional_quotas=[int(x['q'][regions == r].sum()) for r in range(3)]))
        previous = 0
        for limit in (20, 100, 180, 260):
            x['tau'] = limit
            answers = [coordination.solve_groups(x, policy, seed) for policy in ('geo', 'full')]
            # Show the geographic optimum; verify whether adding infrastructure
            # changes its attained coverage instead of assuming equality.
            assert answers[0]['solver']['optimal'] and answers[1]['solver']['optimal']
            assert answers[0]['mean_regions'] >= previous
            previous = answers[0]['mean_regions']
            for a in answers:
                rows.append(dict(seed=seed, limit=limit, **a))
            measured.write(out/'results.json', rows)
            print('Allocation', seed, 'limit', limit,
                  'coverage', [a['mean_regions'] for a in answers], flush=True)
    measured.write(out/'manifest.json', dict(
        source=str(original.resolve()), source_sha256=lab.file_sha256(original),
        code_sha256=coordination.code_hashes() | {
            'coverage_followup.py': lab.file_sha256(Path(__file__))},
        seeds=list(range(10)), rounds=base['T'], validators=base['n'], proposers=base['N'],
        allocation='Lognormal(0,1) weights; hub sampled by regional weight; N distinct validators sampled within hub.',
        pairing='The same quotas and measured cost matrix at every admission limit and for both policies.',
        scope='Offline scheduling sensitivity on recorded SCION path measurements; no new delivery replay.',
        band='Minimum and maximum of ten allocation seeds, not a confidence interval or network jitter.'))


def sensitivity(source, out):
    prepare_allocations(source, out)
    render(out, source)


def summarize_allocations(out, source):
    rows = json.loads((out/'results.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    original=source/'cases'/'p3-shared-s0-rtt20'/'instance.json'
    assert lab.file_sha256(original)==manifest['source_sha256']
    base=measured.load_instance(original)
    assert len(rows)==80
    for seed in range(10):
        allocation=json.loads((out/f'allocation-{seed}.json').read_text())
        x=deepcopy(base)
        x['q']=np.array(allocation['quotas'])
        x['initial']=np.array(allocation['initial'])
        x['g']=3
        assert np.array_equal(quotas(x['initial'],x['n']),x['q'])
        for row in [r for r in rows if r['seed']==seed]:
            x['tau']=row['limit']
            actual=measured.metrics(row['schedule'],x)
            verify_recorded_metrics(row, actual)
            objective=10*actual['geo_deficit']+(actual['excess'] if row['method']=='full' else 0)
            assert row['solver']['optimal'] and abs(objective-row['solver']['dual_bound'])<1e-5
    coverage = []
    equal = json.loads((source/'sweep.json').read_text())
    for limit in (20, 100, 180, 260):
        a = [r for r in rows if r['limit'] == limit and r['method'] == 'geo']
        b = [r for r in rows if r['limit'] == limit and r['method'] == 'full']
        assert len(a) == len(b) == 10
        assert {r['seed'] for r in a} == {r['seed'] for r in b} == set(range(10))
        values = [r['mean_regions'] for r in a]
        controls = [r['mean_regions'] for r in equal if r['providers'] == 3
                    and r['variant'] == 'shared' and r['rtt_limit'] == limit and r['method'] == 'base']
        assert len(controls) == 3
        coverage.append(dict(limit=limit, mean=float(np.mean(values)), low=min(values),
                             high=max(values), equal=float(np.mean(controls)),
                             policies_match={r['seed']: r['mean_regions'] for r in a}
                             == {r['seed']: r['mean_regions'] for r in b}))
    return coverage


def render(out, source):
    coverage = summarize_allocations(out, source)
    measured.write(out/'coverage.json', coverage)
    data = alt.Data(values=coverage)
    base = alt.Chart(data).encode(
        x=alt.X('limit:Q', title='Allowed pairwise probe RTT (ms)',
                axis=alt.Axis(values=[20,100,180,260],grid=False),scale=alt.Scale(domain=[5,275])))
    band = base.mark_area(color=TEAL,opacity=.17).encode(
        y=alt.Y('low:Q',title='Regions per proposer group',scale=alt.Scale(domain=[.85,3.15])),y2='high:Q')
    series = [dict(limit=r['limit'],value=r[key],allocation=label)
              for r in coverage for key,label in [('mean','Unequal allocations'),('equal','Equal allocations')]]
    lines = alt.Chart(alt.Data(values=series)).mark_line(strokeWidth=2.5,point=True).encode(
        x=alt.X('limit:Q',title='Allowed pairwise probe RTT (ms)',axis=alt.Axis(values=[20,100,180,260],grid=False)),
        y=alt.Y('value:Q',title='Regions per proposer group',axis=alt.Axis(values=[1,1.5,2,2.5,3])),
        color=alt.Color('allocation:N',title=None,scale=alt.Scale(
            domain=['Unequal allocations','Equal allocations'],range=[TEAL,'#777f89']),
            legend=alt.Legend(orient='bottom')),
        strokeDash=alt.StrokeDash('allocation:N',title=None,scale=alt.Scale(
            domain=['Unequal allocations','Equal allocations'],range=[[1,0],[6,4]])))
    save((band+lines).properties(width=570,height=185),out,'scion-geographic-reach')


def prepare_peer(source, snapshot, out):
    out.mkdir(parents=True, exist_ok=False)
    observations = measured.load_snapshot(snapshot)
    tasks = []
    for seed in range(3):
        name = f'p3-shared-s{seed}-rtt180'
        case = out/'cases'/name
        case.mkdir(parents=True)
        old = measured.load_instance(source/'cases'/name/'instance.json')
        fresh = expanded_study.instance(observations,seed,3,'shared',180)
        for key in ['q','regions','initial','F','budget']:
            assert np.array_equal(old[key],fresh[key]), key
        previous = json.loads((source/'cases'/name/'schedules.json').read_text())
        selected = deepcopy(next(r for r in previous if r['method'] == 'infra'))
        selected.update(measured.metrics(selected['schedule'], fresh))
        measured.write(case/'instance.json',fresh)
        measured.write(case/'schedules.json',[selected])
        tasks.append(dict(id=f'followup-peer180-s{seed}',case=name,method='infra'))
    names = ['coverage_followup.py', 'expanded_study.py', 'resume_timing.py']
    hashes = coordination.code_hashes() | lab.code_hashes(names)
    measured.write(out/'plan.json',dict(tasks=[],snapshot=str(snapshot.resolve()),
        source=str(source.resolve()),preserved='Original 180 ms schedules and quotas; fresh paths and probes.',
        code_sha256=hashes))
    measured.write(out/'peer-plan.json',dict(tasks=tasks,payload_bytes=512,interval_ms=500,
        deadline_ms=400,start_skew_limit_ms=5))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare-allocations','sensitivity','render','prepare-peer','peers'])
    p.add_argument('--source',type=Path,help='Expanded batch generated by this experiment')
    p.add_argument('--snapshot',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.action != 'peers' and a.source is None:
        p.error('--source must point to the expanded batch for this run')
    if a.action == 'prepare-peer' and a.snapshot is None:
        p.error('--snapshot must point to the new path and RTT observations')
    lab.select_profile('expanded')
    if a.action == 'prepare-allocations': prepare_allocations(a.source, a.out)
    elif a.action == 'sensitivity': sensitivity(a.source,a.out)
    elif a.action == 'render': render(a.out,a.source)
    elif a.action == 'prepare-peer': prepare_peer(a.source,a.snapshot,a.out)
    else: run_peers(a.out)
