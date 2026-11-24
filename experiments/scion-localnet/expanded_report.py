"""Reconstruct the expanded measurements, delivery tables and peer timing figure."""
import argparse
import json
from pathlib import Path
import shutil

import altair as alt
import numpy as np

import lab
import check_expanded
import check_coordination
import measured
import peer_exchange
import recorded_delivery
from coordination_report import save, table, TEAL, BLUE, CORAL


def main(out, publish=False, strict_out=None, coverage_out=None, peer_out=None):
    check_expanded.membership_checks()
    check_expanded.schedules(out)
    check_coordination.peer_records(out)
    check_coordination.payload_records(out)
    strict_results=[]
    if strict_out:
        check_expanded.schedules(strict_out)
        check_coordination.payload_records(strict_out)
        strict_results=json.loads((strict_out/'results.json').read_text())
    report=out/'report';report.mkdir(exist_ok=True)
    rows=json.loads((out/'sweep.json').read_text())
    table(report/'schedules.csv',rows)
    coverage=[]
    for limit in (20,100,180,260):
        subset=[r for r in rows if r['providers']==3 and r['variant']=='shared' and r['rtt_limit']==limit]
        base=[r for r in subset if r['method']=='base'];full=[r for r in subset if r['method']=='infra']
        assert len(base)==len(full)==3
        assert {r['seed'] for r in base} == {r['seed'] for r in full} == {0, 1, 2}
        values=[r['mean_regions'] for r in full]
        coverage.append(dict(limit=limit,mean=float(np.mean(values)),low=min(values),high=max(values),label=f'{np.mean(values):.2f}'))
    table(report/'coverage.csv',coverage)
    if coverage_out:
        from coverage_followup import summarize_allocations
        measured.write(report / 'allocation-sensitivity.json', summarize_allocations(coverage_out, out))

    outcomes=[]
    all_results=json.loads((out/'results.json').read_text())
    for item in all_results:
        task,summary=item['task'],item['summary']
        case=next(r for r in rows if r['case']==task['case'] and r['method']==task['method'])
        losses=summary['observed_unsuccessful']
        outcomes.append(dict(case=task['case'],seed=case['seed'],providers=case['providers'],variant=case['variant'],limit=case['rtt_limit'],method=task['method'],fault=bool(task['fault']),
            delivered=summary['proposals_delivered'],attempts=summary['proposal_attempts'],
            loss_histogram=[losses.count(i) for i in range(4)],
            exact_prediction=losses==summary['predicted_unavailable'],
            max_start_skew_ms=summary['max_start_skew_ms'],mean_delivered_regions=summary['mean_delivered_regions']))
    table(report/'outcomes.csv',[dict(r,loss_histogram=json.dumps(r['loss_histogram'])) for r in outcomes])
    pooled=[]
    for limit,title in [(20,'Local'),(260,'Distributed')]:
        for method in ('base', 'infra'):
            selected=[r for r in outcomes if r['providers']==3 and r['variant']=='shared' and r['limit']==limit and r['method']==method and r['fault']]
            assert len(selected) == 3 and {r['seed'] for r in selected} == {0, 1, 2}
            hist=np.sum([r['loss_histogram'] for r in selected],axis=0)
            pooled.append(dict(group=title,method=method,rounds=int(hist.sum()),loss_histogram=hist.tolist(),delivered=sum(r['delivered'] for r in selected)))
        if limit == 20 and strict_results:
            hist = [sum(r['summary']['observed_unsuccessful'].count(i) for r in strict_results)
                    for i in range(4)]
            pooled.append(dict(group=title, method='strict', rounds=sum(hist), loss_histogram=hist,
                               delivered=sum(r['summary']['proposals_delivered'] for r in strict_results)))
    if strict_out:
        receipt_csv = report / 'recorded-delivery.csv'
        recorded_delivery.export(out, strict_out, receipt_csv)
        cutoff_rows, _, from_receipts = recorded_delivery.cutoff_results(receipt_csv)
        recorded_delivery.write_cutoff_table(report, cutoff_rows)
        if pooled != from_receipts:
            raise ValueError('Pooled results disagree with per-observer receipt times')
    recorded_delivery.write_empty_rounds_table(report, pooled)

    peers=json.loads((out/'peer-results.json').read_text());cdf=[];peer_summary=[]
    origins={r['task']['id']:out for r in peers}
    if peer_out:
        check_coordination.peer_records(peer_out)
        additional=json.loads((peer_out/'peer-results.json').read_text())
        assert len(additional)==3
        assert not ({r['task']['id'] for r in additional} & set(origins))
        for r in additional:
            assert json.loads((peer_out/'peers'/r['task']['id']/'manifest.json').read_text())['admission_rtt_ms']==180
            origins[r['task']['id']]=peer_out
        peers+=additional
    for item in peers:
        directory=origins[item['task']['id']]/'peers'/item['task']['id']
        lab.verify_recording(directory)
        actual=peer_exchange.verify(directory)
        assert actual==item['summary'] and actual['accepted']
    limits=(20,100,180,260) if peer_out else (20,100,260)
    for limit in limits:
        selected=[r['summary'] for r in peers if next(s for s in rows if s['case']==r['task']['case'])['rtt_limit']==limit]
        assert len(selected)==3
        values=sorted(v for r in selected for v in r['group_completion_ms'] if v is not None)
        total=sum(r['rounds'] for r in selected)
        label=f'{limit} ms RTT limit'
        cdf.append(dict(ms=0,fraction=0,limit=label))
        cdf.extend(dict(ms=v,fraction=(i+1)/total,limit=label) for i,v in enumerate(values))
        cdf.append(dict(ms=400,fraction=len(values)/total,limit=label))
        peer_summary.append(dict(
            limit=limit, rounds=total, completed=len(values),
            p50_ms=float(np.percentile(values, 50)) if values else None,
            p95_ms=float(np.percentile(values, 95)) if values else None,
            max_ms=max(values, default=None),
            max_start_skew_ms=max(r['max_start_skew_ms'] for r in selected)))
    table(report/'peer-summary.csv',peer_summary)
    chart_end = max(160, max((r['max_ms'] or 0) for r in peer_summary))
    chart=alt.Chart(alt.Data(values=cdf)).mark_line(interpolate='step-after',strokeWidth=2.8,clip=True).encode(
        x=alt.X('ms:Q',title='Time to receive all six peer messages (ms)',scale=alt.Scale(domain=[0,chart_end])),
        y=alt.Y('fraction:Q',title='Completed rounds',axis=alt.Axis(format='%'),scale=alt.Scale(domain=[0,1.03])),
        color=alt.Color('limit:N',scale=alt.Scale(domain=[f'{v} ms RTT limit' for v in limits],range=[TEAL,BLUE,'#8d679d',CORAL] if peer_out else [TEAL,BLUE,CORAL])))
    save(chart.properties(width=570,height=230),report,'scion-peer-exchange')
    summary=dict(schedules=len(rows),cases=len({r['case'] for r in rows}),payload_runs=len(outcomes),peer_runs=len(peers),
        proposal_attempts=sum(r['attempts'] for r in outcomes),peer_messages=sum(r['summary']['expected_messages'] for r in peers),
        coverage=coverage,losses=pooled,peers=peer_summary,
        max_payload_start_skew_ms=max(r['max_start_skew_ms'] for r in outcomes),
        prediction_matches=sum(r['exact_prediction'] for r in outcomes),outcomes=outcomes)
    if strict_out:
        summary['local_delivery_by_cutoff'] = cutoff_rows
    summary['strict_results']=strict_results
    summary['allocation_sensitivity']=str(coverage_out) if coverage_out else None
    summary['peer_followup']=str(peer_out) if peer_out else None
    summary['total_payload_runs']=len(outcomes)+len(strict_results)
    summary['total_proposal_attempts']=summary['proposal_attempts']+sum(r['summary']['proposal_attempts'] for r in strict_results)
    summary['rejected_runs']=json.loads((out/'rejected-runs.json').read_text()) if (out/'rejected-runs.json').exists() else []
    measured.write(report/'summary.json',summary)
    measured.write(report/'provenance.json',dict(
        snapshot_sha256=lab.file_sha256(Path(json.loads((out / 'plan.json').read_text())['snapshot']) / 'manifest.json'),
        plan_sha256=lab.file_sha256(out / 'plan.json'),
        report_sha256=lab.file_sha256(Path(__file__)),
        result_sha256=lab.file_sha256(out / 'results.json'),
        peer_result_sha256=lab.file_sha256(out / 'peer-results.json'),
        followup_peer_sha256=lab.file_sha256(peer_out / 'peer-results.json') if peer_out else None,
        allocation_sensitivity_sha256=lab.file_sha256(coverage_out / 'results.json') if coverage_out else None,
        scope='Single-host container deployment with emulated regions; fixed observation receivers are not consensus voters.',
        inference='Three deployment seeds; allocation sensitivity uses ten seeds. Ranges are descriptive, not population confidence intervals.',
        figures='Peer completion time covers the sampled RTT limits. Figure 3 uses the separate 24-validator reach experiment.',
        methods='Local base preserves input hub order; distributed base maximizes coverage. Infrastructure-aware uses the same allocations and geography target.',
        backups='Known-fault surviving-path selection; does not measure automatic detection or retry.'))
    if publish:
        dest=Path(__file__).resolve().parents[2]/'figures'
        dest.mkdir(exist_ok=True)
        if strict_out:
            shutil.copy2(report/'scion-local-delivery.tex', dest/'scion-local-delivery.tex')
        for name in ('scion-loss-distribution.tex', 'scion-peer-exchange.pdf'):
            shutil.copy2(report / name, dest / name)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('outcomes','rejected_runs','strict_results')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--publish',action='store_true');p.add_argument('--strict-out',type=Path)
    p.add_argument('--coverage-out',type=Path);p.add_argument('--peer-out',type=Path)
    a=p.parse_args();main(a.out,a.publish,a.strict_out,a.coverage_out,a.peer_out)
