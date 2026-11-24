"""Verify stored schedules and generate manuscript tables/plots from recorded runs."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean, median
import numpy as np
from scheduler import adopt_schedule, metrics, verify_recorded_metrics, weighted_score
from local_hub_comparison import OUT as LOCAL_RESULTS, independent_checks, plot

ROOT=Path(__file__).resolve().parents[1]


def excess_by_asn(schedule, instance):
    """Score one physical AS outage across its hosting and routing roles."""
    affected = {}
    for (label,), members in zip(instance['scenarios'], instance['F'], strict=True):
        role, asn = label.split(':')
        assert role in {'host', 'AS'}
        affected.setdefault(asn, np.zeros_like(members, dtype=bool))
        affected[asn] |= members.astype(bool)
    # The public-routing experiments allow one affected proposer per outage.
    assert np.all(instance['budget'] == 1)
    groups = np.asarray(schedule)
    return sum(int(np.maximum(0, members[groups].sum(axis=1) - 1).sum())
               for members in affected.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ripe', action='store_true', help='Summarize the independent RIPE run.')
    public = parser.parse_args().ripe
    output=ROOT/'results' / ('ripe' if public else '')
    path=output/'experiments.json'
    data=json.loads(path.read_text()); rows=data['results']
    instances={}
    for row in rows:
        i=row['instance']
        if i not in instances:
            d=json.loads((output/f'instance-{i:02d}.json').read_text())
            for k in ['regions','F','budget','provider_F','provider_budget','cost','initial','q']:
                d[k]=np.array(d[k])
            instances[i]=d
        if 'schedule' in row:
            actual=metrics(np.array(row['schedule']),instances[i])
            verify_recorded_metrics(row, actual, approximate_floats=True)
            if row['method']=='MILP':
                val=weighted_score(actual)
                assert row['dual_bound'] is None or row['dual_bound'] <= val+1e-5
                if row['status']==0: assert abs(row['objective']-val)<1e-5
    data['environment']['logical_cpus_visible']=os.cpu_count()
    info=Path('/proc/cpuinfo').read_text()
    data['environment']['cpu_model']=next((l.split(':',1)[1].strip() for l in info.splitlines()
                                           if l.startswith('model name')),'unknown')
    data['validation']={'verified_schedules':sum('schedule' in r for r in rows),
                        'instances':len(instances),'method_runs':len(rows)}
    path.write_text(json.dumps(data,indent=2)+'\n')
    if public:
        summary = {method: mean(excess_by_asn(row['schedule'], instances[row['instance']])
                                for row in rows if row['method'] == method)
                   for method in ('geo', 'provider', 'full')}
        (output/'summary.json').write_text(json.dumps({'mean_excess_by_unique_asn': summary}, indent=2)+'\n')
        print('Mean excess exposure by unique AS across the three runs:', summary)
        return
    base=[r for r in rows if r['source']=='synthetic' and r['backup_fraction']==.5
          and r['h'] is None and r['tau']==180 and 'g' not in r and 'allocation_tau' not in r]
    sizes=sorted({r['n'] for r in base})
    # Keep construction quality and the deployable decision separate. A solver
    # can stop with an incumbent that is worse than the base protocol's input.
    adoption = []
    for i in sorted({r['instance'] for r in base}):
        candidates = {r['method']:r.get('schedule') for r in base if r['instance']==i}
        instance = instances[i]
        for methods in [('full',), ('MILP',), ('full', 'MILP')]:
            winner, _, stats = adopt_schedule(instance, {m:candidates[m] for m in methods})
            adoption.append(dict(group='geographic', instance=i, candidates=list(methods),
                                 adopted=winner, score=weighted_score(stats), compliant=stats['compliant']))
    local_path=LOCAL_RESULTS/'results.json'
    if not local_path.exists():
        raise FileNotFoundError('Run scripts/local_hub_comparison.py before generating Figure 2')
    local_rows=json.loads(local_path.read_text())['results']
    for row in local_rows:
        ins=json.loads((LOCAL_RESULTS/f'input-{row["n"]}-{row["N"]}-{row["seed"]}.json').read_text())
        for key in ['initial','q','F','cost','budget','regions','provider_F',
                    'provider_budget','eligible','hub_sequence']:
            ins[key]=np.array(ins[key])
        actual=independent_checks(np.array(row['schedule']),ins)
        verify_recorded_metrics(row, actual)
        if row['method']=='local_full':
            winner, _, stats = adopt_schedule(ins, {'full':row['schedule']})
            adoption.append(dict(group='local', n=row['n'], N=row['N'], seed=row['seed'],
                                 candidates=['full'], adopted=winner, score=weighted_score(stats),
                                 compliant=stats['compliant']))
    (ROOT/'results'/'adoption.json').write_text(json.dumps(adoption,indent=2)+'\n')
    plot(local_rows,base,destination=ROOT/'figures'/'scheduler-results')
    lines=[r'\begin{table}[t]',r'\centering',
           r'\renewcommand{\arraystretch}{1.2}',
           r'\setlength{\tabcolsep}{6pt}',
           r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}rcrcr@{}}',r'\toprule',
           r'\shortstack[r]{\textbf{Validators}\\$|V|$} & '
           r'\shortstack[c]{\textbf{Proposers}\\$N$} & '
           r'\shortstack[r]{\textbf{Model}\\\textbf{variables}} & '
           r'\shortstack[c]{\textbf{Optimal}\\\textbf{runs}} & '
           r'\shortstack[r]{\textbf{Median time}\\seconds} \\',r'\midrule']
    for n in sizes:
        if n != 50:
            lines.append(r'\addlinespace[0.5em]')
        for N in [4,8]:
            a=[r for r in base if r['method']=='MILP' and r['n']==n and r['N']==N]
            lines.append(f'{n} & {N} & {a[0]["variables"]:,} & {sum(r["status"]==0 for r in a)} of 3 & {median(r["total_seconds"] for r in a):.2f}'+r' \\')
    lines += [r'\bottomrule',r'\end{tabular*}',
              r'\caption{MILP size, runtime, and optimality.}',
              r'\label{tab:solver}',r'\end{table}']
    (ROOT/'figures'/'solver-results.tex').write_text('\n'.join(lines)+'\n')
    summary={}
    for method in ['random','latency','geo','provider','full','MILP']:
        a=[r for r in base if r['method']==method and 'excess' in r]
        summary[method]={k:mean(r[k] for r in a) for k in ['excess','geo_deficit','mean_regions','mean_pair_ms']} if a else None
    solver=[r for r in rows if r['method']=='MILP']
    summary['solver_statuses']={str(k):sum(r['status']==k for r in solver) for k in [0,1,2]}
    summary['heuristic_runtime_range']=[min(r['seconds'] for r in rows if r['method']=='full'),
                                        max(r['seconds'] for r in rows if r['method']=='full')]
    summary['checks']=data['validation']
    (ROOT/'results'/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
