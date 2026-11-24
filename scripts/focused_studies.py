"""Paired infrastructure and service-limit checks, separate from the main suite."""
import argparse
import gzip
import hashlib
import json
import platform
from pathlib import Path
from statistics import mean

import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_experiments import make_instance
from scheduler import catalogue, local_search, metrics, verify_recorded_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'focused'
ARRAYS = ['regions', 'F', 'budget', 'provider_F', 'provider_budget', 'cost', 'initial', 'q']
VARIANTS = ['balanced', 'shared', 'backup']
METHODS = ['geo', 'provider', 'full']


def serial(d):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.items()}


def network_case(base, variant, seed):
    """Common random draws change only transit sharing and usable backups."""
    rng = np.random.default_rng(9000 + seed)
    draws = rng.random(base['n'])
    offsets = rng.integers(1, 6, base['n'])
    probabilities = [1/6]*6 if variant == 'balanced' else [.45,.11,.11,.11,.11,.11]
    transit = np.searchsorted(np.cumsum(probabilities), draws)
    records = []
    for v, old in enumerate(base['records']):
        provider = old['mandatory'][0]
        primary = int(transit[v])
        paths = [dict(dependencies=[provider, f'transit:{primary}'],
                      accepted=True, expires=2, latency_ms=20)]
        if variant == 'backup':
            alternate = (primary + int(offsets[v])) % 6
            paths.append(dict(dependencies=[provider, f'transit:{alternate}'],
                              accepted=True, expires=2, latency_ms=25))
        records.append(dict(mandatory=old['mandatory'], services=[paths]))
    scenarios, F = catalogue(records, latency_ms=100, max_order=1)
    mask = np.array([len(s) == 1 and next(iter(s)).startswith('host:') for s in scenarios])
    budget = np.full(len(F), max(1, base['N']//3))
    result = dict(base, records=records, scenarios=[sorted(s) for s in scenarios],
                  F=F, budget=budget, provider_F=F[mask], provider_budget=budget[mask])
    assert np.array_equal(result['provider_F'], base['provider_F'])
    return result


def extra_metrics(schedule, instance):
    a = np.array(schedule)
    cost = instance['cost']
    selected = cost[a[:, :, None], a[:, None, :]]
    pairs = np.triu_indices(instance['n'], 1)
    e = instance['F'][:, a].sum(axis=2)
    # An independently checked accounting identity for every scenario.
    assert np.array_equal(e.sum(axis=1), instance['F'] @ instance['q'])
    assert float(selected.max()) <= instance['tau'] + 1e-8
    return dict(max_selected_pair_ms=float(selected.max()),
                excluded_pair_fraction=float(np.mean(cost[pairs] > instance['tau'])),
                exposure_lower_bound=int(np.maximum(
                    instance['F'] @ instance['q'] - instance['budget']*instance['T'], 0).sum()))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    data = dict(environment=dict(python=platform.python_version(), numpy=np.__version__,
                                 scipy=scipy.__version__, platform=platform.platform()),
                parameters=dict(n=600, T=64, seeds=[0,1,2], proposers=[4,8],
                                iterations=4000, weights=[10,1,2], solve_method='swap heuristic'),
                bases={}, cases=[], results=[])
    for study in ['infrastructure', 'service']:
        for N in [4,8]:
            for seed in [0,1,2]:
                base = make_instance(n=600, N=N, seed=seed,
                                     allocation_tau=50 if study == 'service' else None)
                name = f'{study}-N{N}-seed{seed}.json.gz'
                payload = gzip.compress(json.dumps(serial(base)).encode(), mtime=0)
                (OUT/name).write_bytes(payload)
                data['bases'][name] = hashlib.sha256(payload).hexdigest()
                variants = VARIANTS if study == 'infrastructure' else [50,100,180]
                for variant in variants:
                    instance = network_case(base, variant, seed) if study == 'infrastructure' else dict(base, tau=variant)
                    for k in ['q','initial','regions','cost']:
                        assert np.array_equal(instance[k], base[k])
                    case_id = len(data['cases'])
                    override = {k: instance[k] for k in ['F','budget','provider_F','provider_budget','records','scenarios']}
                    data['cases'].append(dict(id=case_id, study=study, N=N, seed=seed,
                                              variant=variant, base=name, tau=instance['tau'],
                                              network=serial(override)))
                    for method in METHODS if study == 'infrastructure' else ['full']:
                        schedule, seconds = local_search(instance, method, iterations=4000, seed=seed+100)
                        data['results'].append(dict(case=case_id, method=method, seconds=seconds,
                                                    schedule=schedule.tolist(), **metrics(schedule, instance),
                                                    **extra_metrics(schedule, instance)))
                (OUT/'experiments.json').write_text(json.dumps(data, indent=2)+'\n')
                print(f'{study}: N={N}, seed={seed} complete', flush=True)


def report():
    (ROOT/'figures').mkdir(exist_ok=True)
    data = json.loads((OUT/'experiments.json').read_text())
    bases = {}
    for name, digest in data['bases'].items():
        payload = (OUT/name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
        d = json.loads(gzip.decompress(payload))
        for k in ARRAYS: d[k] = np.array(d[k])
        bases[name] = d
    cases = {}
    for c in data['cases']:
        base = bases[c['base']]
        override = c['network'].copy()
        for k in ['F','budget','provider_F','provider_budget']: override[k] = np.array(override[k])
        instance = dict(base, **override, tau=c['tau'])
        scenarios, F = catalogue(instance['records'], latency_ms=100, max_order=1)
        assert [sorted(s) for s in scenarios] == instance['scenarios']
        assert np.array_equal(F, instance['F'])
        assert np.array_equal(instance['provider_F'], base['provider_F'])
        cases[c['id']] = instance
    for row in data['results']:
        instance = cases[row['case']]
        actual = dict(**metrics(row['schedule'], instance), **extra_metrics(row['schedule'], instance))
        verify_recorded_metrics(row, actual, approximate_floats=True)
    # Geography-only and provider-only choices must be unchanged across transit cases.
    for N in [4,8]:
        for seed in [0,1,2]:
            ids = [c['id'] for c in data['cases'] if c['study']=='infrastructure' and c['N']==N and c['seed']==seed]
            for method in ['geo','provider']:
                schedules = [r['schedule'] for r in data['results'] if r['case'] in ids and r['method']==method]
                assert len(schedules)==3 and all(s==schedules[0] for s in schedules)

    def select(study, N, variant, method):
        ids = {c['id'] for c in data['cases'] if c['study']==study and c['N']==N and c['variant']==variant}
        rows = [r for r in data['results'] if r['case'] in ids and r['method']==method]
        assert len(rows)==3
        return rows

    summary = dict(verified_schedules=len(data['results']), instances=len(cases), infrastructure=[], service=[])
    for study in ['infrastructure','service']:
        for N in [4,8]:
            for variant in VARIANTS if study=='infrastructure' else [50,100,180]:
                for method in METHODS if study=='infrastructure' else ['full']:
                    rows = select(study,N,variant,method)
                    summary[study].append(dict(N=N, variant=variant, method=method,
                        **{k:mean(r[k] for r in rows) for k in ['excess','mean_regions','geo_deficit',
                           'max_selected_pair_ms','excluded_pair_fraction','exposure_lower_bound']},
                        worst_selected_pair_ms=max(r['max_selected_pair_ms'] for r in rows)))

    colors = ['#b27813','#8172b2','#187c92']
    labels = ['Geography only','Provider aware','Path aware']
    fig, axes = plt.subplots(1,2,figsize=(6.1,2.65),sharey=True,layout='constrained')
    for ax,N in zip(axes,[4,8]):
        for j,method in enumerate(METHODS):
            values = [[r['excess'] for r in select('infrastructure',N,v,method)] for v in VARIANTS]
            y = np.array([mean(v) for v in values])
            err = np.array([y-np.array([min(v) for v in values]), np.array([max(v) for v in values])-y])
            ax.bar(np.arange(3)+(j-1)*.24,y,width=.23,color=colors[j],label=labels[j],yerr=err,
                   capsize=2,error_kw=dict(linewidth=.8))
        ax.set_xticks(range(3),['Balanced\ntransit','Shared\ntransit','Shared +\nbackup'])
        ax.set_title(f'{N} concurrent proposers',fontsize=9)
        ax.tick_params(labelsize=8); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Exposure excess per epoch',fontsize=9)
    axes[1].legend(fontsize=7,frameon=False)
    fig.savefig(ROOT/'figures'/'infrastructure-comparison.pdf')
    fig.savefig(ROOT/'figures'/'infrastructure-comparison.png',dpi=180)
    plt.close(fig)
    lines=[r'\begin{table}[t]',r'\centering',r'\renewcommand{\arraystretch}{1.2}',
           r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}rrrrrr@{}}',r'\toprule',
           r'$N$ & \shortstack[r]{Limit\\ms} & \shortstack[r]{Excluded\\pairs, \%} & '
           r'\shortstack[r]{Largest used\\delay, ms} & \shortstack[r]{Mean\\regions} & '
           r'\shortstack[r]{Exposure\\excess} \\',r'\midrule']
    for s in summary['service']:
        if s['N']==8 and s['variant']==50: lines.append(r'\addlinespace[0.5em]')
        lines.append(f'{s["N"]} & {s["variant"]} & {100*s["excluded_pair_fraction"]:.1f} & '
                     f'{s["worst_selected_pair_ms"]:.2f} & {s["mean_regions"]:.2f} & {s["excess"]:.2f}'+r' \\')
    lines += [r'\bottomrule',r'\end{tabular*}',
              r'\caption{Scheduling under service limits.}',r'\label{tab:service-limits}',r'\end{table}']
    (ROOT/'figures'/'service-limits.tex').write_text('\n'.join(lines)+'\n')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if not args.report_only: run()
    report()
