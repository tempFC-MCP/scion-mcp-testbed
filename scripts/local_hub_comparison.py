"""Paired local-hub experiment; writes review figures without changing the paper.

Run from the repository root with the experiment Python environment.
Raw inputs/schedules stay under ignored results/. Local and distributed cases
share topologies, but not allocations: only the within-local comparison holds
both per-validator quotas and the hub sequence fixed.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_experiments import make_instance
from scheduler import local_search, metrics, quotas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'local-hub-comparison'


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    return value


def local_instance(n, N, seed):
    ins = make_instance(n=n, N=N, seed=seed, g=1)
    rng = np.random.default_rng(10000 + seed)
    weights = rng.lognormal(0, 1, n)
    weights /= weights.sum()
    counts = np.bincount(ins['regions'], minlength=ins['nr'])
    hubs = np.flatnonzero(counts >= N)
    # A hub with fewer than N validators cannot host N distinct proposers.
    # Report excluded hubs and weight explicitly; do not pad or relabel them.
    hub_weights = np.array([weights[ins['regions'] == h].sum() for h in hubs])
    excluded_weight = float(1 - hub_weights.sum())
    hub_weights /= hub_weights.sum()
    sequence = rng.choice(hubs, size=ins['T'], p=hub_weights)
    initial = []
    for hub in sequence:
        candidates = np.flatnonzero(ins['regions'] == hub)
        p = weights[candidates] / weights[candidates].sum()
        initial.append(rng.choice(candidates, size=N, replace=False, p=p))
    ins['initial'] = np.asarray(initial)
    ins['q'] = quotas(ins['initial'], n)
    ins['eligible'] = sequence[:, None] == ins['regions'][None, :]
    ins['hub_sequence'] = sequence
    ins['sampling_weights'] = weights
    meta = dict(hub_validator_counts=counts.tolist(), eligible_hubs=hubs.tolist(),
                excluded_weight=excluded_weight,
                hub_round_counts=np.bincount(sequence, minlength=ins['nr']).tolist())
    return ins, meta


def independent_checks(schedule, ins):
    """Recompute losses from path records, not the catalogue matrix alone."""
    assert all(len(set(group)) == ins['N'] for group in schedule)
    assert np.array_equal(np.bincount(schedule.ravel(), minlength=ins['n']), ins['q'])
    assert all(all(ins['regions'][v] == ins['hub_sequence'][t] for v in group)
               for t, group in enumerate(schedule))
    failures = []
    for scenario in ins['scenarios']:
        failed = set(scenario)
        row = []
        for record in ins['records']:
            mandatory = bool(failed.intersection(record['mandatory']))
            service_lost = any(all(failed.intersection(p['dependencies'])
                                  for p in service if p['accepted']
                                  and p['expires'] >= 1 and p['latency_ms'] <= 100)
                               for service in record['services'])
            row.append(mandatory or service_lost)
        failures.append(row)
    F = np.array(failures, dtype=int)
    assert np.array_equal(F, ins['F'])
    exposure = np.array([[sum(int(F[s, v]) for v in group) for group in schedule]
                         for s in range(len(F))])
    exceeded = exposure > ins['budget'][:, None]
    m = metrics(schedule, ins)
    assert m['mean_regions'] == 1 and m['geo_deficit'] == 0
    assert m['excess'] == int(np.maximum(exposure-ins['budget'][:, None], 0).sum())
    assert np.array_equal(exposure.sum(axis=1), F @ ins['q'])
    # Necessary lower bounds within each fixed hub, summed over scenarios/hubs.
    lower = 0
    worst_lower = 0
    for hub in range(ins['nr']):
        slots = int(np.sum(ins['hub_sequence'] == hub))
        loads = F[:, ins['regions'] == hub] @ ins['q'][ins['regions'] == hub]
        if slots:
            lower += int(np.maximum(loads-ins['budget']*slots, 0).sum())
            worst_lower = max(worst_lower, int(np.max((loads+slots-1)//slots)))
        else:
            assert not loads.any()
    assert m['excess'] >= lower and m['exposure_max'] >= worst_lower
    return dict(**m, excess_lower_bound=lower, worst_lower_bound=worst_lower,
                scenario_rounds=int(exposure.size),
                scenario_rounds_above_allowance=int(exceeded.sum()),
                rounds_with_any_exceedance=int(exceeded.any(axis=0).sum()),
                full_group_loss_pairs=int((exposure == ins['N']).sum()),
                loss_histogram=np.bincount(exposure.ravel(), minlength=ins['N']+1).tolist(),
                total_exposure=int(exposure.sum()))


FIGURE_DATA = ROOT / 'results' / 'figure2-exposure.csv'


def figure_rows(local, geographic):
    """One row per seed; all three geographic policies use the same quotas."""
    result = []
    for regime, rows, baseline, method in [('Geographic', geographic, 'geo', 'full'),
                                          ('Local', local, 'local_base', 'local_full')]:
        cases = {}
        for row in rows:
            key = row['n'], row['N'], row['seed']
            cases.setdefault(key, {})[row['method']] = row
        for (n, N, seed), methods in sorted(cases.items()):
            base = methods[baseline]
            selected = methods[method]
            initial = methods['random'] if regime == 'Geographic' else base
            result.append(dict(
                regime=regime, validators=n, proposers=N, seed=seed, rounds=base['T'],
                original_excess=initial['excess'], baseline_excess=base['excess'],
                infrastructure_excess=selected['excess'], original_mean_regions=initial['mean_regions'],
                baseline_mean_regions=base['mean_regions'], infrastructure_mean_regions=selected['mean_regions'],
                infrastructure_missing_regions=selected['geo_deficit'],
                infrastructure_missed_windows=selected['window_misses'],
                infrastructure_meets_all_targets=int(selected['compliant']),
                baseline_rounds_over_cap=base.get('rounds_with_any_exceedance', ''),
                infrastructure_rounds_over_cap=selected.get('rounds_with_any_exceedance', ''),
            ))
    return result


def draw_exposure(records, destination):
    destination.parent.mkdir(exist_ok=True, parents=True)
    panels = [('Geographic', 'Geographic groups'), ('Local', 'Local groups')]
    widths = sorted({int(r['proposers']) for r in records})
    sizes = sorted({int(r['validators']) for r in records})
    series = [('baseline_excess', 'Baseline', '#4d5158', 'o', '--'),
              ('infrastructure_excess', 'Hosting and transit considered', '#087566', 'D', '-')]
    fig, axes = plt.subplots(len(widths), len(panels), figsize=(6.3, 2.75),
                             sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=.105, right=.885, top=.78, bottom=.25, hspace=.18, wspace=.13)
    peak = max(int(r[field]) for r in records for field, *_ in series)
    x = np.arange(len(sizes))
    for i, N in enumerate(widths):
        for j, (regime, heading) in enumerate(panels):
            ax = axes[i, j]
            rows = [r for r in records if r['regime'] == regime and int(r['proposers']) == N]
            for field, label, color, marker, style in series:
                samples = [[int(r[field]) for r in rows if int(r['validators']) == n] for n in sizes]
                means = np.array([np.mean(v) for v in samples])
                lower = means - [min(v) for v in samples]
                upper = [max(v) for v in samples] - means
                ax.errorbar(x, means, yerr=[lower, upper], color=color, linestyle=style,
                            marker=marker, markerfacecolor='white' if style == '--' else color,
                            markersize=3.8, linewidth=1.45, capsize=2.3, elinewidth=.8, label=label)
            if i == 0:
                ax.text(.5, 1.43, heading, transform=ax.transAxes, ha='center',
                        fontsize=10, weight='bold')
                subtitle = 'Baseline seeks regional coverage' if j == 0 else 'Baseline draws by weight within each hub'
                ax.text(.5, 1.15, subtitle, transform=ax.transAxes, ha='center', fontsize=7.4)
            if j == 1:
                ax.text(1.025, .50, f'{N}\nproposers', transform=ax.transAxes,
                        ha='left', va='center', fontsize=7.4)
            ax.set_xlim(-.18, len(sizes)-.82)
            ax.set_ylim(0, peak * 1.08)
            ax.set_xticks(x, sizes)
            ax.tick_params(axis='both', labelsize=8, length=0, pad=3)
            ax.set_yticks([0, 50, 100])
            ax.grid(axis='y', color='#dce1e2', linewidth=.55)
            ax.spines[['left', 'right', 'top']].set_visible(False)
            ax.spines['bottom'].set_color('#b9c1c2')
    fig.text(.015, .53, 'Exposures above the cap', rotation=90, ha='left', va='center', fontsize=9)
    fig.text(.51, .11, 'Validators', ha='center', fontsize=8.5)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=8.5,
               handlelength=2.3, columnspacing=2.5, bbox_to_anchor=(.51, -.025))
    for ext in ['pdf', 'png']:
        fig.savefig(destination.with_suffix('.' + ext), dpi=200)
    plt.close(fig)


def plot(rows, original, destination=None):
    records = figure_rows(rows, original)
    if destination is None:
        destination = OUT / 'comparison-preview'
    else:
        with FIGURE_DATA.open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=records[0], lineterminator='\n')
            writer.writeheader()
            writer.writerows(records)
    draw_exposure(records, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure-only', action='store_true', help='Draw Figure 2 from its saved per-seed scores.')
    if parser.parse_args().figure_only:
        if not FIGURE_DATA.exists():
            parser.error('Run make simulate first to generate your Figure 2 scores.')
        with FIGURE_DATA.open(newline='') as source:
            draw_exposure(list(csv.DictReader(source)), ROOT / 'figures' / 'scheduler-results')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'results' / 'experiments.json'
    original = [r for r in json.loads(source.read_text())['results']
                if r['source'] == 'synthetic' and r['backup_fraction'] == .5
                and r['h'] is None and r['tau'] == 180 and 'g' not in r
                and 'allocation_tau' not in r]
    rows, inputs = [], []
    for n in [50, 100, 300, 600]:
        for N in [4, 8]:
            for seed in range(3):
                ins, meta = local_instance(n, N, seed)
                reference = next(r for r in original if r['n'] == n
                                 and r['N'] == N and r['seed'] == seed)
                saved = json.loads((ROOT / 'results' /
                                    f'instance-{reference["instance"]:02d}.json').read_text())
                for key in ['regions', 'F', 'cost', 'budget']:
                    assert np.array_equal(ins[key], saved[key]), key
                path = OUT / f'input-{n}-{N}-{seed}.json'
                path.write_text(json.dumps(serializable(ins))+'\n')
                inputs.append(dict(n=n, N=N, seed=seed, **meta,
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                base = independent_checks(ins['initial'], ins)
                selected, seconds = local_search(ins, mode='full', iterations=4000, seed=100+seed)
                full = independent_checks(selected, ins)
                assert full['excess'] <= base['excess']
                assert base['total_exposure'] == full['total_exposure']
                for method, schedule, values, elapsed in [
                    ('local_base', ins['initial'], base, 0.),
                    ('local_full', selected, full, seconds),
                ]:
                    rows.append(dict(n=n, N=N, T=64, seed=seed, method=method,
                                     seconds=elapsed, **values, schedule=schedule.tolist()))
                print(f'n={n} N={N} seed={seed}: excess {base["excess"]}->{full["excess"]}; '
                      f'worst {base["exposure_max"]}->{full["exposure_max"]}', flush=True)
    source_paths = [Path(__file__), ROOT/'scripts'/'scheduler.py',
                    ROOT/'scripts'/'run_experiments.py', source]
    output = dict(environment=dict(python=platform.python_version(), numpy=np.__version__),
                  source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in source_paths},
                  interpretation=dict(
                      comparison='Within-local pairs preserve quotas and hub sequence; local and distributed allocations differ.',
                      metric='Sum of affected proposers above allowance across scenario-round pairs; not a failure probability.',
                      excluded_hubs='Hubs with fewer than N validators cannot supply a local group and receive no rounds.',
                      solver='4000 attempted swaps; no optimality claim.',
                      uncertainty='Means and min/max across three seeds, not confidence intervals.'),
                  parameters=dict(iterations=4000, seeds=[0, 1, 2], T=64,
                                  allocation_seed_offset=10000, search_seed_offset=100,
                                  hub_selection='weight-proportional among hubs with at least N validators',
                                  baseline='weighted sampling without replacement inside each assigned hub',
                                  tau_ms=180, geo_target=1, allowance='max(1, N//3)'),
                  inputs=inputs, results=rows)
    (OUT / 'results.json').write_text(json.dumps(output, indent=2)+'\n')
    plot(rows, original)
    summary = []
    keys = ['excess', 'excess_lower_bound', 'exposure_max', 'scenario_rounds_above_allowance',
            'rounds_with_any_exceedance', 'full_group_loss_pairs', 'seconds']
    for n in [50, 100, 300, 600]:
        for N in [4, 8]:
            result = dict(n=n, N=N)
            for method in ['local_base', 'local_full']:
                cases = [r for r in rows if r['n'] == n and r['N'] == N and r['method'] == method]
                result[method] = {key: float(np.mean([r[key] for r in cases])) for key in keys}
            summary.append(result)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Saved 24 paired cases and review figure to', OUT)


if __name__ == '__main__':
    main()
