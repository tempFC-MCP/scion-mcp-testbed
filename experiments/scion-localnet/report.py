"""Verify the pilot measurements and report observed and predicted losses."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.state/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import lab
import measured
import replay
from scheduler import verify_recorded_metrics


LABELS = {'geo':'Geography','provider':'Geography\n+ hosting','full':'Geography\n+ hosting + paths'}
VARIANTS = ['balanced','shared','backup']
TITLES = {'balanced':'Spread transit dependence','shared':'Concentrated transit dependence','backup':'Concentrated + backup'}


def verified_results(root):
    plan=json.loads((root/'plan.json').read_text())
    rows=[]
    for task in plan['tasks']:
        folder=root/'runs'/task['id']
        if not (folder/'summary.json').exists() or not (folder/'checksums.json').exists():
            continue
        lab.verify_recording(folder)
        manifest=json.loads((folder/'manifest.json').read_text())
        if 'link_condition' in task:
            expected=plan['link_conditions'][task['link_condition']]
            if any(manifest['runtime_link_profile'].get(k)!=v for k,v in expected.items()):
                raise ValueError('Replay used the wrong link condition')
            for interface in manifest['runtime_link_profile']['interfaces']:
                actual=[q['options'] for q in interface['actual_qdisc'] if q['kind']=='netem' and q.get('root')]
                if len(actual)!=1:
                    raise ValueError('Missing or ambiguous recorded link queue')
                options=actual[0];delay=options['delay']
                if (abs(delay['delay']-interface['delay_ms']/1000)>2e-6
                    or abs(delay['jitter']-expected['jitter_ms']/1000)>2e-6
                    or options['rate']['rate']!=expected['rate_mbit']*1e6/8):
                    raise ValueError('Recorded kernel queue differs from requested link condition')
        ins=measured.load_instance(root/'cases'/task['case']/'instance.json')
        selected=json.loads((folder/'schedule.json').read_text())
        recomputed_metrics = measured.metrics(selected['schedule'],ins)
        verify_recorded_metrics(selected, recomputed_metrics)
        if lab.file_sha256(root / 'cases' / task['case'] / 'instance.json')!=manifest['instance_sha256']:
            raise ValueError('The replay input has changed')
        senders=[json.loads(s) for s in (folder/'senders.jsonl').read_text().splitlines()]
        receipts=[json.loads(s) for s in (folder/'receipts.jsonl').read_text().splitlines()]
        with tempfile.TemporaryDirectory(dir=root) as temp:
            actual=replay.summarize(Path(temp),ins,selected,manifest,senders,receipts)
        saved=json.loads((folder/'summary.json').read_text())
        if actual!=saved:
            raise ValueError(f'Receiver-log recomputation differs for {task["id"]}')
        completion = {}
        for receipt in receipts:
            if not receipt['verified']:
                continue
            key = (int(receipt['round']), receipt['validator'])
            elapsed = (receipt['complete_ns'] - manifest['start_ns']
                       - key[0] * manifest['interval_ns']) / 1e6
            by_receiver = completion.setdefault(key, {})
            receiver = receipt['receiver']
            by_receiver[receiver] = min(by_receiver.get(receiver, float('inf')), elapsed)
        deadlines = manifest['deadline_ns'] / 1e6
        times = sorted(max(r.values()) for r in completion.values()
                       if len(r) == len(replay.network.RECEIVERS) and max(r.values()) <= deadlines)
        if len(times) != actual['proposals_delivered']:
            raise ValueError('Completion-time counts differ from delivery counts')
        by_id={v['id']:v['region'] for v in ins['validators']}
        regional_arrivals={}
        for (round_id,validator),received in completion.items():
            if len(received)!=len(replay.network.RECEIVERS) or max(received.values())>deadlines:
                continue
            key=(round_id,by_id[validator]);arrival=max(received.values())
            regional_arrivals[key]=min(regional_arrivals.get(key,float('inf')),arrival)
        if len(regional_arrivals)/actual['rounds'] != actual['mean_delivered_regions']:
            raise ValueError('Regional timing disagrees with final regional delivery')
        rows.append(dict(task=task,summary=actual,manifest=manifest,
                         completion_times_ms=times, predicted_catalogue_excess=selected['excess'],
                         regional_arrival_times_ms=sorted(regional_arrivals.values()),
                         retry_attempts=sum(r['kind']=='attempt' and r.get('attempt',0)>0
                                            for r in senders) if manifest['routing']=='retry' else 0))
    return plan,rows


def save_figure(figure, out, stem, preview_dpi=None):
    filename = f'{stem}.pdf'
    figure.savefig(out / filename)
    if preview_dpi is not None:
        figure.savefig(out / f'{stem}.png', dpi=preview_dpi)
    plt.close(figure)
    return filename


def make_figures(out,rows):
    main=[r for r in rows if r['task']['phase']=='comparison' and r['summary']['accepted']]
    methods=['geo','provider','full']
    def select(variant,method,fault):
        return [r['summary'] for r in main if r['summary']['variant']==variant
                and r['summary']['method']==method and bool(r['summary']['failed_core'])==fault]
    if any(not select(v,m,f) for v in VARIANTS for m in methods for f in (False,True)):
        return []
    generated=[]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(10,3.3),sharey=True,layout='constrained')
    for ax,v in zip(axes,VARIANTS):
        samples=[select(v,m,True) for m in methods]
        predicted=[np.mean([r['predicted_excess'] for r in rs]) for rs in samples]
        observed=[np.mean([r['observed_excess'] for r in rs]) for rs in samples]
        x=np.arange(3)
        ax.bar(x-.18,predicted,.35,facecolor='white',edgecolor='#62727b',label='Predicted from paths')
        ax.bar(x+.18,observed,.35,color='#187c92',label='Measured at receivers')
        for j,(p,o) in enumerate(zip(predicted,observed)):
            ax.text(j-.18,p+.3,f'{p:g}',ha='center',fontsize=8)
            ax.text(j+.18,o+.3,f'{o:g}',ha='center',fontsize=8)
        ax.set_xticks(x,[LABELS[m] for m in methods],rotation=20,ha='right')
        ax.set_title(TITLES[v]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Misses beyond the per-round allowance\n(summed over one epoch)')
    axes[-1].legend(frameon=False,fontsize=8,loc='upper right')
    fig.suptitle('Local SCION pilot: predicted and observed failure concentration')
    generated.append(save_figure(fig, out, 'transit-exposure'))

    fig,axes=plt.subplots(1,3,figsize=(10,3.3),sharey=True,layout='constrained')
    for ax,v in zip(axes,VARIANTS):
        groups=[select(v,m,True) for m in methods]
        x=np.arange(3)
        series=[('Scheduled','#aeb8bc',[np.mean([r['mean_scheduled_regions'] for r in rs]) for rs in groups]),
                ('Delivered, no outage','#dca44a',[np.mean([r['mean_delivered_regions'] for r in select(v,m,False)]) for m in methods]),
                ('Delivered, outage','#187c92',[np.mean([r['mean_delivered_regions'] for r in rs]) for rs in groups])]
        for j,(label,color,values) in enumerate(series):ax.bar(x+(j-1)*.25,values,.24,label=label,color=color)
        ax.set_xticks(x,[LABELS[m] for m in methods],rotation=20,ha='right');ax.set_title(TITLES[v]);ax.set_ylim(0,3.5)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Mean represented regions per round')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8,
               loc='outside lower center', ncol=3)
    fig.suptitle('Local SCION pilot: scheduled and delivered regional participation')
    generated.append(save_figure(fig, out, 'regional-delivery'))
    return generated


CONDITION_LABELS={'fixed-1g':'1 Gbit/s, fixed delay',
                  'jitter-1g':'1 Gbit/s, small jitter',
                  'jitter-100m':'100 Mbit/s, small jitter'}


def round_outcome_figure(out,rows):
    """Show every outcome, rather than treating a lower total score as fewer bad rounds."""
    accepted=[r for r in rows if r['summary']['accepted'] and r['summary']['proposers']==4]
    sensitivity=any('link_condition' in r['task'] for r in accepted)
    keys=list(CONDITION_LABELS) if sensitivity else VARIANTS
    methods=['geo','provider','full']
    def select(key,method):
        return [r for r in accepted if r['summary']['method']==method
                and bool(r['summary']['failed_core'])
                and ((r['task'].get('link_condition')==key and r['task']['phase']=='outage-distribution')
                     if sensitivity else (r['summary']['variant']==key and r['task']['phase']=='comparison'))]
    if any(not select(k,m) for k in keys for m in methods):return []
    all_counts=[np.array([np.bincount([f for r in select(key,m)
                                      for f in r['summary']['observed_unsuccessful']],minlength=5)
                          for m in methods]) for key in keys]
    if sensitivity and all(np.array_equal(all_counts[0],c) for c in all_counts[1:]):
        # Do not repeat identical panels merely because three profiles were run.
        fig,ax=plt.subplots(figsize=(7,3.8),layout='constrained')
        for j,(method,color) in enumerate(zip(methods,['#a6afb6','#d5ad66','#187c92'])):
            bars=ax.bar(np.arange(5)+(j-1)*.25,all_counts[0][j],.24,
                        label=LABELS[method].replace('\n',' '),color=color)
            ax.bar_label(bars,padding=2,fontsize=9)
        rounds=int(all_counts[0][0].sum())
        ax.set_xticks(range(5));ax.set_xlabel('Missing proposals out of four')
        ax.set_ylabel(f'Rounds (out of {rounds})');ax.set_ylim(0,max(all_counts[0].ravel())*1.15)
        ax.set_title('Round outcomes during a transit outage')
        ax.text(.5,.96,'Identical counts in all three tested link settings',transform=ax.transAxes,
                ha='center',va='top',fontsize=8,color='#596169')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        handles,labels=ax.get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=8)
        return [save_figure(fig, out, 'round-outcomes', preview_dpi=170)]
    fig,axes=plt.subplots(1,3,figsize=(11,4.1),sharey=True,layout='constrained')
    colors=['#3b8e86','#a1c7bf','#f1ce79','#df8c57','#9e3b40']
    for ax,key,counts in zip(axes,keys,all_counts):
        proportions=100*counts/counts.sum(axis=1,keepdims=True)
        bottom=np.zeros(3)
        for missed in range(5):
            ax.bar(range(3),proportions[:,missed],bottom=bottom,color=colors[missed],
                   label=f'{missed} missing',width=.65)
            for j,p in enumerate(proportions[:,missed]):
                if p>=5:ax.text(j,bottom[j]+p/2,f'{p:.0f}%',ha='center',va='center',fontsize=9)
            bottom+=proportions[:,missed]
        ax.set_xticks(range(3),[LABELS[m] for m in methods],fontsize=8)
        ax.set_title(CONDITION_LABELS[key] if sensitivity else TITLES[key],fontsize=10)
        ax.set_ylim(0,100)
    axes[0].set_ylabel('Percentage of rounds')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False)
    deadline=accepted[0]['manifest']['deadline_ns']/1e6
    fig.suptitle(f'How many of four proposals miss delivery?  |  {deadline:g} ms deadline')
    return [save_figure(fig, out, 'round-outcomes', preview_dpi=170)]


def sensitivity_figure(out,rows):
    accepted=[r for r in rows if r['summary']['accepted'] and 'link_condition' in r['task']]
    if not accepted:return []
    panels=[('No outage','delivery-time','surviving'),
            ('Outage: backup selected before sending','backup-time','surviving'),
            ('Outage: retry after 150 ms','backup-time','retry')]
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),sharey=True,layout='constrained')
    for ax,(title,phase,routing) in zip(axes,panels):
        for condition,color,style in zip(CONDITION_LABELS,['#246a98','#d0892e','#963a61'],['-','--',':']):
            group=[r for r in accepted if r['task']['link_condition']==condition
                   and r['task']['phase']==phase and r['manifest']['routing']==routing]
            if not group:plt.close(fig);return []
            values=sorted(v for r in group for v in r['completion_times_ms'])
            total=sum(r['summary']['proposal_attempts'] for r in group)
            deadline=group[0]['manifest']['deadline_ns']/1e6
            ax.step([0,*values,deadline],[0,*[100*i/total for i in range(1,len(values)+1)],100*len(values)/total],
                    where='post',label=CONDITION_LABELS[condition],color=color,linestyle=style,lw=1.6)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Time from round start (ms)')
        ax.set_xticks([0,250,500,750]);ax.set_xlim(0,750);ax.set_ylim(0,102);ax.grid(alpha=.2)
    axes[0].set_ylabel('Proposals received at both endpoints (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    return [save_figure(fig, out, 'network-sensitivity', preview_dpi=170)]


def regional_timing_figure(out,rows):
    accepted=[r for r in rows if r['summary']['accepted'] and r['task']['phase']=='outage-distribution']
    if not accepted:return []
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    scheduled={r['summary']['mean_scheduled_regions'] for r in accepted}
    if len(scheduled)==1:
        ax.axhline(next(iter(scheduled)),color='#a8adb1',linestyle='-.',lw=1,
                   label='Scheduled before outage')
    grid=np.unique([0,750,*[t for r in accepted for t in r['regional_arrival_times_ms']]])
    for method,color,style in zip(['geo','provider','full'],['#7a8590','#bf8637','#187c92'],[':', '--','-']):
        series=[]
        for condition in CONDITION_LABELS:
            group=[r for r in accepted if r['task']['link_condition']==condition and r['summary']['method']==method]
            if not group:plt.close(fig);return []
            values=sorted(t for r in group for t in r['regional_arrival_times_ms'])
            rounds=sum(r['summary']['rounds'] for r in group)
            series.append(np.searchsorted(values,grid,side='right')/rounds)
        ax.step(grid,series[0],where='post',label=LABELS[method].replace('\n',' '),
                color=color,linestyle=style,lw=1.7)
        ax.fill_between(grid,np.min(series,axis=0),np.max(series,axis=0),step='post',color=color,alpha=.15)
    ax.set_title('Regional participation during a transit outage')
    ax.text(.02,.97,'Lines: fixed-delay control. Shading: range across the three settings.',
            transform=ax.transAxes,ha='left',va='top',fontsize=8,color='#596169')
    ax.set_xlabel('Time from round start (ms)');ax.set_ylabel('Mean represented regions per round')
    ax.set_xlim(0,750);ax.set_xticks([0,250,500,750]);ax.set_ylim(0,3);ax.grid(alpha=.2)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=9)
    return [save_figure(fig, out, 'regional-delivery-over-time', preview_dpi=170)]


def payload_figure(root,out,plan,rows):
    if 'payload_baseline' not in plan:return []
    baseline_root=Path(plan['payload_baseline'])
    _,baseline=verified_results(baseline_root)
    fig,axes=plt.subplots(1,2,figsize=(8,3.6),sharey=True,layout='constrained')
    for ax,condition in zip(axes,['jitter-100m','jitter-1g']):
        current=[r for r in rows if r['task']['link_condition']==condition and r['summary']['accepted']]
        if len(current)!=1:plt.close(fig);return []
        row=current[0]
        matching=[r for r in baseline if r['task'].get('link_condition')==condition
                  and r['task']['phase']=='delivery-time' and r['task']['case']==row['task']['case']
                  and r['summary']['method']==row['summary']['method'] and r['summary']['accepted']]
        if len(matching)!=1:raise ValueError('Missing unique payload baseline')
        previous=matching[0]
        if previous['manifest']['instance_sha256']!=row['manifest']['instance_sha256']:
            raise ValueError('Payload control changed its input')
        if ((baseline_root/'runs'/previous['task']['id']/'schedule.json').read_bytes()
                !=(root/'runs'/row['task']['id']/'schedule.json').read_bytes()):
            raise ValueError('Payload control changed its schedule')
        for r,color in zip([previous,row],['#7a8590','#187c92']):
            values=r['completion_times_ms'];total=r['summary']['proposal_attempts']
            deadline=r['manifest']['deadline_ns']/1e6
            size=r['manifest']['payload_bytes']
            size_label=f'{size/1048576:g} MiB' if size%1048576==0 else f'{size/1024:g} KiB'
            ax.step([0,*values,deadline],[0,*[100*i/total for i in range(1,len(values)+1)],100*len(values)/total],
                    where='post',color=color,lw=1.7,label=f'{size_label} per proposal')
        ax.set_title(CONDITION_LABELS[condition]);ax.set_xlim(0,750);ax.set_ylim(0,102)
        ax.set_xticks([0,250,500,750]);ax.set_xlabel('Time from round start (ms)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Proposals received at both endpoints (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    return [save_figure(fig, out, 'payload-sensitivity', preview_dpi=170)]


def deployment_overview(out,rows):
    """One paired deployment figure, without interpreting the score as bad rounds."""
    methods=['geo','provider','full'];chosen=[]
    for method in methods:
        candidates=[r for r in rows if r['summary']['accepted']
                    and r['summary']['variant']=='shared' and r['summary']['method']==method
                    and r['task']['phase']=='outage-distribution'
                    and r['task'].get('link_condition')=='fixed-1g'
                    and r['summary']['failed_core']=='1-ff00:0:110'
                    and r['summary']['proposers']==4]
        if len(candidates)!=1:return []
        chosen.append(candidates[0])
    for key in ('instance_sha256','deadline_ns','payload_bytes','interval_ns'):
        if len({r['manifest'][key] for r in chosen})!=1:
            raise ValueError('Deployment overview requires matched inputs and workload')
    if len({r['summary']['rounds'] for r in chosen})!=1:
        raise ValueError('Deployment overview requires equal horizons')
    T=chosen[0]['summary']['rounds'];colors=['#9ca7b0','#d6ad62','#187c92']
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.9),gridspec_kw={'width_ratios':[1,1.5]},layout='constrained')
    ax=axes[0]
    values=[r['summary']['mean_delivered_regions'] for r in chosen]
    bars=ax.bar(range(3),values,color=colors,width=.6)
    ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=3,fontsize=9)
    scheduled={r['summary']['mean_scheduled_regions'] for r in chosen}
    if len(scheduled)==1:
        level=next(iter(scheduled));ax.axhline(level,color='#9ca7b0',ls='--',lw=1)
        ax.text(.02,level+.09,f'Scheduled before outage: {level:.3f}',fontsize=8,
                transform=ax.get_yaxis_transform(),color='#596169')
    ax.set_xticks(range(3),['Geography\nonly','Add\nhosting','Add\npaths'])
    ax.set_ylim(0,3);ax.set_ylabel('Mean regions represented per round')
    ax.set_title('Delivered geographic participation',fontsize=10)
    ax=axes[1];positions=np.arange(5)
    for i,(r,color,method) in enumerate(zip(chosen,colors,methods)):
        counts=[r['summary']['observed_unsuccessful'].count(4-delivered) for delivered in positions]
        bars=ax.bar(positions+(i-1)*.25,counts,.24,color=color,label=LABELS[method].replace('\n',' '))
        ax.bar_label(bars,padding=2,fontsize=8)
    highest=max(r['summary']['observed_unsuccessful'].count(i) for r in chosen for i in range(5))
    ax.set_xticks(positions);ax.set_ylim(0,max(5,highest*1.18))
    ax.set_xlabel('Proposals received out of four');ax.set_ylabel(f'Rounds out of {T}')
    ax.set_title('Delivery in individual rounds',fontsize=10)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    return [save_figure(fig, out, 'deployment-overview', preview_dpi=180)]


def timing_figure(out, rows):
    """Use all scheduled proposals as the denominator, including timeouts."""
    accepted = [r for r in rows if r['summary']['accepted']]
    baseline = [r for r in accepted if r['task']['case'].startswith('shared-')
                and r['task']['phase'] == 'comparison' and r['summary']['method'] == 'full'
                and not r['summary']['failed_core']]
    oracle = [r for r in accepted if r['task']['case'].startswith('backup-')
              and r['task']['phase'] == 'comparison' and r['summary']['method'] == 'full'
              and r['summary']['failed_core']]
    tight = [r for r in accepted if r['task']['phase'] == 'deadline']
    retries = [[r for r in accepted if r['task']['phase'] == 'retry'
                and r['manifest']['deadline_ns'] / 1e6 == d] for d in (250, 500)]
    if any(not group for group in [baseline, oracle, tight, *retries]):
        return []
    # Extra controls use the first seed and one repeat. Match their baseline.
    seed = tight[0]['task'].get('seed', baseline[0]['task']['seed'])
    baseline = [r for r in baseline if r['task']['seed'] == seed and r['task']['repeat'] == 0]
    oracle = [r for r in oracle if r['task']['seed'] == seed and r['task']['repeat'] == 0]
    panels = [[('250 ms deadline', tight), ('500 ms deadline', baseline)],
              [('Backup selected beforehand, 500 ms', oracle),
               ('Retry after 150 ms, 250 ms deadline', retries[0]),
               ('Retry after 150 ms, 500 ms deadline', retries[1])]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, layout='constrained')
    for ax, series in zip(axes, panels):
        for label, group in series:
            values = sorted(v for r in group for v in r['completion_times_ms'])
            total = sum(r['summary']['proposal_attempts'] for r in group)
            deadline = group[0]['manifest']['deadline_ns'] / 1e6
            ax.step([0, *values, deadline], [0, *[i/total for i in range(1,len(values)+1)], len(values)/total],
                    where='post', label=label)
            ax.plot(deadline, len(values)/total, 'o', markersize=3, color=ax.lines[-1].get_color())
        ax.set_xlim(0, 510); ax.set_ylim(0, 1.03)
        ax.set_xlabel('Elapsed time from scheduled round start (ms)')
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=7, loc='lower right')
    axes[0].set_ylabel('Fraction of all proposals delivered to both receivers')
    axes[0].set_title('Shared transit, no outage')
    axes[1].set_title('Primary transit outage, backup available')
    fig.suptitle('Local SCION pilot: deadlines and recovery policy')
    return [save_figure(fig, out, 'delivery-timing')]


def resource_table(root, out, rows):
    """Summarize recorded counters without calling them payload wire bytes."""
    fields = ['run', 'mean_host_cpu_busy_percent', 'max_sample_host_cpu_busy_percent',
              'host_cpu_samples', 'data_interface_tx_bytes', 'data_interface_rx_bytes',
              'data_interface_tx_dropped', 'data_interface_rx_dropped']
    records = []
    for row in rows:
        run, m = row['summary']['run'], row['manifest']
        if m.get('orchestrator_recovery',{}).get('resource_measurements_complete') is False:
            records.append(dict({k:None for k in fields},run=run,host_cpu_samples=0))
            continue
        folder = root/'runs'/row['task']['id']
        samples = [json.loads(s) for s in (folder/'resources.jsonl').read_text().splitlines()]
        end = m['start_ns'] + row['summary']['rounds'] * m['interval_ns']
        selected = [s for s in samples if m['start_ns'] <= s['monotonic_ns'] <= end]
        busy, total = [], []
        for first, second in zip(selected, selected[1:]):
            # /proc/stat guest fields are already counted in user/nice.
            a, b = [[int(x) for x in s['cpu'].split()[1:9]] for s in (first, second)]
            delta = [y-x for x,y in zip(a,b)]
            if any(x < 0 for x in delta) or sum(delta) <= 0:
                raise ValueError('Host CPU counters reset or stopped')
            total.append(sum(delta)); busy.append(sum(delta)-delta[3]-delta[4])
        record = dict(run=run, host_cpu_samples=len(total),
                      mean_host_cpu_busy_percent=100*sum(busy)/sum(total) if total else None,
                      max_sample_host_cpu_busy_percent=max((100*b/t for b,t in zip(busy,total)), default=None))
        before = json.loads((folder/'interfaces-before.json').read_text())
        after = json.loads((folder/'interfaces-after.json').read_text())
        counts = {f'data_interface_{direction}_{metric}':0
                  for direction in ('tx','rx') for metric in ('bytes','dropped')}
        for as_id, interfaces in before.items():
            later = {i['ifname']:i for i in after[as_id]}
            for interface in interfaces:
                name = interface['ifname']
                if name in ('lo','eth0'):
                    continue
                if name not in later or later[name]['ifindex'] != interface['ifindex']:
                    raise ValueError('Data interface changed during replay')
                for direction in ('tx','rx'):
                    for metric in ('bytes','dropped'):
                        delta = later[name]['stats64'][direction][metric] - interface['stats64'][direction][metric]
                        if delta < 0:
                            raise ValueError('Interface counters reset during replay')
                        counts[f'data_interface_{direction}_{metric}'] += delta
        records.append(dict(record, **counts))
    with (out/'resources.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    return records


def control_table(root, out, plan, rows):
    if 'comparison_source' not in plan:
        return []
    baseline_root = Path(plan['comparison_source'])
    baseline_plan, baseline_rows = verified_results(baseline_root)
    def key(row, current_plan):
        t = row['task']
        return tuple(t.get(k) for k in ('case','method','fault','routing','deadline_ms')) + (
            t.get('seed',current_plan['seeds'][0]), t.get('repeat',0))
    indexed = {key(r,baseline_plan):r for r in baseline_rows}
    comparisons = []
    lines = ['', '## Paired transport controls', '',
             'Each matched row uses byte-identical scheduler inputs and schedule output. '
             'The earlier batch used a 30-second receiver idle timeout; the control records '
             '300 seconds at both peers. These single replays isolate a configuration change '
             'but do not estimate run-to-run uncertainty.', '',
             '| Policy | N | Routing | Outage | Earlier delivered | Control delivered |',
             '|---|---:|---|---|---:|---:|']
    for row in rows:
        old = indexed.get(key(row,plan))
        if old is None:
            continue
        if old['manifest']['instance_sha256'] != row['manifest']['instance_sha256']:
            raise ValueError('Transport control changed its scheduler input')
        old_schedule = (baseline_root/'runs'/old['task']['id']/'schedule.json').read_bytes()
        new_schedule = (root/'runs'/row['task']['id']/'schedule.json').read_bytes()
        if old_schedule != new_schedule:
            raise ValueError('Transport control changed its schedule')
        s = row['summary']; previous = old['summary']
        lines.append(f'| {s["method"]} | {s["proposers"]} | {s["routing"]} | '
                     f'{"yes" if s["failed_core"] else "no"} | '
                     f'{previous["proposals_delivered"]}/{previous["proposal_attempts"]} | '
                     f'{s["proposals_delivered"]}/{s["proposal_attempts"]} |')
        comparisons.append(dict(baseline_run=previous['run'],control_run=s['run'],
                                identical_input_and_schedule=True,
                                baseline_delivered=previous['proposals_delivered'],
                                control_delivered=s['proposals_delivered']))
    measured.write(out/'paired-controls.json',comparisons)
    return lines


def report(root, diagnostic_plots=False):
    plan,rows=verified_results(root)
    out=root/'report';out.mkdir(exist_ok=True)
    fields=['run','phase','seed','repeat','link_condition','variant','method','proposers','payload_bytes','deadline_ms','routing','failed_core',
            'delivered','opportunities','predicted_catalogue_excess','predicted_tested_fault_excess',
            'observed_tested_fault_excess','scheduled_regions','delivered_regions','max_start_skew_ms',
            'application_bytes_offered','retry_attempts','accepted']
    with (out/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            s,t,m=row['summary'],row['task'],row['manifest']
            writer.writerow(dict(run=s['run'],phase=t['phase'],seed=t.get('seed',plan['seeds'][0]),repeat=t.get('repeat',0),
                link_condition=t.get('link_condition',''),
                variant=s['variant'],method=s['method'],proposers=s['proposers'],payload_bytes=m['payload_bytes'],deadline_ms=m['deadline_ns']/1e6,
                routing=s['routing'],failed_core=s['failed_core'],delivered=s['proposals_delivered'],opportunities=s['proposal_attempts'],
                predicted_catalogue_excess=row['predicted_catalogue_excess'],predicted_tested_fault_excess=s['predicted_excess'],
                observed_tested_fault_excess=s['observed_excess'],scheduled_regions=s['mean_scheduled_regions'],
                delivered_regions=s['mean_delivered_regions'],max_start_skew_ms=s['max_start_skew_ms'],
                application_bytes_offered=s['application_bytes_offered'],retry_attempts=row['retry_attempts'],accepted=s['accepted']))
    figures = []
    if diagnostic_plots:
        figures=(make_figures(out,rows) + timing_figure(out,rows) + round_outcome_figure(out,rows)
                 + sensitivity_figure(out,rows) + regional_timing_figure(out,rows) + payload_figure(root,out,plan,rows)
                 + deployment_overview(out,rows))
    round_records=[]
    for row in rows:
        s=row['summary']; counts=np.bincount(s['observed_unsuccessful'],minlength=s['proposers']+1)
        for missing,count in enumerate(counts):
            round_records.append(dict(run=s['run'],missing=missing,rounds=int(count),
                                      percentage=100*int(count)/s['rounds']))
    with (out/'round-outcomes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['run','missing','rounds','percentage'])
        writer.writeheader();writer.writerows(round_records)
    deadline_records=[];regional_records=[]
    for row in rows:
        s=row['summary']; maximum=row['manifest']['deadline_ns']/1e6
        for deadline in sorted({250,500,750,maximum}):
            if deadline>maximum:continue
            delivered=sum(t<=deadline for t in row['completion_times_ms'])
            deadline_records.append(dict(run=s['run'],deadline_ms=deadline,delivered=delivered,
                                         scheduled=s['proposal_attempts'],percentage=100*delivered/s['proposal_attempts']))
            regional_records.append(dict(run=s['run'],deadline_ms=deadline,
                mean_represented_regions=sum(t<=deadline for t in row['regional_arrival_times_ms'])/s['rounds']))
    with (out/'delivery-by-time.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['run','deadline_ms','delivered','scheduled','percentage'])
        writer.writeheader();writer.writerows(deadline_records)
    with (out/'regional-by-time.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['run','deadline_ms','mean_represented_regions'])
        writer.writeheader();writer.writerows(regional_records)
    resources = resource_table(root, out, rows)
    accepted = sum(row['summary']['accepted'] for row in rows)
    lines = [
        '# SCION pilot results', '',
        f'{len(rows)} of {len(plan["tasks"])} replays verified against receiver logs. '
        f'{accepted} accepted and {len(rows) - accepted} rejected.', '',
        'A delivery counts when both receivers get the complete proposal by the deadline. '
        'In results.csv, predicted_tested_fault_excess and observed_tested_fault_excess sum '
        'the losses beyond one proposer per round. For the route-model comparison, select '
        'accepted comparison replays with a failed transit network.', '',
        '## Data', '',
        '- [Results for each replay](results.csv), including rejected runs and both '
        'the full-catalogue and tested-outage predictions.',
        '- [Proposals missed per round](round-outcomes.csv)',
        '- [Delivery by time](delivery-by-time.csv) and [regional coverage by time](regional-by-time.csv)',
        '- [Host and interface counters](resources.csv)',
        '- [Scheduling results before replay](../offline-scheduling.json)',
        '- [Run plan, seeds and control settings](../plan.json)',
    ]
    if (root / 'interpretation-notes.md').is_file():
        lines.append('- [Notes on this batch](../interpretation-notes.md)')
    cpu = [r['max_sample_host_cpu_busy_percent'] for r in resources
           if r['max_sample_host_cpu_busy_percent'] is not None]
    if cpu:
        lines.extend(['', f'Highest sampled host CPU utilization: {max(cpu):.1f}%.'])
    lines.extend(control_table(root, out, plan, rows))
    if figures:
        lines.extend(['', '## Figures', ''])
        lines.extend(f'- [{name}]({name})' for name in figures)
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    measured.write(out/'verification.json',dict(verified_runs=len(rows),planned_runs=len(plan['tasks']),
                                               figures=figures,receiver_metrics_recomputed=True))
    print(f'Verified {len(rows)} replays; report: {out/"RESULTS.md"}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    parser.add_argument('--diagnostic-plots', action='store_true', help='Draw additional pilot diagnostics')
    args = parser.parse_args()
    report(args.batch, args.diagnostic_plots)
