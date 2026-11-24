"""Follow-up: enforce the load-derived transit allowance in fixed local hubs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import lab
import measured
from scheduler import metrics, solve_direct


def prepare(source,out):
    out.mkdir(parents=True,exist_ok=False);(out/'cases').mkdir()
    tasks=[];rows=[]
    for seed in range(3):
        name=f'p3-shared-s{seed}-rtt20-strict'
        original=source/'cases'/f'p3-shared-s{seed}-rtt20'
        x=measured.load_instance(original/'instance.json')
        core=x['scenarios'].index(['AS:1-ff00:0:110'])
        allocated=int(x['F'][core].astype(int)@x['q'])
        allowance=int(np.ceil(allocated/x['T']))
        x['budget'][core]=allowance
        x['eligible']=np.array([x['regions']==x['regions'][g[0]] for g in x['initial']])
        x['measurement']['followup']='Strict budget from fixed transit load; same quotas, hub order, and measured paths.'
        x['measurement']['source_case_sha256']=lab.file_sha256(original / 'instance.json')
        a,meta=solve_direct(x,strict=True,time_limit=30)
        if a is None:
            raise RuntimeError(f'{name}: no strict schedule for the measured delay and hub constraints. '
                               f'Solver status: {meta["message"]}')
        stats=metrics(a,x);assert stats['compliant']
        # Feasibility gives an optimum here: all shortfalls have lower bound zero.
        meta['optimal']=True
        selected=dict(method='strict',schedule=a.tolist(),solver=meta,**stats)
        case=out/'cases'/name;case.mkdir()
        measured.write(case/'instance.json',x);measured.write(case/'schedules.json',[selected])
        exposure=x['F'][:,a].sum(axis=2)
        hosting=x['provider_F'][:,a].sum(axis=2)
        assert exposure[core].sum() == allocated and np.all(exposure[core] <= allowance)
        compliant_rounds=(exposure<=x['budget'][:,None]).all(axis=0)
        rows.append(dict(case=name,seed=seed,providers=3,variant='shared',rtt_limit=20,
                         method='strict',mean_regions=stats['mean_regions'],excess=stats['excess'],
                         hosting_worst=int(hosting.max()),single_provider_rounds=int((hosting.max(axis=0)==x['N']).sum()),
                         optimal=True,all_limits_rounds=int(compliant_rounds.sum())))
        tasks.append(dict(id=f'strict-local-s{seed}-outage',case=name,method='strict',fault='1-ff00:0:110',routing='surviving',deadline_ms=750))
    names=['strict_local.py','network.py','lab.py','measured.py','replay.py','batch.py','cmd/mcp-traffic/main.go','../../scripts/scheduler.py','resume_timing.py']
    measured.write(out/'plan.json',dict(tasks=tasks,interval_ms=1000,warmup_payloads=1,payload_bytes=262144,
        source_plan_sha256=lab.file_sha256(source / 'plan.json'),
        followup_reason='Aggregate-optimal schedules can retain fully exposed rounds. Test a feasible hard allowance of two, derived from fixed load, without changing allocations or routes.',
        code_sha256=lab.code_hashes(names)))
    measured.write(out/'sweep.json',rows)
    print('Three strict schedules meet the load-derived transit bound and every hosting budget.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--wait-for-source',action='store_true')
    a=p.parse_args();lab.select_profile('expanded')
    if a.wait_for_source:
        until=time.monotonic()+3600
        while not (a.source/'progress.json').exists() or json.loads((a.source/'progress.json').read_text())['status']!='complete':
            if time.monotonic()>until:raise TimeoutError('Source batch incomplete; no follow-up traffic launched')
            time.sleep(2)
    prepare(a.source,a.out)
    from resume_timing import run
    run(a.out)
