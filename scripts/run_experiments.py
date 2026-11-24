"""Reproducible scheduler experiment suite. All latency and quota inputs are modeled."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import numpy as np
import scipy
from scheduler import catalogue, quotas, metrics, local_search, solve_direct

ROOT = Path(__file__).resolve().parents[1]


def make_instance(n=50, N=4, seed=0, backup_fraction=.5, tau=180, h=None, public=None,
                  allocation_tau=None, g=None):
    rng = np.random.default_rng(seed)
    T = 64
    centers = np.array([[-74,41],[-122,38],[8,50],[104,1],[139,36],[151,-34]])
    regions = rng.choice(6,n,p=[.15,.10,.40,.15,.10,.10])
    coordinates = centers[regions]+rng.normal(0,2,(n,2))
    providers = rng.integers(0,12,n)
    records = []
    if public:
        n = len(public)
        coordinates = np.array([p['coordinates'] for p in public])
        regions = np.argmin(((coordinates[:,None,:]-centers[None,:,:])**2).sum(axis=2),axis=1)
        providers = np.array([p['asn'] for p in public])
    for v in range(n):
        provider = 'host:'+str(providers[v])
        if public:
            # One observed route. Collector alternatives are NOT treated as usable backups.
            paths = [{'dependencies':['AS:'+str(x) for x in public[v]['paths'][0]],
                      'accepted':True,'expires':2,'latency_ms':1}]
        else:
            transit = int(rng.choice(6,p=[.45,.11,.11,.11,.11,.11]))
            paths = [{'dependencies':[provider,'transit:'+str(transit)],
                      'accepted':True,'expires':2,'latency_ms':20}]
            backup_draw = rng.random()
            other = (transit+int(rng.integers(1,6)))%6
            if backup_draw < backup_fraction:
                paths.append({'dependencies':[provider,'transit:'+str(other)],
                              'accepted':True,'expires':2,'latency_ms':25})
        records.append({'mandatory':[provider],'services':[paths]})
    scenarios,F = catalogue(records,latency_ms=100,max_order=1)
    provider_mask = np.array([len(s)==1 and next(iter(s)).startswith('host:') for s in scenarios])
    # Relabel only the represented regions; no invented region is required.
    _,regions = np.unique(regions,return_inverse=True)
    nr = int(regions.max())+1
    lon,lat = np.radians(coordinates[:,0]),np.radians(coordinates[:,1])
    delta_lat = lat[:,None]-lat[None,:]; delta_lon = lon[:,None]-lon[None,:]
    arc = np.sin(delta_lat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(delta_lon/2)**2
    km = 2*6371*np.arcsin(np.sqrt(np.clip(arc,0,1)))
    cost = 2+1.5*km/200  # Synthetic propagation estimate, NOT RTT observations.
    np.fill_diagonal(cost,0)
    stake = rng.lognormal(0,1,n); stake /= stake.sum()
    initial = []
    for t in range(T):
        chosen = []
        for _ in range(N):
            candidates = [v for v in range(n) if v not in chosen
                          and all(cost[v,u] <= (tau if allocation_tau is None else allocation_tau) for u in chosen)]
            if not candidates: raise ValueError('Unable to seed hard-QoS grouping')
            p = stake[candidates]; p /= p.sum()
            chosen.append(int(rng.choice(candidates,p=p)))
        initial.append(chosen)
    initial = np.array(initial)
    budget = np.full(len(F),max(1,N//3))
    return dict(n=n,N=N,T=T,regions=regions,nr=nr,F=F,budget=budget,
                provider_F=F[provider_mask],provider_budget=budget[provider_mask],
                g=min(N,nr) if g is None else g,tau=tau,cost=cost,initial=initial,q=quotas(initial,n),h=h,
                scenarios=[sorted(s) for s in scenarios],records=records)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--ripe-sample', type=Path,
                        help='Run the three public-routing cases using this independently acquired probe sample.')
    parser.add_argument('--resume',action='store_true',
                        help='Keep complete saved instances and run only missing cases')
    args=parser.parse_args()
    out=ROOT/'results'
    if args.ripe_sample:
        out=out/'ripe'
    out.mkdir(parents=True, exist_ok=True)
    specs=[]
    for n in ([50] if args.quick else [50,100,300]):
        for N in ([4] if args.quick else [4,8]):
            for seed in ([0] if args.quick else [0,1,2]):
                specs.append(dict(n=n,N=N,seed=seed))
    # Paired sensitivity cases; the random path draw count is independent of backup_fraction.
    for backup in [0.,1.]:
        for seed in ([0] if args.quick else [0,1,2]):
            specs.append(dict(n=100,N=4,seed=seed,backup_fraction=backup))
    for h in [1,2,4]:
        specs.append(dict(n=100,N=4,seed=0,h=h))
    for tau in [50,100,180]: specs.append(dict(n=100,N=4,seed=0,tau=tau,allocation_tau=50))
    specs.extend([dict(n=100,N=4,seed=0,g=2),dict(n=300,N=8,seed=0,g=4)])
    if not args.quick:
        for N in [4,8]:
            for seed in [0,1,2]: specs.append(dict(n=600,N=N,seed=seed))
    if args.ripe_sample:
        public=json.loads(args.ripe_sample.read_text())
        if len(public)<16:
            parser.error('The supplied RIPE sample has fewer than 16 eligible endpoints.')
        specs=[dict(n=len(public),N=4,seed=seed,public=public) for seed in [0,1,2]]
    result_path=out/'experiments.json'
    saved=json.loads(result_path.read_text()) if args.resume and result_path.exists() else None
    rows=[] if saved is None else saved['results']
    environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                     platform=platform.platform())
    if saved is not None:
        for k,v in environment.items():
            if saved['environment'].get(k)!=v:
                raise ValueError(f'Resume environment differs: {k}')
        environment=saved['environment']
        if any(r['instance']>=len(specs) for r in rows):
            raise ValueError('Saved suite differs from requested suite')
    for i,spec in enumerate(specs):
        key={k:v for k,v in spec.items() if k!='public'}
        key.update(source='RIPE-informed' if 'public' in spec else 'synthetic',
                   backup_fraction=spec.get('backup_fraction',.5),tau=spec.get('tau',180),
                   h=spec.get('h'),T=64)
        if args.ripe_sample and 'public' in spec:
            key['public_input_sha256']=hashlib.sha256(args.ripe_sample.read_bytes()).hexdigest()
        existing=[r for r in rows if r['instance']==i]
        if existing:
            if len(existing)!=6 or {r['method'] for r in existing}!={'random','latency','geo','provider','full','MILP'}:
                raise ValueError(f'Incomplete saved instance {i}')
            if any({k:v for k,v in r.items() if k in key}!=key for r in existing):
                raise ValueError(f'Saved parameters differ for instance {i}')
            if not (out/f'instance-{i:02d}.json').exists():
                raise ValueError(f'Missing saved input for instance {i}')
            continue
        instance=make_instance(**spec)
        snapshot = {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in instance.items()}
        (out/f'instance-{i:02d}.json').write_text(json.dumps(snapshot)+'\n')
        for mode in ['random','latency','geo','provider','full']:
            if mode=='random': schedule,seconds=instance['initial'],0.
            else: schedule,seconds=local_search(instance,mode,iterations=1500 if args.quick else 4000,seed=spec['seed']+100)
            row=dict(instance=i,**key,method=mode,seconds=seconds,**metrics(schedule,instance),
                     schedule=schedule.tolist())
            rows.append(row)
        # Solver timeouts are reported, never interpreted as infeasibility.
        schedule,meta=solve_direct(instance,time_limit=2 if args.quick else 5)
        row=dict(instance=i,**key,method='MILP',**meta)
        if schedule is not None:
            row.update(metrics(schedule,instance),schedule=schedule.tolist())
        rows.append(row)
        result_path.write_text(json.dumps(dict(environment=environment, results=rows),indent=2)+'\n')
        print(f'{i+1}/{len(specs)} n={instance["n"]} N={instance["N"]} {key["source"]} MILP={meta["status"]}',flush=True)
    print('Saved',len(rows),'method runs',flush=True)


if __name__=='__main__': main()
