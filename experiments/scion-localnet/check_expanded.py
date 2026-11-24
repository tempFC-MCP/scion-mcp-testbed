"""Independent checks for fixed-hub scheduling and the expanded deployment."""
import argparse
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import measured
import check_coordination
from scheduler import metrics, local_search, solve_direct, quotas, verify_recorded_metrics


def membership_checks():
    x = dict(n=4, N=2, T=4, nr=2, g=1, regions=np.array([0,0,1,1]),
             q=np.full(4,2), initial=np.array([[0,1],[0,1],[2,3],[2,3]]),
             cost=np.zeros((4,4)), tau=1, F=np.array([[1,1,0,0]], dtype=bool),
             budget=np.array([1]), h=None,
             eligible=np.array([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]], dtype=bool))
    x['provider_F'], x['provider_budget'] = x['F'], x['budget']
    groups=list(combinations(range(4),2))
    feasible=[np.array(a) for a in product(groups, repeat=4)
              if all(x['eligible'][t,list(g)].all() for t,g in enumerate(a))
              and np.array_equal(quotas(a,4), x['q'])]
    optimum=min(metrics(a,x)['excess'] for a in feasible)
    a, result=solve_direct(x,time_limit=10)
    assert result['status']==0 and metrics(a,x)['excess']==optimum==2
    a,_=local_search(x,iterations=100,seed=4)
    assert metrics(a,x)['excess']==2
    impossible=dict(x,q=np.array([3,1,2,2]))
    a,result=solve_direct(impossible,time_limit=10)
    assert a is None and result['status']==2
    print('Fixed-hub MILP matches exhaustive optimum; swaps preserve membership; impossible quotas rejected.')


def schedules(out):
    rows=json.loads((out/'sweep.json').read_text())
    checked=0
    for case in sorted((out/'cases').iterdir()):
        x=measured.load_instance(case/'instance.json')
        assert np.array_equal(x['q'], np.full(36,2))
        for row in json.loads((case/'schedules.json').read_text()):
            a=np.array(row['schedule']); actual=metrics(a,x)
            verify_recorded_metrics(row, actual)
            if x['tau']==20:
                assert all(set(x['regions'][g])=={int(x['regions'][old[0]])}
                           for g,old in zip(a,x['initial']))
            target=next(r for r in rows if r['case']==case.name and r['method']==row['method'])
            e=x['F'][:,a].sum(axis=2)
            assert target['all_limits_rounds']==int((e<=x['budget'][:,None]).all(axis=0).sum())
            assert np.array_equal(e.sum(axis=1),x['F'].astype(int)@x['q'])
            assert row.get('solver',{}).get('optimal', True), 'Unproven optimum: report as incumbent'
            checked+=1
    assert checked==len(rows)
    print(f'Independently verified {checked} schedules, allocations, hub membership, and exposure identities.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path)
    p.add_argument('--complete',action='store_true')
    args=p.parse_args()
    membership_checks();check_coordination.small_cases()
    if args.out:
        schedules(args.out)
        if args.complete:
            check_coordination.peer_records(args.out)
            check_coordination.payload_records(args.out)
