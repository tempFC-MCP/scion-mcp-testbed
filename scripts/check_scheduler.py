"""Compare the implementation with exhaustive schedules and path-cut semantics."""
from itertools import combinations, product
import json
from pathlib import Path
import numpy as np
from scheduler import NoVerifiedSchedule, verify_recorded_metrics, adopt_schedule, catalogue, InvalidSchedule, minimal_cuts, metrics, solve_direct, quotas, local_search, normalize_scion_path, window_misses
from run_experiments import make_instance


def check_adoption():
    base = np.array([[0, 1], [2, 3]])
    better = np.array([[0, 2], [1, 3]])
    ins = dict(n=4, N=2, T=2, nr=1, regions=np.zeros(4, dtype=int), g=1, h=None,
               F=np.array([[1,1,0,0], [1,1,1,1]]), budget=np.array([1,1]),
               initial=base, q=np.ones(4, dtype=int), cost=np.zeros((4,4)), tau=1)
    ins['scenarios'] = [['host:A'], ['observer']]
    before = metrics(base, ins)
    after = metrics(better, ins)
    assert before['exposure_max'] == after['exposure_max'] == 2
    assert before['exposure_max_by_scenario'] == [2, 2]
    assert after['exposure_max_by_scenario'] == [1, 2]
    verify_recorded_metrics(json.loads(json.dumps(after)), after)
    legacy = after.copy()
    legacy.pop('exposure_max_by_scenario')
    verify_recorded_metrics(legacy, after)
    try:
        verify_recorded_metrics(before | {'excess': after['excess']}, after)
    except ValueError as error:
        assert 'exposure_max_by_scenario' in str(error)
    else:
        raise AssertionError('Reporting the same global maximum hides a changed host maximum')
    empty = dict(ins, F=[], scenarios=[], budget=np.array([], dtype=int))
    assert metrics(base, empty)['exposure_max_by_scenario'] == []
    assert metrics(base, empty)['exposure_max'] == 0
    # A new delay snapshot rules out the original. The valid alternatives may tie.
    changed = dict(ins, cost=ins['cost'].copy())
    changed['cost'][0, 1] = 2
    assert adopt_schedule(changed, {'late': None, 'stale': base, 'first': better,
                                    'tied': better[::-1]})[0] == 'first'
    compare = dict(changed, F=changed['F'].copy())
    compare['F'][0] = [1, 0, 1, 0]
    alternative = np.array([[0, 3], [1, 2]])
    assert adopt_schedule(compare, {'higher': better, 'lower': alternative})[0] == 'lower'
    try:
        adopt_schedule(changed, {'late': None, 'stale': base})
    except NoVerifiedSchedule:
        pass
    else:
        raise AssertionError('No verified schedule must be reported when every option fails')
    for candidate in [None, base[::-1], [[0,0],[2,3]], [[0,1],[1,3]], [[0,-1],[2,3]],
                      [[0,4],[2,3]], [[0,1]], [[0,1],[2]], [[0.,1.],[2.,3.]]]:
        assert adopt_schedule(ins, {'candidate':candidate})[0]=='original'
    # Partial gains count, including when two candidates tie for the best score.
    name, _, stats = adopt_schedule(ins, {'first':better, 'second':better[::-1]})
    assert name=='first' and stats['excess']==2 and not stats['compliant']
    assert adopt_schedule(dict(ins, initial=better), {'worse':base})[0]=='original'
    assert adopt_schedule(dict(ins, budget=np.array([1,2])), {'better':better})[2]['compliant']
    eligible = np.ones((2,4), dtype=bool)
    eligible[0,2]=False
    assert adopt_schedule(dict(ins, eligible=eligible), {'better':better})[0]=='original'
    ins['cost'][2,0]=2
    assert adopt_schedule(ins, {'better':better})[0]=='original'
    print('Verified schedule selection with valid and invalid starting schedules.')


def main():
    check_adoption()
    presence = np.zeros((300, 2), dtype=np.int8)
    presence[:, 0] = 1
    assert window_misses(presence, 256) == 300
    # Only the reverse direction of this pair exceeds the limit.
    directed = make_instance(n=2, N=2, seed=0)
    directed['cost'] = np.array([[0, 1], [4, 0]])
    directed['tau'] = 2
    for group in ([0, 1], [1, 0]):
        try:
            metrics(np.tile(group, (directed['T'], 1)), directed)
        except InvalidSchedule:
            pass
        else:
            raise AssertionError('Both directions of every pair must satisfy the limit')
    singleton = make_instance(n=2, N=1, seed=0)
    assert metrics(singleton['initial'], singleton)['mean_pair_ms'] == 0
    one_round = make_instance(n=3, N=2, seed=0)
    original = one_round['initial'][:1]
    one_round.update(T=1, initial=original, q=quotas(original, one_round['n']))
    selected, _ = local_search(one_round)
    assert np.array_equal(selected, original)
    assert adopt_schedule(one_round, {'swap': selected})[0] == 'original'
    exported=dict(record_id='verified-1',as_entries=[{'isd_as':'1-ff00:0:110'},
                  {'isd_as':'1-ff00:0:120'},{'isd_as':'1-ff00:0:110'}],expires=2,latency_ms=10)
    parsed=normalize_scion_path(exported,{'verified-1'})
    assert parsed['dependencies']==['AS:1-ff00:0:110','AS:1-ff00:0:120']
    assert not normalize_scion_path(exported,set())['accepted']
    paths=[frozenset(['a','x']),frozenset(['b','x'])]
    assert minimal_cuts(paths,2)=={frozenset(['x']),frozenset(['a','b'])}
    def p(deps,**kw):
        return dict(dependencies=deps,accepted=True,expires=2,latency_ms=10,**kw)
    records=[dict(mandatory=['host'],services=[[p(['a']),p(['b'])]])]
    scenarios,F=catalogue(records,max_order=2)
    assert not F[scenarios.index(frozenset(['a'])),0]
    assert F[scenarios.index(frozenset(['a','b'])),0]
    assert F[scenarios.index(frozenset(['host'])),0]
    records[0]['services'][0][1]['expires']=0
    scenarios,F=catalogue(records,max_order=2)
    assert F[scenarios.index(frozenset(['a'])),0]
    records[0]['services'][0][0]['accepted']=False
    try: catalogue(records)
    except ValueError: pass
    else: raise AssertionError('Missing path accepted')
    # Exhaustive hitting-set oracle, including overlaps and common endpoints.
    universe=list('abcd'); families=[frozenset(s) for k in [1,2,3] for s in combinations(universe,k)]
    cut_cases=0
    for family in combinations(families,3):
        hits={frozenset(s) for k in [1,2] for s in combinations(universe,k)
              if all(set(s)&p for p in family)}
        expected={s for s in hits if not any(t<s for t in hits)}
        assert minimal_cuts(family,2)==expected
        cut_cases+=1
    cases=0
    rng=np.random.default_rng(42)
    for N in [2,3]:
        for seed in range(8):
            n,T=4,3
            groups=list(combinations(range(n),N))
            initial=np.array([groups[j] for j in rng.integers(len(groups),size=T)])
            F=rng.integers(0,2,size=(3,n),dtype=np.int8)
            cost=np.ones((n,n)); np.fill_diagonal(cost,0)
            # Some instances forbid one edge not present in the initial schedule.
            for u,v in combinations(range(n),2):
                if not any(u in g and v in g for g in initial):
                    cost[u,v]=cost[v,u]=9; break
            ins=dict(n=n,T=T,N=N,regions=np.array([0,0,1,2]),nr=3,
                     F=F,budget=np.ones(3,dtype=int),g=min(3,N),cost=cost,tau=2,
                     initial=initial,q=quotas(initial,n),h=1+seed%3,
                     provider_F=F,provider_budget=np.ones(3,dtype=int))
            best=float('inf'); strict_exists=False
            for sched in product(groups,repeat=T):
                arr=np.array(sched)
                if not np.array_equal(quotas(arr,n),ins['q']): continue
                if any(cost[u,v]>2 for g in arr for u,v in combinations(g,2)):continue
                m=metrics(arr,ins)
                val=10*m['geo_deficit']+m['excess']+2*m['window_misses']
                best=min(best,val); strict_exists |= m['compliant']
            sched,meta=solve_direct(ins,time_limit=10)
            assert meta['status']==0 and abs(meta['objective']-best)<1e-6,(N,seed,meta,best)
            m=metrics(sched,ins)
            assert 10*m['geo_deficit']+m['excess']+2*m['window_misses']==best
            ss,mm=solve_direct(ins,strict=True,time_limit=10)
            assert (mm['status']==0)==strict_exists
            assert mm['status'] in [0,2]
            hs,_=local_search(ins,iterations=50)
            metrics(hs,ins)
            cases+=1
    a=make_instance(50,4,0,backup_fraction=0)
    b=make_instance(50,4,0,backup_fraction=1)
    assert np.array_equal(a['initial'],b['initial'])
    assert np.array_equal(a['cost'],b['cost'])
    low=make_instance(100,4,0,tau=50,allocation_tau=50)
    high=make_instance(100,4,0,tau=180,allocation_tau=50)
    assert np.array_equal(low['q'],high['q'])
    assert np.array_equal(low['initial'],high['initial'])
    # Joint cyclic capacity bound checked independently on region sequences.
    regions=np.array([0,0,1,2]); T=3
    groups=[g for g in combinations(range(4),2) if g!=(2,3)]
    joint_checks=0
    for sched in product(groups,repeat=T):
        sets=[set(regions[list(g)]) for g in sched]
        for h in [1,2,3]:
            if not all(any(r in sets[(t+j)%T] for j in range(h))
                       for r in range(3) for t in range(T)):continue
            for k in [1,2,3]:
                for subset in combinations(range(3),k):
                    capacity=max(len(set(subset)&set(regions[list(g)])) for g in groups)
                    assert k*((T+h-1)//h) <= T*capacity
                    joint_checks+=1
    dest=Path(__file__).resolve().parents[1]/'results'
    dest.mkdir(exist_ok=True)
    result=dict(exhaustive_MILP_instances=cases,strict_and_soft=True,
                minimal_cut_families=cut_cases,joint_regional_bound_checks=joint_checks,
                backup_ablation_same_allocations=True,latency_ablation_same_allocations=True)
    (dest/'implementation-checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':main()
