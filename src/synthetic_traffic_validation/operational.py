"""Cycle-boundary reuse/repair/recomputation of a 12-slot plan template.

Each decision fixes one entire next-cycle plan. Tests do not assert validity
for within-cycle observation-dependent replanning or a deterministic L bound.
"""
import json
from pathlib import Path
import time

import numpy as np

from .allocation import report_channel,service_margin,solve_allocation,matching_plan,hybrid
from .experiments import metrics,update_error
from .resources import Meter,write_csv,write_json

POLICIES=['every_assessment','periodic','margin_trigger','margin_and_gap','adaptive_interval','certified_matching']


def estimate_L(f,indices):
    changes=[]
    for ja,jb in zip(indices[:-1],indices[1:]):
        ma,sa=report_channel(cycle=int(ja)); mb,sb=report_channel(cycle=int(jb))
        # A fixed plan is compared at both epochs. Reoptimized plan margins
        # cannot estimate deterioration of the old plan.
        b=matching_plan(ma,f['farima'][ja]+f['reserve'])['bandwidth']
        a=service_margin(b,ma,sa,f['farima'][ja]+f['reserve'])
        c=service_margin(b,mb,sb,f['farima'][jb]+f['reserve'])
        changes.append(np.maximum(a-c,0)/((f['origins'][jb]-f['origins'][ja])*.002))
    v=np.asarray(changes)
    return np.maximum(np.quantile(v,.99,axis=0)*1.2,1e-8),v


def replay(f,indices,policy,L,*,beta=.7,rho=.03,threshold=.25,lam=1.,gamma=.9,kappa=1.,
           initial_error=None,draws=128,record=True,cache=None):
    rows=[]; old=None; next_epoch=0; err=np.zeros(8) if initial_error is None else initial_error.copy()
    last_pattern=None; past_raw=None; cycle_s=.024
    for pos,j in enumerate(indices):
        mu,sd=report_channel(cycle=int(j)); pred=f['farima'][j]; reserve=f['reserve']; target=pred+reserve
        start=time.perf_counter(); cpu=time.process_time()
        info=dict(solver_calls=0,solver_s=0.,full_calls=0,repair_calls=0,certificate_s=0.,matching_s=0.,score_s=0.,solve_wall_s=0.)
        gap=np.nan; trigger=old is None; action='reuse'; assess=True
        if old is not None:
            margin=service_margin(old,mu,sd,target)
            assess=policy!='adaptive_interval' or pos>=next_epoch
            if policy=='every_assessment': trigger=True
            elif policy=='periodic': trigger=(pos%4==0)
            elif policy in ['margin_trigger','margin_and_gap','adaptive_interval'] and assess:
                tick=time.perf_counter(); trigger=margin.min() < -1e-6
                info['certificate_s']=time.perf_counter()-tick
                if not trigger and policy in ['margin_and_gap','adaptive_interval']:
                    relax=solve_allocation(mu,sd,target,relaxed=True)
                    for field in ['solver_calls','solver_s','solve_wall_s']: info[field]+=relax[field]
                    upper=relax['upper_bound_mbit']
                    if upper is not None:
                        gap=max(0.,(upper-float((old*mu).sum()))/max(upper,1e-9)); trigger=gap>rho
                    else: trigger=True
        if policy=='certified_matching':
            sol=hybrid(mu,sd,pred,reserve,np.exp(-kappa*err),threshold=threshold,lam=lam)
            old=sol['bandwidth']; action=sol['action']
            for field in info: info[field]+=sol.get(field,0.)
        elif trigger:
            sol=None
            # Plan with a 3% extra operating reserve, assessed against target.
            planning_target=target+.03*pred
            if old is not None and policy not in ['every_assessment','periodic']:
                bad=service_margin(old,mu,sd,target)<0
                if bad.mean()<=threshold:
                    sol=solve_allocation(mu,sd,planning_target,mode='bandwidth',old_b=old,affected=bad)
                    for field in info: info[field]+=sol.get(field,0.)
                    action='bandwidth_repair'
                    if sol['bandwidth'] is None:
                        # Donor set includes the largest available margin.
                        affected=bad.copy(); slack=service_margin(old,mu,sd,target)
                        donor=int(np.argmax(np.where(bad,-np.inf,slack)))
                        affected[donor]=True
                        if affected.mean()<=threshold:
                            sol=solve_allocation(mu,sd,planning_target,mode='local',old_b=old,affected=affected)
                            for field in info: info[field]+=sol.get(field,0.)
                            action='local_repair'
            if sol is None or sol['bandwidth'] is None:
                key=(int(j),'full')
                if cache is not None and key in cache:
                    sol=cache[key]
                else:
                    sol=solve_allocation(mu,sd,planning_target)
                    if cache is not None: cache[key]=sol
                for field in info: info[field]+=sol.get(field,0.)
                action='full_recompute'
            if sol['bandwidth'] is None:
                old=matching_plan(mu,pred,lam=lam)['bandwidth']; action='infeasible_matching_fallback'
            else: old=sol['bandwidth']
        margin=service_margin(old,mu,sd,target)
        safe_s=float(np.min(np.maximum(margin,0)/L))
        if policy=='adaptive_interval' and assess:
            # Down-round to assessment resolution; never extend a short safe
            # interval through a lower clip. Sub-cycle safety is not claimed.
            interval=int(np.floor(beta*safe_s/cycle_s))
            next_epoch=pos+max(1,min(8,interval))
        elapsed=time.perf_counter()-start; cpu=time.process_time()-cpu
        pattern=old>1e-6
        changes=0 if last_pattern is None else int(np.count_nonzero(pattern!=last_pattern))
        last_pattern=pattern.copy()
        z=np.random.default_rng(77123+int(j)).normal(size=(draws,8,12))
        row=dict(policy=policy,cycle_index=pos,forecast_index=int(j),origin=int(f['origins'][j]),
             action=action,assessment_performed=assess,min_margin_mbit=float(margin.min()),
             accepted_certificate=bool(margin.min()>=-2e-5),safe_interval_s=safe_s,
             safe_interval_below_cycle=bool(beta*safe_s<cycle_s),gap_bound=gap,
             beta=beta,rho=rho,affected_fraction_threshold=threshold,wall_s=elapsed,cpu_s=cpu,
             schedule_changes=changes,**info,**metrics(old,mu,sd,f['truth'][j],z))
        rows.append(row)
        # End-of-cycle truth becomes available only after allocation is fixed.
        e=abs(f['truth'][j]-pred)/np.maximum(reserve,1e-6)
        err=gamma**12*err+(1-gamma**12)*e
    return rows


def run_operational(root,count=120,draws=512,scenario_filter=None):
    root=Path(root); allrows=[]; tuning=[]
    for folder in sorted((root/'forecast').iterdir()):
        if not (folder/'predictions.npz').exists(): continue
        if scenario_filter is not None and folder.name!=scenario_filter: continue
        scenario=folder.name
        f=dict(np.load(folder/'predictions.npz'))
        params=json.loads((root/'comparison'/scenario/'parameters.json').read_text())
        cal=np.flatnonzero(f['calibration']); L,observed=estimate_L(f,cal)
        out=root/'operational'/scenario; out.mkdir(parents=True,exist_ok=True)
        write_json(out/'deterioration.json',dict(L_mbit_per_s=L,calibration_quantile=.99,inflation=1.2,
                   interpretation='empirical fixed-plan deterioration; not deterministic Lipschitz constants'))
        # Calibration replay, cached full solves across parameter combinations.
        # Every final test policy runs normally, without using this cache.
        cache={}; candidates=[]; cal_idx=cal[:12]
        with Meter(root,'operational_'+scenario):
            for beta in [.5,.7,.9]:
                for rho in [.01,.03,.05]:
                    for threshold in [.25,.5]:
                        rows=replay(f,cal_idx,'adaptive_interval',L,beta=beta,rho=rho,threshold=threshold,
                                    lam=params['lam'],draws=32,cache=cache)
                        violation=np.mean([r['exact_gaussian_average_violation'] for r in rows])
                        time_cost=sum(r['solver_s'] for r in rows)
                        loss=violation+1e-3*time_cost
                        r=dict(scenario=scenario,beta=beta,rho=rho,threshold=threshold,
                               calibration_violation=violation,cumulative_solver_s=time_cost,loss=loss)
                        candidates.append(r); tuning.append(r)
            best=min(candidates,key=lambda x:x['loss'])
            idx=np.flatnonzero(f['test'])[:count]
            rows=[]
            for policy in POLICIES:
                result=replay(f,idx,policy,L,beta=best['beta'],rho=best['rho'],threshold=best['threshold'],
                        lam=params['lam'],gamma=params['gamma'],kappa=params['kappa'],
                        initial_error=np.array(params['calibration_error_state']),draws=draws)
                for r in result: r['scenario']=scenario
                rows.extend(result)
                write_csv(out/'cycles.csv',rows)
                print('operational',scenario,policy,len(idx),flush=True)
        write_json(out/'parameters.json',best)
        # Test fixed-plan deterioration at previously unseen epochs.
        _,test_changes=estimate_L(f,idx)
        write_json(out/'L_test.json',dict(cell_exceedance=(test_changes>L).mean(0),
                   any_cell_exceedance=float(np.any(test_changes>L,axis=1).mean())))
        allrows.extend(rows)
    write_csv(root/'operational_results.csv',allrows); write_csv(root/'operational_tuning.csv',tuning)
