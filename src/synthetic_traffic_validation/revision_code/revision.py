"""Isolated October revision. Run from the parent experiment environment.

Existing traffic, strategy definitions and previous outputs are immutable inputs.
The four stages are prepare, pilot, formal and summarize. No sequential run is
implemented here; that experiment requires the user's checkpoint confirmation.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

PARENT = Path(__file__).resolve().parent.parent
from ..allocation import matching_plan, solve_allocation, service_margin, report_channel
from ..experiments import metrics, update_error
from ..forecast import fit_arfima, step_weights, predict_at, quantile_reserve, errors
from ..resources import Meter, environment, write_csv, write_json

OLD = Path(os.environ.get('BEAM_HOPPING_DATA_ROOT', PARENT / 'runs/formal')).resolve()
OUT = Path(os.environ.get('BEAM_HOPPING_REVISION_ROOT', PARENT / 'runs/revision_20261026')).resolve()
SCENARIOS = ('baseline_a1.6', 'hotspot_a1.6')
HORIZONS = (12, 24, 48)
CORE = ('throughput_matching', 'urgency_no_prediction', 'prediction_matching',
        'reserve_matching', 'nominal_horizon', 'chance_horizon')
EXTRA = ('static_trust_matching', 'certified_static_matching')
TIME_LIMIT = 3.0
SEEN_SOLVER_TEMPLATES = set()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def traffic(scenario):
    return np.load(OLD / 'data' / scenario / 'traffic.npz')['traffic_bits'] / 1e6


def origins(start, end, horizon, stride=1):
    return np.arange(start, end-horizon+1, horizon*stride, dtype=int)


def channel(horizon, origin, seed=20260916):
    """Same slot-independent error model; known profile indexed in physical time.

    Twelve-slot blocks extend the original fixture without altering its first
    block. The drift phase uses physical slots/12, never a horizon-specific row.
    This is a synthetic spectral-efficiency fixture, not a propagation model.
    """
    cycle = (origin-800000)/12
    mu0, _ = report_channel(seed=seed, cycle=cycle)
    drift = 1+.07*np.sin(cycle/17+np.arange(8)[:, None]*.6)
    parts = [mu0]
    quality = np.random.default_rng(seed).uniform(2.8, 4.8, (8, 1))
    for block in range(1, math.ceil(horizon/12)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, block]))
        parts.append(quality*rng.lognormal(0, .12, (8, 12))*drift)
    mu = np.concatenate(parts, axis=1)[:, :horizon]
    return mu, .2*mu


def forecast_one(y, origin, H, models, scale=1.):
    # Precomputed linear filters are model artifacts, not future observations.
    return scale*np.maximum(np.array([
        (y[i, origin-m['history']:origin][::-1]-m['fit']['mean']) @ m['weights']
        + H*m['fit']['mean'] for i, m in enumerate(models)]), 0.)


def model_bundle(scenario, H):
    models = read(OUT / 'forecast' / scenario / str(H) / 'models.json')
    for m in models:
        m['weights'] = np.asarray(m['weights'])
    return models


def prepare():
    if (OUT/'prepared.json').exists():
        raise RuntimeError('Preparation already completed; use the existing artifacts.')
    OUT.mkdir(parents=True, exist_ok=True)
    environment(OUT)
    manifest = dict(horizons=list(HORIZONS), cells=8, max_active=3,
        slot_duration_s=.002, total_bandwidth_hz=500e6, total_power_w=1000,
        bmax=.6, epsilon_demand=.025, epsilon_channel=.025,
        alpha=1.6, matching_equal_fraction=1/3, solver_time_limit_s=TIME_LIMIT,
        physical_splits=[0, 600000, 700000, 800000, 1000000],
        load_initial=[.75, 1., 1.25, 1.5, 1.75],
        load_extension=[2., 2.25],
        extension_rule='Append until both spatial scenarios have a proved-infeasible pilot window, at most 2.25.',
        preliminary_windows_per_load=3,
        budget_rule='95% worst-case half-width 0.18 gives ceil(1.96^2/(4*0.18^2))=30 windows; cap by 900 s projected full-solver budget; minimum 8, maximum 30.',
        window_sampling='Evenly spaced nonoverlapping windows across the physical test partition; fixed before formal evaluation.',
        formal_draws=512, uncertainty='Independent Gaussian spectral-efficiency errors; standard deviation 20% of the known mean.',
        history_slots=[128, 256, 512], history_seconds=[.256, .512, 1.024],
        inherited_policies='Unchanged lambda, gamma and kappa from the preceding calibration; no priority-rule changes.',
        demand_scaling='All partitions, forecasts, means and residual reserves scale by the same positive factor. Standardized linear fits and model choices are invariant.',
        information='Forecast at first target slot using strictly earlier observations; complete-cycle cumulative service, without queues or TTL.',
        test_disclosure='The physical test trajectories were already inspected in the preceding study. This is not fresh holdout confirmation.',
        channel_extension='Original 12-slot profile plus independently seeded 12-slot blocks; same conditional independent-slot errors. Drift phase follows physical origin/12.',
        latency='Zero-lead-time logical simulation. Timed forecast, scoring, assignment, all checks, all failed/successful repairs, and full solve; hardware configuration signalling unavailable.',
        checkpoint='Stop after B and C; sequential old-plan reuse has not been authorized.',
        source_sha256={name:digest(PARENT/name) for name in ['forecast.py','allocation.py','experiments.py','operational.py']})
    write_json(OUT/'protocol.json', manifest)
    rows=[]; audit=[]; selections=[]; tests=[]
    for scenario in SCENARIOS:
        y=traffic(scenario); assert y.shape==(8,1000000)
        cfg=read(OLD/'data'/scenario/'config.json')
        params=read(OLD/'comparison'/scenario/'parameters.json')
        write_json(OUT/'inputs'/f'{scenario}.json', dict(config=cfg, policy=params,
            traffic_sha256=digest(OLD/'data'/scenario/'traffic.npz')))
        audit.append(dict(scenario=scenario, training_total_mean_mbit_12=float(12*y[:,:600000].mean(1).sum()),
            full_total_mean_mbit_12=float(12*y.mean(1).sum()),
            first_two_cells_share=float(y[:2].sum()/y.sum()), users=sum(cfg['users']),
            source_count_unchanged=True, complete_graph='Every beam opportunity can serve every cell; opportunities have identical channel weights. No sparse graph is generated.'))
        fits_path=OUT/'forecast'/scenario/'training_fits.json'
        with Meter(OUT,'fit_'+scenario):
            fits=[]
            for i in range(8):
                fits.append(dict(farima=[fit_arfima(y[i,:600000],p) for p in [0,1,2]],
                                 ar=fit_arfima(y[i,:600000],2,d_fixed=0.)))
                print('training fits',scenario,i,flush=True)
            write_json(fits_path,fits)
        cum=np.c_[np.zeros(8),y.cumsum(1)]
        for H in HORIZONS:
            with Meter(OUT,f'forecast_{scenario}_{H}'):
                select=origins(600000,700000,H,4)
                cal=origins(700000,800000,H); test=origins(800000,1000000,H)
                oo=np.r_[cal,test]; is_test=np.r_[np.zeros(len(cal),bool),np.ones(len(test),bool)]
                true=(cum[:,oo+H]-cum[:,oo]).T
                select_true=(cum[:,select+H]-cum[:,select]).T
                preds={name:np.zeros_like(true) for name in ['mean','persistence','ar','farima']}
                models=[]
                for i in range(8):
                    best=None
                    for fit in fits[i]['farima']:
                        for L in [128,256,512]:
                            W=step_weights(fit,L,H)
                            pp=np.maximum(predict_at(y[i],select,W.sum(0),fit['mean'],H),0.)
                            mse=float(np.mean((pp-select_true[:,i])**2))
                            selections.append(dict(scenario=scenario,horizon=H,cell=i,p=fit['p'],history=L,selection_mse=mse))
                            if best is None or mse<best[0]: best=(mse,fit,L,W)
                    _,fit,L,W=best
                    ar=fits[i]['ar']; arw=step_weights(ar,L,H).sum(0)
                    models.append(dict(cell=i,history=L,history_seconds=L*.002,fit=fit,ar_fit=ar,
                        weights=W.sum(0).tolist(),step_weights=W.tolist()))
                    preds['farima'][:,i]=np.maximum(predict_at(y[i],oo,W.sum(0),fit['mean'],H),0.)
                    preds['mean'][:,i]=H*y[i,:600000].mean()
                    preds['persistence'][:,i]=H*y[i,oo-1]
                    preds['ar'][:,i]=np.maximum(predict_at(y[i],oo,arw,ar['mean'],H),0.)
                reserve=quantile_reserve(true[~is_test]-preds['farima'][~is_test])
                folder=OUT/'forecast'/scenario/str(H); folder.mkdir(parents=True,exist_ok=True)
                write_json(folder/'models.json',models)
                np.savez_compressed(folder/'forecast.npz',origins=oo,test=is_test,truth=true,reserve=reserve,**preds)
                for name,pred in preds.items():
                    for i in range(8):
                        for part,mask in [('calibration',~is_test),('test',is_test)]:
                            rows.append(dict(scenario=scenario,horizon=H,duration_ms=H*2,model=name,cell=i,partition=part,
                                **errors(pred[mask,i],true[mask,i]),
                                relative_rmse=float(np.sqrt(np.mean((pred[mask,i]-true[mask,i])**2))/true[mask,i].mean()),
                                reserve_mbit=float(reserve[i]),reserve_exceedance=float(np.mean(true[mask,i]-pred[mask,i]>reserve[i]))))
                for part,oo_part,stop in [('selection',select,700000),('calibration',cal,800000),('test',test,1000000)]:
                    assert np.all(oo_part+H<=stop) and np.all(np.diff(oo_part)>=H)
                    tests.append(dict(test=f'{scenario}_{H}_{part}_boundary_and_nonoverlap',passed=True))
                got=forecast_one(y,int(test[3]),H,models)
                ix=np.where(oo==test[3])[0][0]
                np.testing.assert_allclose(got,preds['farima'][ix],atol=1e-10)
                # A future intervention must not change the predictor input.
                o=int(test[3]); modified=y[:,o-512:o+H].copy(); modified[:,512:]+=1000
                local=np.array([(modified[i,512-m['history']:512][::-1]-m['fit']['mean'])@np.asarray(m['weights'])+H*m['fit']['mean'] for i,m in enumerate(models)])
                np.testing.assert_allclose(np.maximum(local,0),got,atol=1e-10)
                tests.append(dict(test=f'{scenario}_{H}_no_future_access_and_online_filter',passed=True))
                mu,sd=channel(H,o); assert mu.shape==(8,H)
                b=matching_plan(mu,got,lam=params['lam'])['bandwidth']
                assert np.allclose(b.sum(0),1) and np.all((b>0).sum(0)==3)
                solved=solve_allocation(mu,sd,np.ones(8)*.1*H/12,time_limit=TIME_LIMIT)
                assert solved['bandwidth'] is not None
                tests.append(dict(test=f'{scenario}_{H}_matching_and_full_solver_smoke',passed=True,status=solved['status']))
                if H==12:
                    old_models=read(OLD/'forecast'/scenario/'models.json')
                    old=np.load(OLD/'forecast'/scenario/'predictions.npz'); j=int(np.flatnonzero(old['test'])[2])
                    om=[dict(m,weights=step_weights(m['fit'],m['history'],12).sum(0)) for m in old_models]
                    np.testing.assert_allclose(forecast_one(y,int(old['origins'][j]),12,om),old['farima'][j],atol=1e-10)
                    a,c=channel(12,800000+12*7); a0,c0=report_channel(cycle=7)
                    np.testing.assert_allclose(a,a0); np.testing.assert_allclose(c,c0)
                    tests.append(dict(test=scenario+'_legacy_12_forecast_and_channel_reproduced',passed=True))
            print('prepared',scenario,H,flush=True)
    write_csv(OUT/'forecast_metrics.csv',rows); write_csv(OUT/'model_selection.csv',selections)
    write_csv(OUT/'source_audit.csv',audit); write_csv(OUT/'tests.csv',tests)
    write_json(OUT/'prepared.json',dict(tests=len(tests),completed=True))


def status_class(result):
    if result['bandwidth'] is not None:
        return 'optimal' if result['status']=='optimal' else 'feasible_incumbent_unproved_optimal'
    if result['status']=='infeasible': return 'proved_infeasible'
    return 'unresolved_without_feasible_incumbent'


def solve_record(mu,sd,target,attempts,**kwargs):
    key=(mu.shape,kwargs.get('mode','full'),bool(kwargs.get('chance',True) and np.any(sd)))
    startup=key not in SEEN_SOLVER_TEMPLATES
    SEEN_SOLVER_TEMPLATES.add(key)
    start=time.perf_counter()
    result=solve_allocation(mu,sd,target,time_limit=TIME_LIMIT,**kwargs)
    attempts.append(dict(mode=kwargs.get('mode','full'),chance=kwargs.get('chance',True),
        status=result['status'],classification=status_class(result),wall_s=time.perf_counter()-start,
        solver_s=result['solver_s'],first_template_call=startup))
    return result


def hybrid_plan(mu,sd,pred,reserve,theta,lam,attempts,threshold=.25):
    plan=matching_plan(mu,pred+(1-theta)*reserve,lam=lam)
    start=time.perf_counter(); margin=service_margin(plan['bandwidth'],mu,sd,pred+reserve)
    checks=time.perf_counter()-start
    if margin.min()>=-1e-6:
        return plan,'matching_direct',checks
    bad=margin<0; result=None
    if bad.mean()<=threshold:
        result=solve_record(mu,sd,pred+reserve,attempts,mode='bandwidth',old_b=plan['bandwidth'],affected=bad)
        if result['bandwidth'] is None:
            remaining=np.where(~bad)[0]; extra=max(0,int(threshold*8)-int(bad.sum()))
            affected=bad.copy()
            if extra: affected[remaining[np.argsort(margin[remaining])[-extra:]]]=True
            result=solve_record(mu,sd,pred+reserve,attempts,mode='local',old_b=plan['bandwidth'],affected=affected)
        if result['bandwidth'] is not None:
            result.update(score_s=plan['score_s'],matching_s=plan['matching_s'])
            return result,'repair_accepted',checks
    result=solve_record(mu,sd,pred+reserve,attempts)
    if result['bandwidth'] is None:
        plan['status']=result['status']
        return plan,'unprotected_fallback',checks
    result.update(score_s=plan['score_s'],matching_s=plan['matching_s'])
    return result,'full_accepted',checks


def evaluate(scenario,H,scale,indices,policies,condition='natural_random',draws=512):
    y=traffic(scenario); f=np.load(OUT/'forecast'/scenario/str(H)/'forecast.npz')
    models=model_bundle(scenario,H); par=read(OLD/'comparison'/scenario/'parameters.json')
    baseline=np.array([m['fit']['mean']*H for m in models])*scale
    reserve=f['reserve']*scale
    # Static trust is evaluated from the immediately preceding observed window,
    # using frozen per-step training scales. It is not a sequential reuse policy.
    step_scale=np.maximum(np.std(np.diff(y[:,:600000],axis=1),axis=1),1e-6)
    results=[]; attempt_rows=[]
    for idx in indices:
        origin=int(f['origins'][idx]); demand=f['truth'][idx]*scale
        mu,sd=channel(H,origin)
        if condition.endswith('deterministic'): sd=np.zeros_like(sd)
        z=np.random.default_rng(np.random.SeedSequence([77123,origin,H])).normal(size=(draws,8,H))
        trust_wall=time.perf_counter(); trust_cpu=time.process_time()
        prev_origin=origin-H
        prev_pred=np.array([np.maximum(
            np.asarray(m['step_weights'])@(y[i,prev_origin-m['history']:prev_origin][::-1]-m['fit']['mean'])+m['fit']['mean'],0.)
            for i,m in enumerate(models)])
        err=update_error(np.zeros(8),y[:,prev_origin:origin],prev_pred,step_scale,par['gamma'])
        theta=np.exp(-par['kappa']*err)
        trust_s=time.perf_counter()-trust_wall; trust_cpu_s=time.process_time()-trust_cpu
        for policy in policies:
            attempts=[]; wall=time.perf_counter(); cpu=time.process_time()
            needs_forecast=policy not in ('throughput_matching','urgency_no_prediction')
            pred=forecast_one(y,origin,H,models,scale) if needs_forecast else np.zeros(8)
            forecast_s=time.perf_counter()-wall
            if condition.startswith('perfect'): pred=demand.copy()
            checks=0.; path='not_applicable'; fallback=False
            if policy in ('certified_static_matching',):
                plan,path,checks=hybrid_plan(mu,sd,pred,reserve,theta,par['lam'],attempts)
            elif policy in ('nominal_horizon','chance_horizon','chance_equal_bandwidth'):
                target=pred if policy=='nominal_horizon' else pred+reserve
                plan=solve_record(mu,sd,target,attempts,chance=policy!='nominal_horizon',
                                  mode='equal' if policy=='chance_equal_bandwidth' else 'full')
                if plan['bandwidth'] is None:
                    fallback=True
                    rescue=matching_plan(mu,pred+reserve,lam=par['lam'])
                    plan.update(bandwidth=rescue['bandwidth'],score_s=rescue['score_s'],matching_s=rescue['matching_s'])
            else:
                target={'throughput_matching':np.zeros(8),'urgency_no_prediction':baseline,
                    'prediction_matching':pred,'reserve_matching':pred+reserve,
                    'static_trust_matching':pred+(1-theta)*reserve}[policy]
                plan=matching_plan(mu,target,lam=0 if policy=='throughput_matching' else par['lam'])
            check_start=time.perf_counter()
            margin=service_margin(plan['bandwidth'],mu,sd,pred+reserve)
            # Include this validation, even for a rejected or unprotected plan.
            checks+=time.perf_counter()-check_start
            total=time.perf_counter()-wall; cpu_s=time.process_time()-cpu
            uses_trust=policy in EXTRA
            if uses_trust:
                total+=trust_s; cpu_s+=trust_cpu_s
            protected=bool(margin.min()>=-2e-5) and (policy in ('chance_horizon','chance_equal_bandwidth','certified_static_matching')) and not fallback and path!='unprotected_fallback'
            row=dict(scenario=scenario,horizon=H,scale=scale,origin=origin,
                last_available_observation=origin-1,target_end_exclusive=origin+H,
                condition=condition,policy=policy,duration_s=H*.002,
                status=plan['status'],solver_classification=attempts[-1]['classification'] if attempts else 'exact_matching_no_feasibility_claim',
                unprotected_solver_fallback=fallback,final_path=path,conditional_margin_pass=protected,
                min_margin_mbit=float(margin.min()),forecast_s=forecast_s,
                trust_update_s=trust_s if uses_trust else 0.,
                first_template_call=any(a['first_template_call'] for a in attempts),
                score_s=plan.get('score_s',0.),matching_s=plan.get('matching_s',0.),check_s=checks,
                all_optimizer_wall_s=sum(a['wall_s'] for a in attempts),solver_s=sum(a['solver_s'] for a in attempts),
                repair_attempts=sum(a['mode'] in ['bandwidth','local'] for a in attempts),
                full_attempts=sum(a['mode'] in ['full','equal'] for a in attempts),
                total_online_s=total,cpu_s=cpu_s,latency_window_ratio=total/(H*.002),exceeds_window=total>H*.002,
                process_lifetime_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                configuration_signalling_s='unavailable',
                **metrics(plan['bandwidth'],mu,sd,demand,z))
            row['useful_mbps']=row['useful_mbit']/(H*.002); row['raw_mbps']=row['raw_mbit']/(H*.002)
            # Cellwise mean coverage is distinct from mean per-draw minimum.
            service=np.maximum(np.einsum('dnt,nt->dn',mu[None]+sd[None]*z,plan['bandwidth']),0.)
            cellcov=np.minimum(service/np.maximum(demand,1e-12),1).mean(0)
            row['minimum_of_cell_mean_coverage']=float(cellcov.min())
            for i in range(8): row[f'cell_{i}_coverage']=float(cellcov[i])
            results.append(row)
            for k,a in enumerate(attempts):
                attempt_rows.append(dict(scenario=scenario,horizon=H,scale=scale,origin=origin,condition=condition,policy=policy,attempt=k+1,**a))
    return results,attempt_rows


def pilot():
    if (OUT/'formal_budget.json').exists(): raise RuntimeError('Budget already frozen; do not rerun pilot.')
    pilotrows=[]; calls=[]; equal=[]; all_levels=[]
    for scale in [.75,1.,1.25,1.5,1.75,2.,2.25]:
        if scale>=2 and all(any(r['scale']==all_levels[-1] and r['scenario']==s and r['policy']=='chance_horizon' and r['solver_classification']=='proved_infeasible' for r in pilotrows) for s in SCENARIOS): break
        all_levels.append(scale)
        for scenario in SCENARIOS:
            f=np.load(OUT/'forecast'/scenario/'12'/'forecast.npz')
            cal=np.flatnonzero(~f['test']); idx=cal[np.linspace(0,len(cal)-1,3,dtype=int)]
            r,a=evaluate(scenario,12,scale,idx,('nominal_horizon','chance_horizon'),draws=128)
            pilotrows.extend(r); calls.extend(a)
            # Small equal-resource reference diagnoses a different feasible set.
            r,a=evaluate(scenario,12,scale,idx[:1],('chance_equal_bandwidth',),draws=128)
            equal.extend(r); calls.extend(a)
        write_csv(OUT/'pilot.csv',pilotrows); write_csv(OUT/'pilot_equal.csv',equal); write_csv(OUT/'pilot_attempts.csv',calls)
        print('pilot load',scale,flush=True)
    representatives={}
    for scenario in SCENARIOS:
        good=[s for s in all_levels if all(r['solver_classification'] in ['optimal','feasible_incumbent_unproved_optimal'] for r in pilotrows if r['scenario']==scenario and r['scale']==s and r['policy']=='chance_horizon')]
        boundary=max(good) if good else min(all_levels)
        representatives[scenario]={'slack':min(all_levels),'boundary':boundary}
    costs=[r['total_online_s'] for r in pilotrows if r['policy']=='chance_horizon']
    estimated=max(float(np.quantile(costs,.8)),.05)
    precision_n=math.ceil(1.96**2/(4*.18**2))
    count=max(8,min(precision_n,int(900/(len(all_levels)*2*2*estimated))))
    # A short calibration pilot for longer horizons informs a bounded budget.
    longer=[]
    for H in [24,48]:
        for scenario in SCENARIOS:
            f=np.load(OUT/'forecast'/scenario/str(H)/'forecast.npz'); idx=np.flatnonzero(~f['test'])[[0]]
            r,a=evaluate(scenario,H,representatives[scenario]['boundary'],idx,('nominal_horizon','chance_horizon'),draws=128)
            longer.extend(r); calls.extend(a)
    write_csv(OUT/'longer_horizon_pilot.csv',longer); write_csv(OUT/'pilot_attempts.csv',calls)
    # Sample sizes reflect a deliberately exploratory horizon study, not a claim
    # to resolve a 2.5% violation probability from a few independent windows.
    ccount={str(H):max(4,min(12,int(240/max(8*max(r['total_online_s'] for r in longer if r['horizon']==H),.1)))) for H in [24,48]}
    frozen=dict(levels=all_levels,representatives=representatives,load_windows=count,horizon_windows=ccount,
        hybrid_windows=8,ablation_windows=8,core_policies=list(CORE),extra_policies=list(EXTRA),
        pilot_chance_latency_p80_s=estimated,precision_target_half_width=.18,
        precision_target_n=precision_n,confidence_caveat='Worst-case independent-window planning approximation only. Dependence-sensitive block intervals are reported; repeated channel draws do not increase trajectory count.',
        predicted_core_full_calls=2*len(all_levels)*count*2+sum(2*2*ccount[str(H)]*2 for H in [24,48]),
        hybrid_max_calls=2*3*2*8*3,
        formal_indices={})
    for scenario in SCENARIOS:
        for H in HORIZONS:
            f=np.load(OUT/'forecast'/scenario/str(H)/'forecast.npz'); test=np.flatnonzero(f['test'])
            n=count if H==12 else ccount[str(H)]
            frozen['formal_indices'][f'{scenario}_{H}']=test[np.linspace(0,len(test)-1,n,dtype=int)].tolist()
    write_json(OUT/'formal_budget.json',frozen)
    write_json(OUT/'frozen_code.json',dict(sha256=digest(Path(__file__)),frozen_before_test=True))
    print(json.dumps({k:v for k,v in frozen.items() if k!='formal_indices'},indent=2),flush=True)


def formal():
    if (OUT/'formal_complete.json').exists(): raise RuntimeError('Formal evaluation is complete; do not rerun the observed test.')
    budget=read(OUT/'formal_budget.json'); rows=[]; calls=[]
    def save(stage,scenario,H,scale,indices,policies,condition='natural_random'):
        r,a=evaluate(scenario,H,scale,indices,policies,condition)
        for x in r: x['stage']=stage
        for x in a: x['stage']=stage
        rows.extend(r); calls.extend(a)
        write_csv(OUT/'static_results.csv',rows); write_csv(OUT/'static_attempts.csv',calls)
        print(stage,scenario,H,scale,condition,len(r),'rows',flush=True)
    with Meter(OUT,'formal_B_C'):
        for scenario in SCENARIOS:
            idx=budget['formal_indices'][f'{scenario}_12']
            for scale in budget['levels']:
                save('B_load',scenario,12,scale,idx,CORE)
        for scenario in SCENARIOS:
            for H in [24,48]:
                idx=budget['formal_indices'][f'{scenario}_{H}']
                for label,scale in budget['representatives'][scenario].items():
                    save('C_horizon_'+label,scenario,H,scale,idx,CORE)
        # The core comparison precedes trust and certification, as specified.
        for scenario in SCENARIOS:
            for H in HORIZONS:
                idx=budget['formal_indices'][f'{scenario}_{H}'][:budget['hybrid_windows']]
                for label,scale in budget['representatives'][scenario].items():
                    save('C_hybrid_'+label,scenario,H,scale,idx,EXTRA)
        scenario=SCENARIOS[0]; scale=budget['representatives'][scenario]['boundary']
        idx=budget['formal_indices'][f'{scenario}_12'][:budget['ablation_windows']]
        for condition in ['natural_random','perfect_random','natural_deterministic','perfect_deterministic']:
            # Demand reserve is deliberately IDENTICAL under all interventions.
            save('C_ablation_fixed_reserve',scenario,12,scale,idx,
                 ('prediction_matching','reserve_matching','nominal_horizon','chance_horizon','certified_static_matching'),condition)
    write_json(OUT/'formal_complete.json',dict(rows=len(rows),solver_calls=len(calls),
        stages=['A','B','C'],sequential_D_executed=False,source_sha256=digest(Path(__file__))))


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['prepare','pilot','formal'])
    args=p.parse_args()
    if args.stage=='prepare': prepare()
    elif args.stage=='pilot':
        with Meter(OUT,'pilot'): pilot()
    else: formal()


if __name__=='__main__': main()
