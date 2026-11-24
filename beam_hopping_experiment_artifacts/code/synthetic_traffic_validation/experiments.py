"""Paired fixed-cycle allocation comparisons on untouched test cycles."""
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy.special import ndtr

from .allocation import (matching_plan,trust_target,hybrid,solve_allocation,report_channel,
                        service_margin,KAPPA)
from .resources import Meter,write_csv,write_json

MATCHING_POLICIES=['throughput_matching','urgency_no_prediction','prediction_matching','reserve_matching','trust_matching',
                   'fixed_reserve_1pct','fixed_reserve_2pct','fixed_reserve_3pct','fixed_reserve_5pct']
REFERENCES=['certified_matching','nominal_horizon','chance_horizon','chance_equal_bandwidth']


def metrics(b,mu,sd,demand,z):
    mean=(b*mu).sum(1); std=np.linalg.norm(b*sd,axis=1)
    service=np.maximum(np.einsum('dnt,nt->dn',mu[None]+sd[None]*z,b),0.)
    useful=np.minimum(service,demand); short=np.maximum(demand-service,0.)
    cov=np.minimum(service/np.maximum(demand,1e-12),1.)
    pv=np.mean(service<demand-1e-6,axis=0)
    exact=np.where(std>1e-12,ndtr((demand-mean)/np.maximum(std,1e-12)),(mean<demand).astype(float))
    sat=cov.mean(0)
    row=dict(raw_mbit=float(service.sum(1).mean()),useful_mbit=float(useful.sum(1).mean()),
         unmet_mbit=float(short.sum(1).mean()),demand_mbit=float(demand.sum()),
         horizon_coverage=float(useful.sum(1).mean()/max(demand.sum(),1e-12)),
         mean_cell_coverage=float(cov.mean()),min_cell_coverage=float(cov.min(1).mean()),
         jain_fairness=float(sat.sum()**2/(len(sat)*np.sum(sat**2)+1e-12)),
         average_cell_violation=float(pv.mean()),max_cell_violation=float(pv.max()),
         any_cell_violation=float(np.mean(np.any(service<demand-1e-6,axis=1))),
         exact_gaussian_average_violation=float(exact.mean()),exact_gaussian_max_violation=float(exact.max()))
    row.update({f'cell_{i}_violation':pv[i] for i in range(len(demand))})
    return row


def update_error(error,actual_slots,predicted_slots,scale,gamma):
    for h in range(actual_slots.shape[1]):
        error=gamma*error+(1-gamma)*abs(actual_slots[:,h]-predicted_slots[:,h])/scale
    return error


def select_parameters(root,scenario,y,f,cfg):
    """Calibrate policy parameters without reading any test origins."""
    idx=np.flatnonzero(f['calibration'])[:96]
    reserve=f['reserve']; truth=f['truth']; pred=f['farima']; H=cfg['slots']
    rows=[]
    for lam in [.5,1.,2.,4.,8.]:
        losses=[]
        for j in idx[::4]:
            mu,sd=report_channel(cycle=j)
            b=matching_plan(mu,pred[j],lam=lam)['bandwidth']
            mean=(b*mu).sum(1); std=np.linalg.norm(b*sd,axis=1)
            losses.append(float(ndtr((truth[j]-mean)/np.maximum(std,1e-12)).mean()))
        rows.append(dict(lam=lam,loss=float(np.mean(losses))))
    lam=min(rows,key=lambda r:r['loss'])['lam']
    trials=[]
    for gamma in [.8,.9,.95]:
        for kappa in [.5,1.,2.,4.]:
            err=np.zeros(8); loss=[]
            for j in idx:
                mu,sd=report_channel(cycle=j)
                theta=np.exp(-kappa*err)
                b=matching_plan(mu,trust_target(pred[j],reserve,theta),lam=lam)['bandwidth']
                mean=(b*mu).sum(1); std=np.linalg.norm(b*sd,axis=1)
                loss.append(float(ndtr((truth[j]-mean)/np.maximum(std,1e-12)).mean()))
                t=f['origins'][j]
                err=update_error(err,y[:,t:t+H],f['trajectory'][j],f['one_step_scale'],gamma)
            trials.append(dict(gamma=gamma,kappa=kappa,loss=float(np.mean(loss))))
    best=min(trials,key=lambda r:r['loss'])
    # Initialize test trust from all preceding calibration observations in order.
    err=np.zeros(8)
    for j in np.flatnonzero(f['calibration']):
        t=f['origins'][j]; err=update_error(err,y[:,t:t+H],f['trajectory'][j],f['one_step_scale'],best['gamma'])
    params=dict(lam=lam,gamma=best['gamma'],kappa=best['kappa'],calibration_error_state=err.tolist(),
                calibration_origins=len(idx),selection_objective='mean exact conditional Gaussian cell violation')
    out=Path(root)/'comparison'/scenario; out.mkdir(parents=True,exist_ok=True)
    write_json(out/'parameters.json',params)
    write_csv(out/'lambda_selection.csv',rows); write_csv(out/'trust_selection.csv',trials)
    return params


def policy_plan(policy,mu,sd,pred,reserve,baseline,theta,lam=1.,threshold=.25):
    if policy.startswith('fixed_reserve_'):
        fraction=float(policy.removeprefix('fixed_reserve_').removesuffix('pct'))/100
        return matching_plan(mu,pred*(1+fraction),lam=lam)
    if policy=='throughput_matching': return matching_plan(mu,np.zeros(8),lam=0.)
    if policy=='urgency_no_prediction': return matching_plan(mu,baseline,lam=lam)
    if policy=='prediction_matching': return matching_plan(mu,pred,lam=lam)
    if policy=='reserve_matching': return matching_plan(mu,pred+reserve,lam=lam)
    if policy=='trust_matching': return matching_plan(mu,trust_target(pred,reserve,theta),lam=lam)
    if policy=='certified_matching': return hybrid(mu,sd,pred,reserve,theta,threshold=threshold,lam=lam)
    if policy=='nominal_horizon': return solve_allocation(mu,sd,pred,chance=False)
    if policy=='chance_horizon': return solve_allocation(mu,sd,pred+reserve)
    if policy=='chance_equal_bandwidth': return solve_allocation(mu,sd,pred+reserve,mode='equal')
    raise ValueError(policy)


def select_test_indices(f,count):
    idx=np.flatnonzero(f['test'])
    return idx[np.linspace(0,len(idx)-1,min(count,len(idx)),dtype=int)]


def compare(root,cycles=24,draws=512,command='all',scenario_filter=None):
    root=Path(root); output=[]; latency=[]
    policies=MATCHING_POLICIES if command=='run-matching' else REFERENCES if command=='run-chance-reference' else MATCHING_POLICIES+REFERENCES
    for folder in sorted((root/'forecast').iterdir()):
        if not (folder/'predictions.npz').exists(): continue
        if scenario_filter is not None and folder.name!=scenario_filter: continue
        scenario=folder.name; cfg=json.loads((root/'data'/scenario/'config.json').read_text())
        y=np.load(root/'data'/scenario/'traffic.npz')['traffic_bits']/1e6
        f=dict(np.load(folder/'predictions.npz')); params_path=root/'comparison'/scenario/'parameters.json'
        params=json.loads(params_path.read_text()) if params_path.exists() else select_parameters(root,scenario,y,f,cfg)
        idx=select_test_indices(f,cycles)
        # Forecast error sensitivity uses the two alpha=1.6 main scenarios.
        settings=[('natural_combined',1.,0.,True)]
        if scenario.endswith('1.6'):
            settings += [('perfect_demand',0.,0.,True),('error_half',.5,0.,True),
                         ('error_1_5',1.5,0.,True),('error_double',2.,0.,True),
                         ('underprediction_10pct',1.,-.1,True),('demand_only',1.,0.,False)]
        rowfile=root/'comparison'/scenario/(command+'_results.csv')
        timefile=root/'comparison'/scenario/(command+'_latency.csv')
        import csv
        rows=list(csv.DictReader(rowfile.open())) if rowfile.exists() else []
        times=list(csv.DictReader(timefile.open())) if timefile.exists() else []
        completed={(r['condition'],r['policy'],int(r['origin'])) for r in rows}
        err_state=np.array(params['calibration_error_state'])
        theta_by_index={}
        # Native history determines trust. Controlled forecast interventions
        # affect the current cycle only, modelling an unexpected error shock.
        for j in np.flatnonzero(f['test']):
            if j in idx: theta_by_index[j]=np.exp(-params['kappa']*err_state)
            t=f['origins'][j]
            err_state=update_error(err_state,y[:,t:t+12],f['trajectory'][j],f['one_step_scale'],params['gamma'])
        with Meter(root,'compare_'+command.replace('-','_')+'_'+scenario):
            for condition,c,bias,channel_random in settings:
                for rep,j in enumerate(idx):
                    mu,sd=report_channel(cycle=j)
                    if not channel_random: sd=np.zeros_like(sd)
                    demand=f['truth'][j]
                    pred=np.maximum(demand+c*(f['farima'][j]-demand)+bias*demand,0.)
                    # No future-dependent uncertainty in perfect-demand control.
                    reserve=np.zeros(8) if c==0 else f['reserve']
                    baseline=f['mean'][j]; theta=theta_by_index[j]
                    z=np.random.default_rng(77123+int(j)).normal(size=(draws,8,12))
                    for policy in policies:
                        if (condition,policy,int(f['origins'][j])) in completed: continue
                        wall=time.perf_counter(); cpu=time.process_time()
                        sol=policy_plan(policy,mu,sd,pred,reserve,baseline,theta,params['lam'])
                        elapsed=time.perf_counter()-wall; cpu=time.process_time()-cpu
                        valid=sol['bandwidth'] is not None
                        b=sol['bandwidth'] if valid else matching_plan(mu,pred,lam=params['lam'])['bandwidth']
                        row=dict(scenario=scenario,condition=condition,error_scale=c,bias_fraction=bias,
                             replicate=rep,origin=int(f['origins'][j]),policy=policy,status=sol['status'],
                             optimization_feasible=bool(valid and 'infeasible' not in sol['status'] and 'infeasible' not in sol.get('action','')),action=sol.get('action','allocation'),
                             nmae=float(abs(pred-demand).sum()/max(demand.sum(),1e-12)),
                             accepted_certificate=sol.get('accepted_certificate',False),
                             empirical_demand_reserve=True,channel_constraint_exact_given_gaussian_model=True,
                             **metrics(b,mu,sd,demand,z))
                        timing=dict(scenario=scenario,condition=condition,replicate=rep,policy=policy,wall_s=elapsed,cpu_s=cpu,
                             process_lifetime_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                             **{k:sol.get(k,0.) for k in ['score_s','matching_s','solver_s','certificate_s','solve_wall_s','solver_calls','full_calls','repair_calls']})
                        rows.append(row); times.append(timing)
                    if rep%8==0: print('compared',command,scenario,condition,rep,flush=True)
                # Checkpoint at completed conditions supports resumable analysis.
                write_csv(rowfile,rows)
                write_csv(timefile,times)
        output.extend(rows); latency.extend(times)
    write_csv(root/(command+'_results.csv'),output); write_csv(root/(command+'_latency.csv'),latency)
