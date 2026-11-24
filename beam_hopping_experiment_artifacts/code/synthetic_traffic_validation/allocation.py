"""Exact assignment, static plan certificates and MOSEK horizon references.

All traffic is Mbit. Certification applies to a plan fixed before the channel
realization. Past observed cycle errors can adjust trust for the next cycle.
"""
import time
from statistics import NormalDist

import numpy as np
from scipy.optimize import linear_sum_assignment

KAPPA=NormalDist().inv_cdf(.975)
_CACHE={}


def exact_matching(weights):
    w=np.asarray(weights,dtype=float)
    if w.ndim!=2 or w.shape[0]>w.shape[1] or not np.all(np.isfinite(w)):
        raise ValueError('finite rectangular weights require beams <= cells')
    r,c=linear_sum_assignment(w,maximize=True)
    x=np.zeros_like(w); x[r,c]=1.
    return x


def trust_target(pred,reserve,theta):
    return np.maximum(pred+(1-np.asarray(theta))*reserve,0.)


def service_margin(bandwidth,mu,sd,target,kappa=KAPPA):
    return (bandwidth*mu).sum(1)-kappa*np.linalg.norm(bandwidth*sd,axis=1)-target


def report_channel(seed=20260916,cycle=0,uncertainty=.2):
    rng=np.random.default_rng(seed)
    quality=rng.uniform(2.8,4.8,(8,1)); variation=rng.lognormal(0,.12,(8,12))
    base=quality*variation
    # Known predictions drift smoothly across cycles; errors conditional on
    # these predictions stay Gaussian and independent across physical slots.
    drift=1+.07*np.sin(cycle/17+np.arange(8)[:,None]*.6)
    mu=base*drift
    return mu,uncertainty*mu


def matching_plan(mu,target,k=3,lam=1.):
    n,T=mu.shape; b=np.zeros_like(mu); delivered=np.zeros(n)
    score_s=0.; solve_s=0.
    for t in range(T):
        start=time.perf_counter(); urgency=np.maximum(target-delivered,0)/(T-t)
        norm=urgency/max(float(urgency.mean()),1e-12)
        weight=np.broadcast_to(mu[:,t]/k,(k,n))*(1+lam*norm[None,:])
        score_s+=time.perf_counter()-start
        start=time.perf_counter(); x=exact_matching(weight); solve_s+=time.perf_counter()-start
        b[:,t]=x.sum(0)/k; delivered+=b[:,t]*mu[:,t]
    return dict(bandwidth=b,illumination=(b>0).astype(float),status='exact_matching',
                score_s=score_s,matching_s=solve_s,solver_s=0.,solver_calls=0,solve_wall_s=solve_s,
                certificate_s=0.,repair_calls=0,full_calls=0,accepted_certificate=False)


def _template(shape,k,chance,bmax,mode,relaxed):
    import cvxpy as cp
    key=(shape,k,chance,bmax,mode,relaxed)
    if key in _CACHE: return _CACHE[key]
    n,T=shape; mu=cp.Parameter(shape,nonneg=True); sd=cp.Parameter(shape,nonneg=True)
    demand=cp.Parameter(n,nonneg=True); old=cp.Parameter(shape,nonneg=True)
    release=cp.Parameter(shape,nonneg=True); b=cp.Variable(shape,nonneg=True)
    constraints=[cp.sum(b,axis=0)<=1]
    if mode=='bandwidth':
        x=old
        constraints += [b<=bmax*x]
        fixed=cp.Parameter(shape,nonneg=True)
        constraints += [b-fixed<=release,b-fixed>=-release]
    else:
        x=cp.Variable(shape,boolean=not relaxed)
        constraints += [cp.sum(x,axis=0)<=k,b<=bmax*x]
        if relaxed: constraints += [x>=0,x<=1]
        fixed=cp.Parameter(shape,nonneg=True)
        if mode=='local':
            constraints += [x-old<=release,x-old>=-release,b-fixed<=release,b-fixed>=-release]
        if mode=='equal': constraints += [b==x/k]
    service=cp.sum(cp.multiply(b,mu),axis=1)
    if chance:
        for i in range(n):
            constraints += [service[i]-KAPPA*cp.norm(cp.multiply(b[i],sd[i]))>=demand[i]]
    else: constraints += [service>=demand]
    problem=cp.Problem(cp.Maximize(cp.sum(service)),constraints)
    item=(problem,b,x,mu,sd,demand,old,release,fixed)
    _CACHE[key]=item
    return item


def solve_allocation(mu,sd,target,*,k=3,chance=True,bmax=.6,mode='full',old_b=None,affected=None,relaxed=False,time_limit=5.):
    import cvxpy as cp
    mu=np.asarray(mu); target=np.maximum(target,0.)
    # A deterministic channel has no SOC allowance. Eliminate the degenerate
    # zero cone rather than ask the native solver to presolve it repeatedly.
    chance=bool(chance and np.any(np.asarray(sd)>0.))
    p,b,x,pm,ps,pd,po,pr,pf=_template(mu.shape,k,chance,bmax,mode,relaxed)
    pm.value=mu; ps.value=sd; pd.value=target
    old_b=np.zeros_like(mu) if old_b is None else np.asarray(old_b)
    po.value=(old_b>1e-8).astype(float); pf.value=old_b
    release=np.ones_like(mu) if affected is None else np.broadcast_to(np.asarray(affected)[:,None],mu.shape).astype(float)
    pr.value=release
    start=time.perf_counter()
    try:
        p.solve(solver='MOSEK',warm_start=True,verbose=False,mosek_params={
            'MSK_IPAR_NUM_THREADS':2,'MSK_DPAR_MIO_MAX_TIME':time_limit,
            'MSK_DPAR_MIO_TOL_REL_GAP':1e-4})
        status=p.status
    except cp.error.SolverError as exc:
        status='solver_error: '+str(exc)
    wall=time.perf_counter()-start
    val=None if b.value is None else np.maximum(np.asarray(b.value),0.)
    feasible=val is not None and status in ['optimal','optimal_inaccurate','user_limit']
    if feasible:
        check=service_margin(val,mu,sd,target) if chance else (val*mu).sum(1)-target
        xv=np.asarray(x.value if isinstance(x,cp.Variable) else po.value)
        feasible=(check.min()>=-2e-5 and np.max(val.sum(0))<=1+2e-5 and np.max(val-bmax*xv)<=2e-5)
        if isinstance(x,cp.Variable) and not relaxed:
            feasible=feasible and np.max(abs(xv-np.rint(xv)))<1e-5
    if not feasible: val=None
    # Only a solved continuous relaxation is used as a true upper bound.
    upper=float(p.value) if relaxed and status=='optimal' else None
    return dict(bandwidth=val,status=status,objective_mbit=float(p.value) if p.value is not None else None,
                upper_bound_mbit=upper,solver_s=float(p.solver_stats.solve_time or 0.) if p.solver_stats else 0.,
                solve_wall_s=wall,solver_calls=1,full_calls=int(mode in ['full','equal']),repair_calls=int(mode in ['bandwidth','local']),
                score_s=0.,matching_s=0.,certificate_s=0.,accepted_certificate=False)


def hybrid(mu,sd,pred,reserve,theta,threshold=.25,lam=1.):
    start=time.perf_counter(); plan=matching_plan(mu,trust_target(pred,reserve,theta),lam=lam)
    margin=service_margin(plan['bandwidth'],mu,sd,pred+reserve)
    plan['certificate_s']=time.perf_counter()-start-plan['score_s']-plan['matching_s']
    plan['action']='matching'
    if margin.min()>=-1e-6:
        plan['accepted_certificate']=True
        return plan
    bad=margin<0
    # Bandwidth repair fixes unaffected cells exactly. A local pattern repair
    # may additionally release donor cells, subject to the footprint threshold.
    calls=[]
    if bad.mean()<=threshold:
        repair=solve_allocation(mu,sd,pred+reserve,mode='bandwidth',old_b=plan['bandwidth'],affected=bad)
        calls.append(repair)
        if repair['bandwidth'] is not None:
            repair['action']='bandwidth_repair'
        else:
            remaining=np.where(~bad)[0]
            extra=max(0,int(threshold*len(bad))-int(bad.sum()))
            slack=margin[remaining]
            affected=bad.copy()
            if extra: affected[remaining[np.argsort(slack)[-extra:]]]=True
            repair=solve_allocation(mu,sd,pred+reserve,mode='local',old_b=plan['bandwidth'],affected=affected)
            calls.append(repair); repair['action']='local_repair'
        if repair['bandwidth'] is not None:
            result=repair
        else: result=None
    else: result=None
    if result is None:
        result=solve_allocation(mu,sd,pred+reserve)
        calls.append(result); result['action']='full_fallback'
    # If problem itself is infeasible, use a nominal matching for execution but
    # explicitly report certificate failure rather than claim protection.
    if result['bandwidth'] is None:
        result=dict(result,bandwidth=plan['bandwidth'],action='infeasible_matching_fallback')
    for field in ['solver_s','solve_wall_s','solver_calls','repair_calls','full_calls']:
        result[field]=sum(x[field] for x in calls)
    result.update(matching_s=plan['matching_s'],score_s=plan['score_s'],certificate_s=plan['certificate_s'])
    result['accepted_certificate']=bool(service_margin(result['bandwidth'],mu,sd,pred+reserve).min()>=-2e-5)
    return result
