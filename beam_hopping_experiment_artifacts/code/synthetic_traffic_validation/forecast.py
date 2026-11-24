"""Whittle ARFIMA(p,d,0), finite-history direct cumulative predictors."""
import json
from pathlib import Path
import time

import numpy as np
from scipy import linalg, optimize, signal

from .resources import Meter, write_csv, write_json


def split_bounds(n):
    return int(.6*n),int(.7*n),int(.8*n)


def fractional_weights(d, n):
    w=np.ones(n)
    for j in range(1,n):
        w[j]=w[j-1]*(j-1-d)/j
    return w


def pacf_to_ar(k):
    a=np.empty(0)
    for v in k:
        a=np.r_[a-v*a[::-1],v]
    return a


def fit_arfima(train,p,d_fixed=None):
    """Profile Whittle likelihood on log-spaced averaged periodogram bins.

    The first/DC frequency is omitted. Parameter selection uses training only.
    Spectrum bin counts weight the likelihood, not the number of bins.
    """
    y=np.asarray(train,dtype=float); mean=float(y.mean()); scale=max(float(y.std()),1e-12)
    z=(y-mean)/scale; n=len(z)
    period=(abs(np.fft.rfft(z))**2/n)[1:]
    freq=2*np.pi*np.fft.rfftfreq(n)[1:]
    # Retain low-frequency resolution while reducing repeated likelihood cost.
    edges=np.unique(np.r_[0,np.geomspace(1,len(freq),min(8192,len(freq))).astype(int),len(freq)])
    count=np.diff(edges).astype(float)
    f=np.add.reduceat(freq,edges[:-1])/count
    I=np.add.reduceat(period,edges[:-1])/count
    def unpack(v):
        return (float(v[0]),pacf_to_ar(v[1:])) if d_fixed is None else (float(d_fixed),pacf_to_ar(v))
    def loss(v):
        d,phi=unpack(v)
        den=abs(1-sum(a*np.exp(-1j*f*(j+1)) for j,a in enumerate(phi)))**2
        spec=np.maximum((2*np.sin(f/2))**(-2*d)/den,1e-20)
        innovation=np.sum(count*I/spec)/count.sum()
        return float(np.log(max(innovation,1e-30))+np.sum(count*np.log(spec))/count.sum())
    bounds=([(-.45,.49)] if d_fixed is None else [])+[(-.995,.995)]*p
    starts=[np.r_[d,np.zeros(p)] for d in [0.,.25,.45]] if d_fixed is None else [np.zeros(p)]
    fits=[optimize.minimize(loss,v,bounds=bounds,method='L-BFGS-B',options={'maxiter':250,'ftol':1e-10}) for v in starts]
    best=min(fits,key=lambda x:x.fun); d,phi=unpack(best.x)
    return dict(p=p,d=d,phi=phi.tolist(),mean=mean,scale=scale,objective=float(best.fun),
                success=bool(best.success),message=str(best.message),training_samples=n,
                method='profile Whittle with weighted log-frequency bins')


def step_weights(fit, history, horizon):
    # Numerical spectral integration of the fitted stationary ARFIMA covariance.
    # DC uses the midpoint of its integration bin; no empirical test covariance.
    size=2**18; f=2*np.pi*np.fft.rfftfreq(size); f[0]=np.pi/size
    den=abs(1-sum(a*np.exp(-1j*f*(j+1)) for j,a in enumerate(fit['phi'])))**2
    spec=(2*np.sin(f/2))**(-2*fit['d'])/den
    cov=np.fft.irfft(spec,n=size)[:history+horizon+1]; cov/=cov[0]
    mat=linalg.toeplitz(cov[:history])+1e-8*np.eye(history)
    rhs=np.stack([cov[j:j+history] for j in range(1,horizon+1)],axis=1)
    return linalg.cho_solve(linalg.cho_factor(mat,check_finite=False),rhs,check_finite=False).T


def prediction_weights(fit, history, horizon):
    return step_weights(fit,history,horizon).sum(axis=0)


def predict_at(y,origins,weights,mean,horizon):
    origins=np.asarray(origins,dtype=int); L=len(weights)
    if origins.size and (origins.min()<L or origins.max()>len(y)):
        raise ValueError('origins require a complete strictly past history')
    result=[]
    for start in range(0,len(origins),2048):
        idx=origins[start:start+2048,None]-1-np.arange(L)[None,:]
        result.append((y[idx]-mean)@weights+horizon*mean)
    return np.concatenate(result) if result else np.empty(0)


def quantile_reserve(residual,alpha=.025):
    n=len(residual); rank=int(np.ceil((n+1)*(1-alpha)))
    if rank>n:
        return np.full(residual.shape[1:],np.inf)
    return np.maximum(np.sort(residual,axis=0)[rank-1],0.)


def errors(pred,true):
    e=pred-true; mse=float(np.mean(e*e))
    return dict(mae_mbit=float(np.mean(abs(e))),rmse_mbit=np.sqrt(mse),
                nrmse=float(np.sqrt(mse)/max(np.std(true),1e-12)),bias_mbit=float(e.mean()),
                underprediction_probability=float(np.mean(e<0)))


def fit_forecast(root):
    root=Path(root); metrics=[]; selections=[]
    for folder in sorted((root/'data').iterdir()):
        if not (folder/'traffic.npz').exists(): continue
        cfg=json.loads((folder/'config.json').read_text())
        y=np.load(folder/'traffic.npz')['traffic_bits']/1e6; n=y.shape[1]; H=cfg['slots']
        train_end,select_end,test_start=split_bounds(n)
        if train_end<2048: raise ValueError('at least 4000 slots required for forecast pipeline')
        select_orig=np.arange(train_end,select_end-H+1,H*4)
        origins=np.arange(select_end,n-H+1,H)
        calibration=origins+H<=test_start; test=origins>=test_start
        if calibration.sum()<40: raise ValueError('not enough residual calibration cycles')
        cum=np.c_[np.zeros(8),y.cumsum(1)]
        true=(cum[:,origins+H]-cum[:,origins]).T
        select_true=(cum[:,select_orig+H]-cum[:,select_orig]).T
        predictions={name:np.zeros_like(true) for name in ['mean','persistence','ar','farima']}
        trajectory=np.zeros((len(origins),8,H)); models=[]; one_step_scale=[]
        out=root/'forecast'/cfg['scenario']; out.mkdir(parents=True,exist_ok=True)
        with Meter(root,'forecast_'+cfg['scenario']):
            for cell in range(8):
                a=y[cell]; best=None; start=time.perf_counter()
                fits={p:fit_arfima(a[:train_end],p) for p in [0,1,2]}
                for p,fit in fits.items():
                    for history in [128,256,512]:
                        w=prediction_weights(fit,history,H)
                        pred=np.maximum(predict_at(a,select_orig,w,fit['mean'],H),0)
                        mse=float(np.mean((pred-select_true[:,cell])**2))
                        selections.append(dict(scenario=cfg['scenario'],cell=cell,p=p,history=history,selection_mse=mse,d=fit['d']))
                        if best is None or mse<best[0]: best=(mse,fit,history)
                _,fit,L=best; W=step_weights(fit,L,H)
                # Per-step projections remain diagnostic; cumulative predictor is direct.
                predictions['farima'][:,cell]=np.maximum(predict_at(a,origins,W.sum(0),fit['mean'],H),0)
                for h in range(H):
                    trajectory[:,cell,h]=np.maximum(predict_at(a,origins,W[h],fit['mean'],1),0)
                predictions['mean'][:,cell]=H*a[:train_end].mean()
                predictions['persistence'][:,cell]=H*a[origins-1]
                ar=fit_arfima(a[:train_end],2,d_fixed=0.)
                predictions['ar'][:,cell]=np.maximum(predict_at(a,origins,prediction_weights(ar,L,H),ar['mean'],H),0)
                models.append(dict(cell=cell,history=L,fit=fit,ar_fit=ar,fit_and_selection_wall_s=time.perf_counter()-start))
                # Scale from past training observations only.
                one_step_scale.append(max(float(np.std(np.diff(a[:train_end]))),1e-6))
                print('fitted',cfg['scenario'],cell,fit['p'],L,round(fit['d'],3),flush=True)
        residual=true[calibration]-predictions['farima'][calibration]
        reserve=quantile_reserve(residual)
        for model,pred in predictions.items():
            for cell in range(8):
                for partition,mask in [('calibration',calibration),('test',test)]:
                    metrics.append(dict(scenario=cfg['scenario'],model=model,cell=cell,partition=partition,
                                        **errors(pred[mask,cell],true[mask,cell])))
        np.savez_compressed(out/'predictions.npz',origins=origins,truth=true,calibration=calibration,test=test,
                            reserve=reserve,trajectory=trajectory,one_step_scale=one_step_scale,**predictions)
        write_json(out/'models.json',models)
        write_json(out/'split.json',dict(training=[0,train_end],selection=[train_end,select_end],
                    residual_calibration=[select_end,test_start],test=[test_start,n],
                    target='sum of next 12 slots in Mbit',origins_are_first_target_slot=True))
        write_csv(out/'reserve.csv',[dict(cell=i,reserve_mbit=reserve[i],training_mean_cycle_mbit=models[i]['fit']['mean']*H,
                   reserve_percent_of_mean=100*reserve[i]/(models[i]['fit']['mean']*H),
                   calibration_exceedance=float(np.mean(residual[:,i]>reserve[i])),
                   test_exceedance=float(np.mean(true[test,i]-predictions['farima'][test,i]>reserve[i]))) for i in range(8)])
    write_csv(root/'forecast_metrics.csv',metrics); write_csv(root/'model_selection.csv',selections)
