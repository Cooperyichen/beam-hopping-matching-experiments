"""Stationary continuous-time Pareto ON/OFF sources, exact slot integration."""
from pathlib import Path
import time

import numpy as np
import torch

from .resources import write_csv, write_json, Meter

REPORT_DEMAND_MBIT = np.array([3.53263106,3.95898587,3.17113991,3.86735415,2.67452076,3.76284609,3.43954699,2.94307239])


def make_config(steps=1000000, alpha=1.6, hotspot=False, seed=20261026):
    target = REPORT_DEMAND_MBIT.copy()
    if hotspot:
        target[:] = target.mean()*2/3
        target[:2] = REPORT_DEMAND_MBIT.mean()*2
    users = np.rint(target*1e6/(12*.002*1e6*.5)).astype(int)
    return dict(steps=steps,cells=8,slots=12,slot_duration_s=.002,total_bandwidth_hz=500e6,
                total_power_w=1000,max_active=3,max_bandwidth_fraction=.6,alpha=alpha,xmin_s=.1,
                rate_on_bps=1e6,seed=seed,users=users.tolist(),target_demand_mbit=target.tolist(),
                scenario=f'{"hotspot" if hotspot else "baseline"}_a{alpha:.1f}',stationary_start=True,
                channel_model='report spectral-efficiency fixture; power is fixed metadata, not optimized')


def pareto(shape, alpha, xmin, device, generator):
    u = torch.rand(shape,device=device,dtype=torch.float64,generator=generator).clamp_min(torch.finfo(torch.float64).tiny)
    return xmin*u.pow(-1/alpha)


def residual_life(shape, alpha, xmin, device, generator):
    u = torch.rand(shape,device=device,dtype=torch.float64,generator=generator)
    mean = alpha*xmin/(alpha-1)
    return torch.where(u <= (alpha-1)/alpha, u*mean,
                       xmin*(alpha*(1-u).clamp_min(1e-300)).pow(-1/(alpha-1)))


def bin_intervals(starts, ends, bins, dt):
    """Sum occupied seconds over all intervals; final bin is an overflow sink."""
    out = torch.zeros(bins+1,device=starts.device,dtype=torch.float64)
    diff = torch.zeros_like(out)
    s = starts.clamp(0,bins*dt); e = ends.clamp(0,bins*dt)
    keep = e>s; s=s[keep]; e=e[keep]
    i=(s/dt).long().clamp(max=bins-1); j=(e/dt).long().clamp(max=bins)
    same=i==j
    out.scatter_add_(0,i[same],(e-s)[same])
    q=~same; ii=i[q]; jj=j[q]
    out.scatter_add_(0,ii,(ii+1)*dt-s[q]); out.scatter_add_(0,jj,e[q]-jj*dt)
    diff.scatter_add_(0,ii+1,torch.full_like(s[q],dt)); diff.scatter_add_(0,jj,-torch.full_like(s[q],dt))
    return (out+diff.cumsum(0))[:bins]


def simulate_cell(users, steps, dt, alpha, xmin, rate, seed, device='cuda', batch=64):
    if users<1 or steps<1 or not 1<alpha<2 or min(dt,xmin,rate)<=0:
        raise ValueError('invalid source configuration')
    dev=torch.device(device); g=torch.Generator(device=dev).manual_seed(seed)
    duration=steps*dt
    occupied=torch.zeros(steps,device=dev,dtype=torch.float64)
    # At most a small event tensor; two million-bin occupancy buffers dominate.
    chunks=1024 if dev.type=='cpu' else 2048
    for lo in range(0,users,batch):
        size=min(batch,users-lo)
        clock=torch.zeros(size,device=dev,dtype=torch.float64)
        first=True
        while clock.numel():
            n=clock.numel()
            off=pareto((n,chunks),alpha,xmin,dev,g)
            on=pareto((n,chunks),alpha,xmin,dev,g)
            if first:
                start_on=torch.rand(n,device=dev,generator=g)<.5
                off[:,0]=torch.where(start_on,0.,residual_life((n,),alpha,xmin,dev,g))
                on[:,0]=torch.where(start_on,residual_life((n,),alpha,xmin,dev,g),on[:,0])
            edge=torch.stack((off,on),dim=-1).reshape(n,-1).cumsum(1)+clock[:,None]
            occupied.add_(bin_intervals(edge[:,0::2].reshape(-1),edge[:,1::2].reshape(-1),steps,dt))
            clock=edge[:,-1]; clock=clock[clock<duration]; first=False
    result=(occupied*rate).clamp_min(0).cpu().numpy()
    return result


def hurst_estimate(x, min_bins=250):
    x=np.asarray(x,dtype=float); x=x-x.mean(); scales=[]; variances=[]
    k=min_bins
    while len(x)//k>=40:
        y=x[:len(x)//k*k].reshape(-1,k).sum(1)
        if y.var()>0:
            scales.append(k); variances.append(y.var())
        k=max(k+1,int(k*1.5))
    return float(np.polyfit(np.log(scales),np.log(variances),1)[0]/2) if len(scales)>=3 else None


def generate(root, steps=1000000, smoke=False):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    configs=[make_config(steps=steps,alpha=a,hotspot=h) for a in ([1.6] if smoke else [1.2,1.6,1.9]) for h in ([False] if smoke else [False,True])]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    diagnostics=[]
    for cfg in configs:
        out=root/'data'/cfg['scenario']; out.mkdir(parents=True,exist_ok=True)
        file=out/'traffic.npz'
        if file.exists():
            raise FileExistsError(f'{file} exists; use a new output directory to preserve test provenance')
        rows=[]; arrays=[]
        with Meter(root,'generate_'+cfg['scenario'],backend=device):
            for i,users in enumerate(cfg['users']):
                seed=cfg['seed']+i; batch=64
                if device=='cuda':
                    free,_=torch.cuda.mem_get_info(); budget=min(1.5*2**30,.5*free)
                    batch=max(1,min(64,int((budget-3*steps*8)/(2048*2*8*8))))
                start=time.perf_counter()
                while True:
                    try:
                        a=simulate_cell(users,steps,cfg['slot_duration_s'],cfg['alpha'],cfg['xmin_s'],cfg['rate_on_bps'],seed,device,batch)
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if batch==1:
                            raise
                        batch=max(1,batch//2)
                elapsed=time.perf_counter()-start; arrays.append(a)
                theory=users*cfg['rate_on_bps']*.5*cfg['slot_duration_s']
                rows.append(dict(scenario=cfg['scenario'],cell=i,users=users,seed=seed,backend=device,batch=batch,
                                 steps=steps,wall_s=elapsed,samples_per_s=steps/elapsed,
                                 empirical_bits_per_slot=float(a.mean()),theoretical_bits_per_slot=theory,
                                 relative_mean_error=float(a.mean()/theory-1),std_bits=float(a.std()),
                                 hurst_estimate=hurst_estimate(a),hurst_limit=(3-cfg['alpha'])/2))
                print('generated',cfg['scenario'],i,round(elapsed,2),flush=True)
        start=time.perf_counter(); data=np.stack(arrays)
        # Lossless storage preserves exact integrated volumes.
        np.savez_compressed(file,traffic_bits=data)
        io=time.perf_counter()-start; cfg.update(backend=device,write_s=io,file_bytes=file.stat().st_size)
        write_json(out/'config.json',cfg); write_csv(out/'diagnostics.csv',rows)
        short=data[:,int(.8*steps):int(.8*steps)+12]
        write_csv(out/'example_cycle.csv',[dict(slot=t,**{f'cell_{i}_bits':short[i,t] for i in range(8)}) for t in range(12)])
        diagnostics.extend(rows)
    write_csv(root/'generation_diagnostics.csv',diagnostics)
