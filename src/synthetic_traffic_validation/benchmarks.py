"""Resource benchmarks and online forecast latency, without parameter tuning."""
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
from scipy import linalg

from .traffic import simulate_cell
from .forecast import step_weights
from .resources import Meter,write_csv,write_json


def run(root):
    root=Path(root); timings=[]
    torch.set_num_threads(2)
    # Warm CUDA before timing; context startup is reported separately in stage logs.
    if torch.cuda.is_available():
        torch.ones(1,device='cuda').sum().item()
    for users,steps in [(128,200000),(300,1000000)]:
        for rep in range(3):
            for device in (['cpu','cuda'] if rep%2==0 else ['cuda','cpu']):
                if device=='cuda' and not torch.cuda.is_available(): continue
                with Meter(root,f'benchmark_generation_{device}_{steps}_{rep}',users=users,steps=steps):
                    start=time.perf_counter(); cpu=time.process_time()
                    a=simulate_cell(users,steps,.002,1.6,.1,1e6,107+rep,device=device,batch=64)
                    elapsed=time.perf_counter()-start
                    timings.append(dict(device=device,replicate=rep,users=users,steps=steps,wall_s=elapsed,
                                         cpu_s=time.process_time()-cpu,mean_bits=float(a.mean()),std_bits=float(a.std())))
    write_csv(root/'generation_backend_benchmark.csv',timings)
    rows=[]
    for folder in sorted((root/'forecast').iterdir()):
        if not (folder/'models.json').exists(): continue
        models=json.loads((folder/'models.json').read_text()); f=dict(np.load(folder/'predictions.npz'))
        y=np.load(root/'data'/folder.name/'traffic.npz')['traffic_bits']/1e6
        ids=np.flatnonzero(f['test'])[:1000]; weights=[]
        for model in models:
            weights.append(step_weights(model['fit'],model['history'],12).sum(0))
        for j in ids:
            t=int(f['origins'][j]); start=time.perf_counter(); cpu=time.process_time(); values=[]
            for i,model in enumerate(models):
                L=model['history']; mean=model['fit']['mean']
                values.append(max(0.,float((y[i,t-L:t][::-1]-mean)@weights[i]+12*mean)))
            elapsed=time.perf_counter()-start
            np.testing.assert_allclose(values,f['farima'][j],rtol=1e-9,atol=1e-8)
            rows.append(dict(scenario=folder.name,origin=t,wall_s=elapsed,cpu_s=time.process_time()-cpu,
                             process_lifetime_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    write_csv(root/'online_forecast_latency.csv',rows)
