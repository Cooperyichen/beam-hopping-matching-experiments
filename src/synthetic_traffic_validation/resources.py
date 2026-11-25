"""Stage level wall/CPU/RSS and sampled whole-device GPU telemetry."""
import csv
import json
import os
import platform
import resource
import subprocess
import shutil
import threading
import time
from pathlib import Path

import numpy as np


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))


def write_csv(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)


class Meter:
    def __init__(self, root, stage, **labels):
        self.root = Path(root); self.stage = stage; self.labels = labels
        self.samples = []; self.stop = threading.Event()

    def sample(self):
        tool = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
        while not self.stop.is_set():
            try:
                raw = subprocess.check_output([tool, '--query-gpu=utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits'], timeout=3, text=True)
                util, mem, power = [float(v.strip()) for v in raw.splitlines()[0].split(',')]
                self.samples.append(dict(elapsed_s=time.perf_counter()-self.start, utilization_percent=util, device_memory_mib=mem, device_power_w=power))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop.wait(.25)

    def __enter__(self):
        import torch
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter(); self.cpu = time.process_time()
        self.thread = threading.Thread(target=self.sample, daemon=True); self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        import torch
        if self.cuda:
            torch.cuda.synchronize()
        end = time.perf_counter(); cpu = time.process_time()-self.cpu
        self.stop.set(); self.thread.join(timeout=4)
        row = dict(stage=self.stage, **self.labels, wall_s=end-self.start, cpu_s=cpu,
                   process_lifetime_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if platform.system() == 'Darwin' else 1024),
                   cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if self.cuda else 0,
                   cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if self.cuda else 0,
                   gpu_telemetry_samples=len(self.samples), completed=exc is None)
        if self.samples:
            row.update(gpu_mean_utilization_percent=float(np.mean([r['utilization_percent'] for r in self.samples])),
                       gpu_mean_power_w=float(np.mean([r['device_power_w'] for r in self.samples])),
                       gpu_peak_power_w=max(r['device_power_w'] for r in self.samples),
                       gpu_device_peak_memory_mib=max(r['device_memory_mib'] for r in self.samples))
            # Device energy includes other device users; no per-process attribution.
            tt = [r['elapsed_s'] for r in self.samples]; pp = [r['device_power_w'] for r in self.samples]
            row['sampled_device_energy_j'] = float(np.trapezoid(pp, tt)) if len(tt)>1 else None
        path = self.root/'resources'/f'{self.stage}.json'
        write_json(path,row)
        write_csv(path.with_suffix('.csv'),self.samples)


def environment(root):
    import torch, scipy, cvxpy, mosek
    write_json(Path(root)/'environment.json', dict(python=platform.python_version(),platform=platform.platform(),
        numpy=np.__version__, scipy=scipy.__version__, cvxpy=cvxpy.__version__, mosek=mosek.Env.getversion(),
        torch=torch.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        thread_environment={k:os.getenv(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}))
