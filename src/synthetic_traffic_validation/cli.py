"""Reproducible experiment entry points. Run with --help for stage controls."""
import argparse
from pathlib import Path
import torch

from .resources import environment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['generate','fit-forecast','run-matching','run-chance-reference','run-operational','summarize','all'])
    p.add_argument('--output',type=Path,default=Path('runs/formal'))
    p.add_argument('--steps',type=int,default=1000000)
    p.add_argument('--cycles',type=int,default=24)
    p.add_argument('--draws',type=int,default=512)
    p.add_argument('--operational-cycles',type=int,default=120)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); torch.set_num_threads(2)
    a.output.mkdir(parents=True,exist_ok=True); environment(a.output)
    if a.command in ['generate','all']:
        from .traffic import generate
        generate(a.output,a.steps,a.smoke)
    if a.command in ['fit-forecast','all']:
        from .forecast import fit_forecast
        fit_forecast(a.output)
    if a.command in ['run-matching','run-chance-reference','all']:
        from .experiments import compare
        for command in (['run-matching','run-chance-reference'] if a.command=='all' else [a.command]):
            compare(a.output,a.cycles,a.draws,command)
    if a.command in ['run-operational','all']:
        from .operational import run_operational
        run_operational(a.output,a.operational_cycles,a.draws)
    if a.command in ['summarize','all']:
        from .benchmarks import run
        run(a.output)
        from .reporting import summarize
        summarize(a.output)


if __name__=='__main__': main()
