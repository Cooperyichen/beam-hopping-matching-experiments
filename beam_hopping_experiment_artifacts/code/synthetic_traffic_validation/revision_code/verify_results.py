"""Post-run accounting assertions; never changes test decisions or configuration."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .revision import OUT, PARENT, SCENARIOS, HORIZONS
from ..resources import write_json, write_csv


def main():
    a=pd.read_csv(OUT/'static_results.csv'); b=pd.read_csv(OUT/'static_attempts.csv')
    budget=json.loads((OUT/'formal_budget.json').read_text())
    checks=[]
    def check(name,condition):
        checks.append(dict(test=name,passed=bool(condition)))
        assert condition,name
    keys=['stage','scenario','horizon','scale','origin','condition','policy']
    check('unique paired evaluation records',not a.duplicated(keys).any())
    expected_b=2*len(budget['levels'])*budget['load_windows']*6
    expected_c=sum(2*2*budget['horizon_windows'][str(h)]*6 for h in [24,48])
    expected_hybrid=2*3*2*budget['hybrid_windows']*2
    expected_ablation=4*budget['ablation_windows']*5
    check('all frozen formal records present',len(a)==expected_b+expected_c+expected_hybrid+expected_ablation)
    check('target begins immediately after last available observation',((a.last_available_observation+1)==a.origin).all())
    check('target dimensions match horizon',(a.target_end_exclusive==a.origin+a.horizon).all())
    check('test targets remain in physical partition',((a.origin>=800000)&(a.target_end_exclusive<=1000000)).all())
    check('coverage bounded',a.horizon_coverage.between(-1e-10,1+1e-10).all())
    check('useful plus unmet conserves demand',np.allclose(a.useful_mbit+a.unmet_mbit,a.demand_mbit,atol=1e-6))
    check('raw service is not less than useful service',(a.raw_mbit+1e-8>=a.useful_mbit).all())
    check('all optimizer attempts accounted',len(b)==int(a.repair_attempts.sum()+a.full_attempts.sum()))
    check('window latency ratio uses physical duration',np.allclose(a.latency_window_ratio,a.total_online_s/a.duration_s))
    phases=a.forecast_s+a.score_s+a.matching_s+a.check_s+a.all_optimizer_wall_s+a.trust_update_s
    check('latency includes every measured phase',(a.total_online_s+2e-8>=phases).all())
    hybrid=a[a.policy=='certified_static_matching']
    check('hybrid final branches exclusive and exhaustive',hybrid.final_path.isin(['matching_direct','repair_accepted','full_accepted','unprotected_fallback']).all())
    check('direct matching has no repair or full calls',((hybrid[hybrid.final_path=='matching_direct'][['repair_attempts','full_attempts']])==0).all().all())
    check('repair acceptance does not include full optimization',(hybrid.loc[hybrid.final_path=='repair_accepted','full_attempts']==0).all())
    check('full acceptance has a full call',(hybrid.loc[hybrid.final_path=='full_accepted','full_attempts']>0).all())
    pure=a[a.policy.isin(['throughput_matching','urgency_no_prediction','prediction_matching','reserve_matching','static_trust_matching'])]
    check('pure matching makes no protected reliability claim',not pure.conditional_margin_pass.any())
    check('unresolved status never called proved infeasible',not b.loc[b.status!='infeasible','classification'].eq('proved_infeasible').any())
    protected=a[a.conditional_margin_pass]
    check('certified margins satisfy tolerance',(protected.min_margin_mbit>=-2e-5).all())
    protocol=json.loads((OUT/'protocol.json').read_text())
    check('parent source unchanged',all(hashlib.sha256((PARENT/name).read_bytes()).hexdigest()==v for name,v in protocol['source_sha256'].items()))
    check('frozen formal decision code unchanged',hashlib.sha256((PARENT/'revision_code/revision.py').read_bytes()).hexdigest()==json.loads((OUT/'frozen_code.json').read_text())['sha256'])
    for s in SCENARIOS:
        for h in HORIZONS:
            f=np.load(OUT/'forecast'/s/str(h)/'forecast.npz')
            check(f'{s} {h} calibration and test separated',f['origins'][~f['test']].max()+h<=800000 and f['origins'][f['test']].min()>=800000)
    write_csv(OUT/'postrun_tests.csv',checks)
    write_json(OUT/'verification.json',dict(passed=len(checks),failed=0,formal_rows=len(a),optimizer_attempts=len(b),
        requirements_sha256='bb7b3f08f7dc75738fc2c83659c2367ab795bdb7420bac9f3dd82e3cd761d2ca',
        sequential_D_started=False))
    print(json.dumps(json.loads((OUT/'verification.json').read_text()),indent=2))


if __name__=='__main__': main()
