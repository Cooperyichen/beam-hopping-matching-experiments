"""Standalone figures and a source-linked, evidence-based report in English."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .resources import write_csv,write_json

COLORS=['#245b8a','#d98c24','#2a927d','#87599e','#c75252','#525c68','#6a9e3e','#c176a0','#477f87',
        '#17becf','#393b79','#637939','#8c6d31']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def savefig(root,name):
    ax=plt.gca(); ax.grid(alpha=.18,axis='y')
    plt.tight_layout()
    for ext in ['png','pdf']: plt.savefig(root/f'{name}.{ext}',dpi=180,bbox_inches='tight',pad_inches=.20)
    plt.close()


def block_interval(values,block=4,draws=1000):
    """Moving-block bootstrap across time-ordered cycle clusters.

    Sensitivity estimate only: long memory can exceed the chosen block length.
    """
    v=np.asarray(values); n=len(v)
    if n<2: return [float(v.mean()),float(v.mean())]
    rng=np.random.default_rng(93); start=rng.integers(0,n,(draws,int(np.ceil(n/block))))
    idx=(start[:,:,None]+np.arange(block))%n
    means=v[idx.reshape(draws,-1)[:,:n]].mean(1)
    return np.quantile(means,[.025,.975]).tolist()


def summarize(root):
    root=Path(root); figures=root/'figures'; figures.mkdir(parents=True,exist_ok=True)
    parts=[]; latency=[]
    for name in ['run-matching','run-chance-reference']:
        p=root/(name+'_results.csv'); q=root/(name+'_latency.csv')
        if p.exists(): parts.append(pd.read_csv(p)); latency.append(pd.read_csv(q))
    data=pd.concat(parts,ignore_index=True); times=pd.concat(latency,ignore_index=True)
    data.to_csv(root/'allocation_results.csv',index=False); times.to_csv(root/'allocation_latency.csv',index=False)
    summary=data.groupby(['scenario','condition','policy'],sort=False).agg(
        raw_mbit=('raw_mbit','mean'),useful_mbit=('useful_mbit','mean'),unmet_mbit=('unmet_mbit','mean'),
        coverage=('horizon_coverage','mean'),mean_cell_violation=('average_cell_violation','mean'),
        max_cycle_cell_violation=('max_cell_violation','mean'),any_cell_violation=('any_cell_violation','mean'),
        nominal_feasible_fraction=('optimization_feasible','mean'),certificate_fraction=('accepted_certificate','mean'),
        nmae=('nmae','mean'),cycles=('replicate','count')).reset_index()
    summary.to_csv(root/'allocation_summary.csv',index=False)
    cellcols=[f'cell_{i}_violation' for i in range(8)]
    cellmeans=data.groupby(['scenario','condition','policy'])[cellcols].mean()
    cellmeans['maximum_cell_violation']=cellmeans.max(axis=1)
    cellmeans.to_csv(root/'per_cell_violation_summary.csv')
    perf=times.groupby(['scenario','condition','policy']).agg(mean_wall_s=('wall_s','mean'),median_wall_s=('wall_s','median'),
                  p95_wall_s=('wall_s',lambda x:x.quantile(.95)),p99_wall_s=('wall_s',lambda x:x.quantile(.99)),
                  total_solver_s=('solver_s','sum'),total_solver_calls=('solver_calls','sum'),
                  total_repair_calls=('repair_calls','sum'),total_full_calls=('full_calls','sum')).reset_index()
    perf.to_csv(root/'latency_summary.csv',index=False)
    confidence=[]
    for keys,group in data.groupby(['scenario','condition','policy']):
        for block in [2,4,8]:
            lo,hi=block_interval(group.sort_values('origin')['average_cell_violation'],block)
            confidence.append(dict(zip(['scenario','condition','policy'],keys))|dict(block_cycles=block,lower=lo,upper=hi))
    write_csv(root/'violation_block_bootstrap.csv',confidence)
    natural=data[data.condition=='natural_combined'].copy()
    natural['forecast_error_bin']=natural.groupby('scenario')['nmae'].transform(
        lambda x:pd.qcut(x,4,labels=False,duplicates='drop'))
    natural.groupby(['scenario','policy','forecast_error_bin']).agg(
        mean_nmae=('nmae','mean'),mean_violation=('average_cell_violation','mean'),
        cycles=('replicate','count')).to_csv(root/'natural_error_quantiles.csv')
    scenarios=list(summary.scenario.unique())
    labels={p:p.replace('_',' ') for p in summary.policy.unique()}
    for scenario in scenarios:
        a=summary[(summary.scenario==scenario)&(summary.condition=='natural_combined')]
        l=perf[(perf.scenario==scenario)&(perf.condition=='natural_combined')]
        merged=a.merge(l,on=['scenario','condition','policy'])
        groups={
            'matching':['throughput_matching','urgency_no_prediction','prediction_matching','reserve_matching','trust_matching'],
            'reference':['prediction_matching','certified_matching','nominal_horizon','chance_horizon','chance_equal_bandwidth'],
            'reserve':['prediction_matching','fixed_reserve_1pct','fixed_reserve_2pct','fixed_reserve_3pct','fixed_reserve_5pct','reserve_matching']}
        for group,policies in groups.items():
            g=merged[merged.policy.isin(policies)]
            if g.empty: continue
            for field,ylabel,kind in [('mean_cell_violation','Mean cell service violation probability','violation'),
                                    ('coverage','Useful traffic / aggregate demand','coverage'),
                                    ('mean_wall_s','Mean online allocation wall time (ms)','overhead')]:
                plt.figure(figsize=(7.2,4.2)); x=np.arange(len(g)); scale=1000 if kind=='overhead' else 1
                plt.bar(x,g[field]*scale,color=[COLORS[list(labels).index(p)%len(COLORS)] for p in g.policy])
                plt.xticks(x,[labels[p] for p in g.policy],rotation=35,ha='right'); plt.ylabel(ylabel)
                if kind=='overhead': plt.yscale('log')
                if kind=='violation':
                    plt.axhline(.05,color='black',ls='--',lw=1,label='Per-cell target 0.05')
                    plt.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.18))
                savefig(figures,f'{kind}_{group}_{scenario}')
        core=merged[~merged.policy.str.startswith('fixed_reserve_')]
        for value,caption in [('coverage','Useful traffic / aggregate demand'),
                              ('mean_cell_violation','Mean cell service violation probability')]:
            plt.figure(figsize=(7.0,4.0))
            for _,row in core.iterrows():
                plt.scatter(row.mean_wall_s*1000,row[value],s=45,
                            color=COLORS[list(labels).index(row.policy)%len(COLORS)],label=labels[row.policy])
            plt.xscale('log'); plt.xlabel('Mean online allocation wall time (ms)'); plt.ylabel(caption)
            plt.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.60),ncol=2,fontsize=8)
            savefig(figures,f'tradeoff_{value}_{scenario}')
        if scenario.endswith('1.6'):
            error=summary[(summary.scenario==scenario)&summary.condition.isin(['perfect_demand','error_half','natural_combined','error_1_5','error_double'])]
            plt.figure(figsize=(7.0,4.3))
            for i,(policy,g) in enumerate(error.groupby('policy',sort=False)):
                if policy not in ['prediction_matching','reserve_matching','trust_matching','certified_matching','chance_horizon']: continue
                g=g.sort_values('nmae'); plt.plot(g.nmae,g.mean_cell_violation,'o-',label=labels[policy],color=COLORS[list(labels).index(policy)%len(COLORS)])
            plt.axhline(.05,color='black',ls='--',lw=1)
            plt.xlabel('Aggregate demand forecast NMAE'); plt.ylabel('Mean cell service violation probability')
            plt.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.36),ncol=2)
            savefig(figures,'error_dependence_'+scenario)
    op_path=root/'operational_results.csv'
    if op_path.exists():
        op=pd.read_csv(op_path)
        ops=op.groupby(['scenario','policy']).agg(coverage=('horizon_coverage','mean'),violation=('average_cell_violation','mean'),
                total_wall_s=('wall_s','sum'),total_solver_s=('solver_s','sum'),full_calls=('full_calls','sum'),
                repair_calls=('repair_calls','sum'),certificate_fraction=('accepted_certificate','mean'),
                schedule_changes=('schedule_changes','sum'),subcycle_safe_fraction=('safe_interval_below_cycle','mean')).reset_index()
        ops.to_csv(root/'operational_summary.csv',index=False)
        for scenario,g in ops.groupby('scenario'):
            plt.figure(figsize=(7.2,4.0))
            plt.bar(np.arange(len(g)),g.total_wall_s,color=COLORS[:len(g)])
            plt.xticks(np.arange(len(g)),g.policy.str.replace('_',' '),rotation=35,ha='right'); plt.ylabel('Cumulative allocation wall time (s)')
            savefig(figures,'operational_overhead_'+scenario)
    # Forecast/generation figures, one scientific relationship per figure.
    fm=pd.read_csv(root/'forecast_metrics.csv'); fs=fm[fm.partition=='test'].groupby(['scenario','model']).rmse_mbit.mean().unstack()
    fs.to_csv(root/'forecast_summary.csv')
    fs.plot.bar(figsize=(8,4.3),color=COLORS[:4]); plt.ylabel('Mean cell cycle-demand RMSE (Mbit)'); plt.xlabel('')
    plt.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.2),ncol=4)
    savefig(figures,'forecast_error')
    generation=pd.read_csv(root/'generation_diagnostics.csv')
    for kind in ['baseline','hotspot']:
        g=generation[generation.scenario.str.startswith(kind)].copy()
        g['alpha']=g.scenario.str.extract(r'a([0-9.]+)$')[0].astype(float)
        means=g.groupby('alpha').hurst_estimate.mean()
        plt.figure(figsize=(6.3,3.7)); plt.plot(means.index,means.values,'o-',label='Mean estimated H')
        plt.plot(means.index,(3-means.index)/2,'s--',label='Asymptotic H limit')
        plt.xlabel('Pareto duration tail index'); plt.ylabel('Hurst parameter'); plt.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.22),ncol=2)
        savefig(figures,'hurst_'+kind)
    traffic=np.load(root/'data'/'baseline_a1.6'/'traffic.npz')['traffic_bits']
    for cell in [0,4]:
        plt.figure(figsize=(7.0,3.5)); start=800000
        plt.plot(np.arange(2400)*.002,traffic[cell,start:start+2400]/.002/1e6,lw=.8)
        plt.xlabel('Elapsed time in test segment (s)'); plt.ylabel('Cell offered traffic (Mbit/s)')
        savefig(figures,f'traffic_cell_{cell}')
    benchmark=root/'generation_backend_benchmark.csv'
    if benchmark.exists():
        # Remove only our superseded single-workload figure from this run.
        for ext in ['png','pdf']:
            (figures/f'generation_cpu_gpu.{ext}').unlink(missing_ok=True)
        for steps,g in pd.read_csv(benchmark).groupby('steps'):
            g=g.groupby('device').wall_s.mean()
            plt.figure(figsize=(5.4,3.4)); plt.bar(g.index,g.values,color=COLORS[:len(g)])
            plt.ylabel(f'Mean time to generate {steps:,} slots (s)')
            savefig(figures,f'generation_cpu_gpu_{steps}')
    generate_report(root,summary,perf,fs)


def generate_report(root,summary,perf,forecast_summary):
    diagnostics=pd.read_csv(root/'generation_diagnostics.csv')
    params=json.loads((root/'data'/'baseline_a1.6'/'config.json').read_text())
    resources=[]
    for p in (root/'resources').glob('*.json'):
        resources.append(json.loads(p.read_text()))
    write_csv(root/'resource_summary.csv',resources)
    base=summary[(summary.scenario=='baseline_a1.6')&(summary.condition=='natural_combined')]
    cost=perf[(perf.scenario=='baseline_a1.6')&(perf.condition=='natural_combined')]
    table=base.merge(cost,on=['scenario','condition','policy'])
    worst=pd.read_csv(root/'per_cell_violation_summary.csv')[['scenario','condition','policy','maximum_cell_violation']]
    table=table.merge(worst,on=['scenario','condition','policy'])
    lines=['# Synthetic Traffic, Forecasting and Allocation Validation','',
      '## Experimental interpretation','',
      'Each cell aggregates independent stationary Pareto ON/OFF users. A slot is 2 ms and a cycle contains 12 slots. '+
      'The optimizer receives one demand volume per cell and cycle. There is no packet expiration, arrival queue or slot demand constraint. '+
      'The six datasets each contain eight cells and one million slots per cell. Spatial hotspots change source counts while preserving total theoretical mean load. '+
      'Spectral efficiency follows the report fixture, with smooth cycle-level drift. Total power is fixed metadata at 1000 W. '+
      'These experiments do not derive physical link gains from antenna patterns or propagation measurements.','',
      'Traffic is generated by exact integration of continuous ON intervals, including fractional first and last slots. '+
      'Stationary initialization samples the state and equilibrium residual life. The limit Hurst relation is '+r'$H=(3-\alpha)/2$'+
      '; estimated Hurst values are finite-scale diagnostics rather than exact finite-sample identities.','',
      '## Forecast protocol','',
      'The first 60% of slots fit model parameters. Slots from 60% to 70% select order and history. '+
      'Slots from 70% to 80% calibrate residual reserves. The final 20% evaluate registered comparisons. '+
      'ARFIMA parameters use a profile Whittle objective with weighted frequency bins. Finite-history prediction filters use numerical integration of the fitted spectrum. '+
      'Negative demand forecasts are clipped to zero. Model fit status and parameters are saved in the model manifests.','',
      'The next-cycle forecast is formed strictly from past observations. Prediction trust for the next cycle uses only errors of completed observations. '+
      'Controlled error scaling uses future truth only as an offline counterfactual intervention, never as deployable information. '+
      'These interventions represent isolated forecast shocks with trust inherited from native past predictions. They do not test sustained forecaster distribution shifts.','',
      '## Allocation comparisons','',
      'All matching policies solve an exact rectangular assignment in each slot with three identical illumination opportunities. '+
      'Each active pair receives one third of total bandwidth. In this report-compatible setting the assignment reduces to selecting three cells. '+
      'The general rectangular assignment implementation also handles beam-dependent scores, but beam-dependent channels are not claimed as validated in this study.','',
      'Throughput matching uses spectral efficiency alone. Urgency matching uses training mean demand. Prediction matching uses the FARIMA forecast. '+
      'Reserve matching adds a calibrated upper residual quantile. Trust matching interpolates between forecast and reserved target. '+
      'Certified matching checks the full fixed-plan margin, tries restricted bandwidth or local pattern repair when applicable, and otherwise invokes full optimization.','',
      'Nominal horizon optimization retains deterministic predicted-demand constraints and variable bandwidth. '+
      'The complete chance horizon optimization adds channel risk allowance and empirical demand reserve. '+
      'A complete chance model with equal bandwidth provides a controlled comparison that isolates scheduling from bandwidth flexibility. '+
      'Pure matching contains no cycle demand constraint. Its empirical violations reflect both the urgency scoring rule and prediction errors.','',
      'Normalizing urgency by the mean urgency cancels the common remaining-slot factor. '+
      'The implemented score therefore prioritizes relative residual demand; it does not impose an independent deadline-pressure guarantee.','',
      '| Policy | Useful Mbit | Coverage | Mean cell violation | Worst cell violation | Mean allocation ms | Feasible fraction |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for _,r in table.iterrows():
        lines.append(f"| {r.policy.replace('_',' ')} | {r.useful_mbit:.3f} | {r.coverage:.4f} | {r.mean_cell_violation:.4f} | {r.maximum_cell_violation:.4f} | {1000*r.mean_wall_s:.3f} | {r.nominal_feasible_fraction:.3f} |")
    indexed=table.set_index('policy')
    complete=indexed.loc['chance_horizon']; match=indexed.loc['reserve_matching']
    lines += ['',f"The chance horizon allocation reaches {100*complete.coverage:.3f}% demand coverage with mean cell violation {100*complete.mean_cell_violation:.3f}%. "+
              f"Reserve matching reaches {100*match.coverage:.3f}% coverage with violation {100*match.mean_cell_violation:.3f}%. "+
              f"Their mean allocation times are {1000*complete.mean_wall_s:.3f} ms and {1000*match.mean_wall_s:.3f} ms, respectively. "+
              f"The worst-cell violation estimates are {100*complete.maximum_cell_violation:.2f}% and {100*match.maximum_cell_violation:.2f}%. "+
              'A mean below 5% can therefore conceal an individual cell that exceeds the reliability target.','']
    lines += ['', 'The table is the baseline alpha 1.6 scenario with natural forecast errors and channel uncertainty. '+
              'Raw traffic is offered transmission capacity; useful traffic caps service at the realized cell demand. '+
              'Every competing method uses common traffic cycles and Gaussian channel draws. Per-cycle conditional Gaussian violation is also computed analytically.','',
      'The minimum-cell coverage field averages the worst cell coverage in each realization. '+
      'The per-cell violation summary first averages each cell over test cycles and then takes the maximum across cells; '+
      'it differs from averaging the maximum cell violation of each cycle. Both definitions are preserved explicitly in output tables.','',
      '## Reliability and statistical limits','',
      'The channel margin is exact for a fixed plan under the stated Gaussian spectral-efficiency error model and independent slot errors. '+
      'Demand reserves use the one-sided finite-sample empirical residual rank with nominal tail probability 0.025. '+
      'Pareto ON/OFF data have temporal dependence, so the exchangeability assumption needed for a distribution-free conformal guarantee is not established. '+
      'A passed certificate therefore establishes a channel-model condition for an empirically calibrated demand target. '+
      'The complete end-to-end 5% service probability remains an empirical target in these experiments. '+
      'Union-bound risk splitting alone does not cure a miscalibrated demand quantile.','',
      'Moving-block bootstrap intervals with blocks of 2, 4 and 8 evaluation cycles are sensitivity estimates. '+
      'The snapshots come from one trajectory per scenario and are not independent training repetitions. '+
      'Changing alpha uses paired random seeds; cells use independent source streams.','',
      'Solver infeasibility is reported explicitly. For execution metrics, an infeasible optimizer falls back to prediction matching; '+
      'that fallback receives no reliability certificate. Optimal status, time-limited feasible incumbents and failures remain distinguishable in raw CSV.','',
      '## Operational decisions','',
      'The sequential experiment reuses a complete 12-slot plan template at successive cycle boundaries. '+
      'It compares recomputation every cycle, a four-cycle fixed period, margin triggering, margin plus relaxation-gap triggering, '+
      'adaptive assessment intervals and certified matching. A separate 3% operating reserve creates slack above the calibrated target. '+
      'This is a cycle-boundary controller validation; within-cycle observation-dependent chance constraints require a separate risk analysis.','',
      'Bandwidth repair fixes future illumination and unaffected bandwidth. Local pattern repair releases affected cells and permitted donor cells, '+
      'while keeping all other variables fixed. All remaining demand and payload constraints are imposed. '+
      'Infeasible repairs trigger full optimization. No decisions from completed cycles are modified.','',
      'In the sequential replay, trust updates once per completed cycle using absolute cumulative forecast error divided by the empirical reserve. '+
      'Its forgetting factor is the selected slot factor raised to the twelfth power. '+
      'The snapshot matching experiment instead updates trust from per-slot trajectory errors divided by the training increment scale. '+
      'Both update rules use completed observations only. These two normalizations should be compared explicitly before choosing an operational trust controller.','',
      r'The deterioration estimate $L_i$ is the 99th percentile of fixed-plan margin decrease per second, inflated by 20%. '+
      'Calibration uses a fixed candidate plan at each adjacent pair of epochs. Test exceedance rates are saved separately. '+
      'Assessment intervals are rounded down to the cycle grid. A computed interval below one cycle is flagged rather than described as a proved safe interval.','',
      '## Computation and resource accounting','',
      'GPU generation, compression and forecast fitting are reported separately. Online forecast latency is benchmarked with actual filter evaluation, '+
      'rather than indexing a precomputed prediction array. Policy tables report allocation time; add measured forecast latency for total online latency. '+
      'Peak RSS is the high-water mark of the whole process, not an isolated per-call memory increment. '+
      'GPU power, utilization and memory samples describe the whole device. Energy integrates available power samples and may include other processes. '+
      'Short stages with fewer than two samples do not receive an energy estimate. CPU energy is unavailable and is not invented.','',
      'MOSEK uses two threads and a five-second mixed-integer time limit per solve. Incumbents must pass explicit primal-feasibility checks. '+
      'Continuous relaxation values are used as performance upper bounds only when solved to optimal status. '+
      'Cached full solutions are used only during calibration replay; final test policies execute their solver calls normally.','',
      'Completed serial checkpoints are preserved. Resumed independent scenarios run on disjoint two-CPU sets, with policies sequential inside each scenario. '+
      'Worker affinity is saved in worker_configuration. Absolute wall-time comparisons remain platform dependent; CPU time and solver time provide additional checks.','',
      '## Recommendations for the deliverable','']
    res=pd.read_csv(root/'forecast'/'baseline_a1.6'/'reserve.csv')
    lines += [f"The calibrated demand reserves range from {res.reserve_percent_of_mean.min():.2f}% to {res.reserve_percent_of_mean.max():.2f}% of training mean cycle demand. "+
              'Report the measured cell-specific reserve and held-out exceedance, rather than interpret a fixed 3% as universal protection.','',
      'Keep the report demand definition at cycle level. Add a separate demand-prediction residual term when demand is forecast rather than contractual and known. '+
      'Do not merge demand reserve and channel standard deviation without a joint error model.','',
      f"A cycle lasts 24 ms, while the baseline complete chance solve takes {1000*complete.mean_wall_s:.1f} ms on average. "+
      'The reported service results are logical-cycle simulations and do not subtract service lost while optimization runs. '+
      'Deployment therefore requires advance planning or continued use of a valid old schedule during computation. '+
      'The report should include measurement, computation and configuration delays in the assessment rule. '+
      'A certificate computed from information at solve start should be reassessed at completion, or include an allowance for deterioration during solver latency.','',
      'Describe adaptive trust as a tested score adaptation inspired by learning-augmented allocation. '+
      'Competitive-ratio results for online vertices that can match only once do not directly apply to repeated beam scheduling. '+
      'Any theoretical bound for this controller needs its own assumptions and proof.','',
      'Keep the claim on adaptive intervals empirical unless a verified deterministic deterioration bound is available. '+
      'Use the operational test tables to decide whether repair provides a measurable computation advantage. '+
      'The original report is preserved; this document supplies proposed revisions and reproducible evidence.','',
      '## Source and asset provenance','',
      '- Original report fixture generated from seed 20260916, eight cells, twelve slots, mean cell quality uniform on [2.8,4.8], '+
      'slot variation lognormal with log standard deviation 0.12, channel-error fraction 0.20.',
      '- [Whittle estimation methodology](https://www.frontiersin.org/journals/network-physiology/articles/10.3389/fnetp.2023.1204757/full).',
      '- [Learning-augmented matching with advice](https://www.columbia.edu/~wm2428/papers/matching_with_advice.pdf).',
      '- Input script provenance and SHA256 checksums are recorded in the source manifest.',
      '', 'All generated scientific figures are in `figures/`. No figure contains an internal title. '+
      'Raw results, stage resource records and environment versions accompany this report.']
    bench=pd.read_csv(root/'generation_backend_benchmark.csv')
    lines += ['','## Measured generation workload sensitivity','',
              '| Slots per cell | Users | CPU seconds | GPU seconds | CPU time / GPU time |',
              '|---:|---:|---:|---:|---:|']
    for (steps,users),g in bench.groupby(['steps','users']):
        values=g.groupby('device').wall_s.mean()
        lines.append(f"| {steps:,} | {users} | {values['cpu']:.5f} | {values['cuda']:.5f} | {values['cpu']/values['cuda']:.3f} |")
    lines += ['', 'These are repeated generation measurements, including host transfer, without disk compression. '+
              'GPU execution is not automatically faster for small event counts. The formal data were generated on the GPU as configured.']
    statuses=pd.read_csv(root/'run-chance-reference_results.csv').groupby(['policy','status']).size()
    lines += ['', '## Solver status counts','', '| Policy | Status | Cycles |','|---|---|---:|']
    for (policy,status),n in statuses.items(): lines.append(f"| {policy.replace('_',' ')} | {status} | {n} |")
    lines += ['', 'An optimal-inaccurate equal-bandwidth result denotes a checked feasible incumbent under the solver budget, not a proved exact optimum.']
    opfile=root/'operational_summary.csv'
    if opfile.exists():
        op=pd.read_csv(opfile); g=op[op.scenario=='baseline_a1.6']
        lines += ['', '## Sequential baseline results','',
            '| Policy | Coverage | Mean cell violation | Cumulative wall seconds | Full solves | Repairs |',
            '|---|---:|---:|---:|---:|---:|']
        for _,r in g.iterrows():
            lines.append(f"| {r.policy.replace('_',' ')} | {r.coverage:.5f} | {r.violation:.5f} | {r.total_wall_s:.3f} | {int(r.full_calls)} | {int(r.repair_calls)} |")
        lines += ['', 'These results cover 120 consecutive cycles. A positive test certificate is distinct from a verified safe interval between observations. '+
                  'The subcycle-safe-interval flags show whether the proposed interval can be supported at the cycle assessment resolution.']
        adaptive=g[g.policy=='adaptive_interval'].iloc[0]
        lines += ['',f"For the baseline adaptive controller, {100*adaptive.subcycle_safe_fraction:.1f}% of epochs have a computed safety interval below one cycle. "+
                  'Its savings should be attributed primarily to reuse and repair unless assessment skipping is separately demonstrated. '+
                  'The present results do not establish a guaranteed interval between observations.']
    (root/'experiment_report.md').write_text('\n'.join(lines)+'\n')
    # Plain summary of only observed findings for quick review.
    write_json(root/'completion_manifest.json',dict(datasets=len(list((root/'data').glob('*/traffic.npz'))),
             allocation_rows=len(summary),raw_allocation_rows=len(pd.read_csv(root/'allocation_results.csv')),
             generated_figures=len(list((root/'figures').glob('*.pdf'))),report='experiment_report.md'))
