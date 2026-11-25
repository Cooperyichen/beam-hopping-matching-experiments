"""English checkpoint report and uncrowded, title-free scientific figures."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .revision import OUT, SCENARIOS, CORE
from ..resources import write_json

LABELS={'throughput_matching':'Throughput matching','urgency_no_prediction':'Historical demand matching',
        'prediction_matching':'Prediction matching','reserve_matching':'Reserve matching',
        'nominal_horizon':'Nominal horizon optimizer','chance_horizon':'Chance horizon optimizer',
        'static_trust_matching':'Static prediction trust','certified_static_matching':'Certified hybrid'}
KEY=['stage','scenario','horizon','scale','condition','policy']


def interval(values,statistic=np.mean,block=4,seed=981):
    a=np.asarray(values); n=len(a); rng=np.random.default_rng(seed)
    if n<2: return (np.nan,np.nan)
    estimates=[]
    for _ in range(1000):
        starts=rng.integers(n,size=int(np.ceil(n/block)))
        ix=np.concatenate([(s+np.arange(block))%n for s in starts])[:n]
        estimates.append(statistic(a[ix]))
    return tuple(np.quantile(estimates,[.025,.975]))


def figures(summary,forecasts):
    folder=OUT/'figures'; folder.mkdir(exist_ok=True)
    inventory=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name,caption):
        fig.tight_layout()
        for ext in ['png','pdf']: fig.savefig(folder/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig); inventory.append(dict(name=name,caption=caption))
    for scenario in SCENARIOS:
        g=summary[(summary.scenario==scenario)&(summary.stage=='B_load')]
        for metric,ylabel,suffix in [('pooled_coverage','Useful demand coverage','coverage'),('worst_cell_violation','Worst cell violation probability','violation')]:
            fig,ax=plt.subplots(figsize=(7,4.5))
            for policy in CORE:
                a=g[g.policy==policy].sort_values('scale')
                name=LABELS[policy]+(' / fallback' if policy in ['chance_horizon','nominal_horizon'] else '')
                ax.plot(a.scale,a[metric],marker='o',ms=4,label=name)
            ax.set(xlabel='Uniform demand multiplier',ylabel=ylabel,ylim=(-.03,1.03))
            ax.grid(alpha=.2); ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False,fontsize=8)
            save(fig,f'load_{suffix}_{scenario}',f'{ylabel} against demand multiplier for {scenario}. All policies share the same windows and channel draws. Unprotected solver fallbacks remain in the service statistics and are identified separately in the feasibility table.')
        a=g[g.policy=='chance_horizon'].sort_values('scale')
        fig,ax=plt.subplots(figsize=(6.5,4.3)); bottom=np.zeros(len(a)); xx=np.arange(len(a))
        for values,label in [(a.windows-a.infeasible_windows-a.unresolved_windows,'Feasible incumbent'),(a.infeasible_windows,'Proved infeasible'),(a.unresolved_windows,'Unresolved')]:
            ax.bar(xx,values,bottom=bottom,label=label); bottom+=values.to_numpy()
        ax.set(xticks=xx,xticklabels=a.scale.astype(str),xlabel='Uniform demand multiplier',ylabel='Number of chance optimizer windows')
        ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False,fontsize=8)
        save(fig,f'feasibility_{scenario}','Full variable bandwidth chance model status on the frozen test windows. An unresolved call is not evidence of infeasibility. Failed or unresolved references execute the separately flagged matching fallback.')
        for label in ['slack','boundary']:
            a=summary[(summary.scenario==scenario)&(summary.stage=='C_hybrid_'+label)&(summary.policy=='certified_static_matching')].sort_values('horizon')
            fig,ax=plt.subplots(figsize=(6.2,4.5)); bottom=np.zeros(len(a)); xx=np.arange(len(a))
            for col,title in [('matching_direct_fraction','Direct matching'),('repair_accepted_fraction','Repair accepted'),('full_accepted_fraction','Full optimization accepted'),('unprotected_fallback_fraction','Unprotected fallback')]:
                ax.bar(xx,a[col],bottom=bottom,label=title); bottom+=a[col].to_numpy()
            ax.set(xticks=xx,xticklabels=[f'{int(h)} slots' for h in a.horizon],ylabel='Fraction of independently planned windows',ylim=(0,1.02))
            ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False,fontsize=8)
            save(fig,f'branches_{scenario}_{label}',f'Mutually exclusive final branches at the {label} load. These are new matching plans, not reuse of old schedules. Load multipliers are scenario-specific; the boundary was selected using calibration only.')
    for scenario in SCENARIOS:
        a=forecasts[(forecasts.scenario==scenario)&(forecasts.partition=='test')]
        fig,ax=plt.subplots(figsize=(6.4,4.2))
        for model,g in a.groupby('model'):
            m=g.groupby('horizon').relative_rmse.mean()
            ax.plot(m.index*2,m.values,marker='o',label=model.upper() if model in ['ar','farima'] else model.capitalize())
        ax.set(xlabel='Forecast and service window (ms)',ylabel='Cell-mean relative RMSE')
        ax.grid(alpha=.2); ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False)
        save(fig,f'forecast_horizon_{scenario}','Cell-averaged root mean squared cumulative-demand error divided by each cell mean demand. Physical training and calibration boundaries are identical at every horizon.')
    representative=json.loads((OUT/'formal_budget.json').read_text())['representatives']
    baseline=summary[(summary.stage=='B_load')&summary.apply(lambda r:r.scale==representative[r.scenario]['boundary'],axis=1)]
    a=pd.concat([summary[summary.stage.isin(['C_hybrid_boundary','C_horizon_boundary'])],baseline])
    for scenario in SCENARIOS:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for policy in ['prediction_matching','reserve_matching','nominal_horizon','chance_horizon','certified_static_matching']:
            g=a[(a.scenario==scenario)&(a.policy==policy)].sort_values('horizon')
            ax.plot(g.horizon*2,g.latency_p95_ms,marker='o',label=LABELS[policy])
        ax.plot([24,48,96],[24,48,96],color='black',ls='--',label='Window duration')
        ax.set(xlabel='Window duration (ms)',ylabel='95th percentile online latency (ms)',yscale='log')
        ax.grid(alpha=.2); ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False,fontsize=8)
        save(fig,f'latency_horizon_{scenario}','Online latency includes forecast filtering, trust update where used, assignment, checks, all repair attempts and full optimization. Twelve-slot core values are supplied separately in the load tables; no latency-free real-time execution is assumed.')
    pd.DataFrame(inventory).to_csv(OUT/'figure_inventory.csv',index=False)
    return inventory


def main():
    raw=pd.read_csv(OUT/'static_results.csv'); attempts=pd.read_csv(OUT/'static_attempts.csv')
    forecasts=pd.read_csv(OUT/'forecast_metrics.csv'); budget=json.loads((OUT/'formal_budget.json').read_text())
    summaries=[]; intervals=[]
    for key,g in raw.groupby(KEY):
        g=g.sort_values('origin'); r=dict(zip(KEY,key)); n=len(g)
        r.update(windows=n,pooled_coverage=g.useful_mbit.sum()/g.demand_mbit.sum(),
            mean_window_coverage=g.horizon_coverage.mean(),mean_cell_violation=g.average_cell_violation.mean(),
            worst_cell_violation=max(g[f'cell_{i}_violation'].mean() for i in range(8)),
            any_cell_violation=g.any_cell_violation.mean(),mean_cell_coverage=g.mean_cell_coverage.mean(),
            minimum_cell_coverage=min(g[f'cell_{i}_coverage'].mean() for i in range(8)),
            useful_mbps=g.useful_mbps.mean(),raw_mbps=g.raw_mbps.mean(),
            latency_mean_ms=g.total_online_s.mean()*1000,latency_median_ms=g.total_online_s.median()*1000,
            latency_p95_ms=g.total_online_s.quantile(.95)*1000,latency_p99_ms=g.total_online_s.quantile(.99)*1000,
            mean_latency_window_ratio=g.latency_window_ratio.mean(),exceeds_window_fraction=g.exceeds_window.mean(),
            startup_windows=int(g.first_template_call.sum()),
            steady_latency_median_ms=g.loc[~g.first_template_call,'total_online_s'].median()*1000,
            repair_attempt_window_fraction=(g.repair_attempts>0).mean(),full_call_window_fraction=(g.full_attempts>0).mean(),
            repair_attempts=int(g.repair_attempts.sum()),full_attempts=int(g.full_attempts.sum()),
            infeasible_windows=int((g.solver_classification=='proved_infeasible').sum()),
            unresolved_windows=int((g.solver_classification=='unresolved_without_feasible_incumbent').sum()),
            feasible_unproved_optimal_windows=int((g.solver_classification=='feasible_incumbent_unproved_optimal').sum()),
            unprotected_execution_fraction=(g.unprotected_solver_fallback|(g.final_path=='unprotected_fallback')).mean())
        valid=g[~g.unprotected_solver_fallback&~g.final_path.eq('unprotected_fallback')]
        r['incumbent_only_coverage']=valid.useful_mbit.sum()/valid.demand_mbit.sum() if len(valid) else np.nan
        r['incumbent_only_mean_violation']=valid.average_cell_violation.mean() if len(valid) else np.nan
        r['incumbent_windows']=len(valid)
        for name in ['matching_direct','repair_accepted','full_accepted','unprotected_fallback']:
            r[name+'_fraction']=(g.final_path==name).mean()
        denominator=int((g.repair_attempts>0).sum())
        r['conditional_repair_success']=int((g.final_path=='repair_accepted').sum())/denominator if denominator else np.nan
        summaries.append(r)
        pv=g[[f'cell_{i}_violation' for i in range(8)]].to_numpy()
        for block in [2,4,8]:
            lo,hi=interval(g.horizon_coverage.to_numpy(),block=block)
            vlo,vhi=interval(pv,statistic=lambda a:np.max(a.mean(0)),block=block)
            intervals.append(dict(zip(KEY,key),windows=n,block_windows=block,coverage_lo=lo,coverage_hi=hi,
                worst_violation_lo=vlo,worst_violation_hi=vhi))
    summary=pd.DataFrame(summaries); summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'block_interval_sensitivity.csv',index=False)
    status=attempts.groupby(['stage','scenario','horizon','scale','policy','mode','classification']).size().reset_index(name='calls')
    status.to_csv(OUT/'solver_status_summary.csv',index=False)
    inventory=figures(summary,forecasts)
    blocks=pd.DataFrame(intervals); audit=pd.read_csv(OUT/'source_audit.csv')
    def table(frame,cols):
        f=frame[cols].copy()
        if 'policy' in f: f['policy']=f.policy.map(lambda x:LABELS.get(x,x))
        if 'scenario' in f: f['scenario']=f.scenario.replace({'baseline_a1.6':'Baseline','hotspot_a1.6':'Fixed hotspots'})
        for c in f.select_dtypes('number'): f[c]=f[c].map(lambda x:'N/A' if pd.isna(x) else str(int(x)) if x==int(x) else f'{x:.3f}')
        names={'scenario':'Scenario','horizon':'Slots','scale':'Load','policy':'Method',
            'pooled_coverage':'Coverage','worst_cell_violation':'Worst cell violation',
            'mean_cell_violation':'Mean cell violation','infeasible_windows':'Infeasible',
            'unresolved_windows':'Unresolved','unprotected_execution_fraction':'Fallback fraction',
            'latency_median_ms':'Median ms','latency_p95_ms':'95th percentile ms',
            'exceeds_window_fraction':'Over window fraction','relative_rmse':'Relative RMSE',
            'rmse_mbit':'RMSE Mbit','condition':'Condition','windows':'Windows',
            'matching_direct_fraction':'Direct matching','repair_attempt_window_fraction':'Repair attempted',
            'repair_accepted_fraction':'Repair accepted','conditional_repair_success':'Repair success given attempt',
            'full_call_window_fraction':'Full solve called','unprotected_fallback_fraction':'Unprotected fallback'}
        f=f.rename(columns={c:names.get(c,c.replace('_',' ').capitalize()) for c in cols})
        lines=['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(cols))+' |']
        lines+=['| '+' | '.join(str(v) for v in row)+' |' for row in f.itertuples(index=False,name=None)]
        return '\n'.join(lines)
    load=summary[summary.stage=='B_load']
    def point(scenario,scale,policy):
        return load[(load.scenario==scenario)&(load.scale==scale)&(load.policy==policy)].iloc[0]
    p1=point('baseline_a1.6',1.,'prediction_matching')
    p2=point('baseline_a1.6',1.25,'prediction_matching')
    rp=point('baseline_a1.6',1.25,'reserve_matching')
    ch=point('baseline_a1.6',1.25,'chance_horizon')
    forecast_mean=forecasts[forecasts.partition=='test'].groupby(['scenario','horizon','model']).relative_rmse.mean()
    improvement={h:1-forecast_mean['baseline_a1.6',h,'farima']/forecast_mean['baseline_a1.6',h,'persistence'] for h in [12,24,48]}
    hv=summary[(summary.policy=='certified_static_matching')&summary.stage.str.startswith('C_hybrid_boundary')]
    slack=summary[(summary.policy=='certified_static_matching')&(summary.stage=='C_hybrid_slack')&(summary.horizon==48)]
    selected_intervals=blocks[(blocks.stage=='B_load')&(blocks.scenario=='baseline_a1.6')&(blocks.scale==1.25)&(blocks.block_windows==4)&blocks.policy.isin(['prediction_matching','reserve_matching','chance_horizon'])]
    resources=json.loads((OUT/'resources/formal_B_C.json').read_text())
    findings=[
        f'The load scan supports a resource margin explanation for part of the earlier near perfect coverage. In the baseline, prediction matching has {p1.mean_cell_violation:.1%} mean cell violation at load 1.00 and {p2.mean_cell_violation:.1%} at load 1.25. Its pooled coverage changes from {p1.pooled_coverage:.2%} to {p2.pooled_coverage:.2%}. Coverage alone therefore conceals many small service shortfalls. At load 1.25 the worst cell violation is {p2.worst_cell_violation:.1%}, rather than the average {p2.mean_cell_violation:.1%}.',
        f'At that same load, reserve matching reduces mean cell violation to {rp.mean_cell_violation:.1%} and raises coverage to {rp.pooled_coverage:.2%}. The chance optimizer execution, including two unresolved matching fallbacks, has {ch.mean_cell_violation:.1%} mean cell violation. Conditional on its {int(ch.incumbent_windows)} feasible optimizer outputs, the corresponding mean is {ch.incumbent_only_mean_violation:.1%}. The latter conditional statistic excludes difficult unresolved cases and must not replace the complete execution statistic.',
        'At loads 1.50 and 1.75, every tested variable bandwidth chance problem is proved infeasible in both spatial scenarios. At 1.50 the nominal problems remain feasible. These observations distinguish failure of the specified reliable service requirement from inability to transmit most of the traffic; they do not prove that no other risk model or service relaxation could be feasible. Solver failure at 1.25 is separately recorded as unresolved, not as a capacity proof.',
        f'In the baseline, FARIMA improves relative RMSE over persistence by {improvement[12]:.1%}, {improvement[24]:.1%} and {improvement[48]:.1%} at 12, 24 and 48 slots. Its relative cumulative demand error nevertheless increases with horizon. The difference from ordinary autoregression is very small at 12 and 24 slots. These results do not justify a general claim that fractional modelling is necessary.',
        f'At the boundary load, none of the {int(hv.windows.sum())} static hybrid windows directly accepts matching. The complete optimizer is consequently still the dominant branch. At 48 slots and the slack load, direct acceptance occurs in {int(round((slack.matching_direct_fraction*slack.windows).sum()))} of {int(slack.windows.sum())} windows. These small samples indicate a load dependent opportunity for reduced optimization, not established long term reuse savings.',
        f'In the baseline at 12 slots and load 1.25, reserve matching has median latency {rp.latency_median_ms:.2f} ms. Chance optimization with fallback has median {ch.latency_median_ms:.1f} ms and 95th percentile {ch.latency_p95_ms:.0f} ms against a 24 ms window. Every such chance optimizer execution exceeds the window. This static experiment has not demonstrated real time execution of the protected policy.',
        f'The formal B and C evaluation takes {resources["wall_s"]:.1f} s wall time and {resources["cpu_s"]:.1f} s process CPU time, with lifetime peak resident memory {resources["process_lifetime_peak_rss_mib"]:.0f} MiB. No new GPU traffic generation is needed; measured GPU allocation is zero. Device telemetry includes background device power and must not be interpreted as incremental optimizer energy.'
    ]
    text=['# Demand load and forecast horizon validation',
      '## Scope and experimental status',
      'This independent experiment completes stages A, B and C of the revision requirements. Stage D, which concerns reuse of an earlier schedule across successive cycles, has not been run. The existing deliverable, presentation and earlier formal results are unchanged.',
      '## Main findings',
      *findings,
      'The study uses the previously observed traffic trajectories. It therefore provides a paired sensitivity analysis, not a new independent holdout confirmation. Each spatial scenario contains one continuous traffic trajectory. Repeated Gaussian channel draws are conditional repetitions, not independent traffic replications.',
      '## Model and data',
      'Eight cells share three illumination opportunities. Slots last 2 ms, total bandwidth is 500 MHz and the recorded power is 1000 W. The spectral-efficiency fixture is synthetic and is not derived from a physical link budget. The bandwidth cap of 0.6 is an exploratory model choice. Every opportunity can illuminate every cell, with identical opportunity weights for a given cell and slot. No sparse graph, cell geometry or intercell interference is introduced.',
      'The baseline has the original unequal cell proportions. The first two cells in the fixed-hotspot scenario carry approximately half the total mean demand. Source counts and independence are unchanged. Uniform load scaling multiplies training, calibration and test traffic, predictions and residual reserves by the same factor. The standardized linear predictor and its selected parameters are invariant under this scaling.',
      table(audit,['scenario','training_total_mean_mbit_12','full_total_mean_mbit_12','first_two_cells_share','users']),
      'The original demand levels were based on service from a known feasible validation schedule, rather than measured population or production traffic. Small differences in scenario totals arise from integer source counts and finite simulated trajectories.',
      'At the first slot of a window the predictor observes only earlier slots. It predicts cumulative demand over the immediately upcoming 12, 24 or 48 slots. The corresponding service windows are 24, 48 and 96 ms. There is no queue, arrival-time service restriction or traffic expiration. The physical partition boundaries remain 600000, 700000 and 800000 slots. Model fitting uses the first 60%, order and history selection the next 10%, reserve calibration the next 10%, and evaluation the final 20%. Targets do not cross boundaries. Histories of 128, 256 and 512 slots span 256, 512 and 1024 ms.',
      'The existing independent Gaussian channel-error model is preserved, with standard deviation equal to 20% of the mean spectral efficiency. Longer mean profiles extend the original fixture in independently seeded twelve-slot blocks; the drift phase follows physical time and is shared across load levels. This extension is a modelling convention, not measured channel data.',
      r'The upcoming target is $D_{i,j}=\sum_{t\in\mathcal{W}_j}A_i^t$, where $\mathcal{W}_j=\{n_j,\ldots,n_j+T-1\}$ and the information cutoff is $n_j-1$. Traffic volumes and service are expressed in Mbit. Let $c=T_{\mathrm{slot}}B_{\mathrm{tot}}/10^6$, let $\mathbf{b}_i$ collect bandwidth fractions across a window, and let $\overline{\mathbf{\Lambda}}_i$ and $\mathbf{\Sigma}_i$ denote the spectral efficiency mean vector and diagonal error covariance matrix. The protected service condition is',
      r'$$c\left(\mathbf{b}_i^{\mathsf T}\overline{\mathbf{\Lambda}}_i-\kappa_\Lambda\left\|\mathbf{\Sigma}_i^{1/2}\mathbf{b}_i\right\|_2\right)\geq\widehat D_{i,j}+R_{i,T},\qquad i\in\mathcal{I}.$$' ,
      r'Here $\kappa_\Lambda=\Phi^{-1}(0.975)$, $\Phi$ is the standard normal cumulative distribution function, and $R_{i,T}$ is the empirical 97.5th percentile calibration reserve for horizon $T$. The nominal reference sets the allowance and reserve to zero, while keeping the point forecast. Pure matching enforces neither of these cumulative demand conditions.',
      '## Strategies and interpretation',
      'Throughput matching uses spectral efficiency alone. Historical-demand matching uses the training mean cumulative demand. Prediction matching uses the forecast residual target. Reserve matching adds a cell-specific empirical one-sided residual quantile. Each matching plan allocates one third of the bandwidth to each active link and solves each slot assignment exactly. These strategies do not impose a horizon chance constraint.',
      'The nominal horizon optimizer maximizes expected raw service subject to the forecast demand and variable bandwidth constraints, without an uncertainty allowance. The chance horizon optimizer adds the Gaussian channel allowance and the calibrated demand reserve. Both permit bandwidth reallocation up to the cap of 0.6, so their advantage over matching cannot be attributed solely to uncertainty protection.',
      'The channel and demand risks are each set to 0.025. The channel protection is conditional on the stated Gaussian model and a schedule fixed before realization. Residual quantiles are empirically calibrated on dependent traffic. This experiment establishes no unconditional end-to-end 5% guarantee.',
      'The static prediction-trust extension uses the immediately preceding observed window, per-slot training error scales and the existing scenario-specific smoothing parameters. Its error state is initialized to zero for each independently evaluated window. It is separately named from the earlier sequential trust rule. The certified static hybrid checks a newly generated matching plan, attempts the existing restricted repair if at most one quarter of cells fail, and otherwise invokes full optimization. A local repair may release the same donor cells as the existing implementation. The bandwidth-only attempt fixes all unaffected cell bandwidths, which can prevent resource transfer when the slot is already fully allocated.',
      r'The inherited urgency weight is $\lambda=0.5$ for the baseline and $\lambda=2$ for fixed hotspots. The respective trust pairs $(\gamma,\kappa)$ are $(0.8,1)$ and $(0.95,2)$. These values are fixed from the earlier calibration, rather than selected on this test. A comparison between spatial scenarios consequently includes their inherited tuning; it does not isolate spatial concentration as the sole cause of every performance difference.',
      '## Budget and feasibility',
      f'The calibration pilot froze load levels {budget["levels"]}, with {budget["load_windows"]} windows per scenario and load in stage B. The independent-window planning target was a conservative 95% half-width of 0.18; temporal dependence can widen the uncertainty. Longer-horizon core comparisons use {budget["horizon_windows"]} windows per scenario and representative load. The projected initial core budget was {budget["predicted_core_full_calls"]} optimizer calls. The actual attempt log records {len(attempts)} calls including every failed repair and fallback.',
      f'Representative calibration-selected loads are `{json.dumps(budget["representatives"])}`. The boundary label denotes the largest tested load with feasible pilot incumbents in every pilot window; it is not a proof of the global physical capacity boundary.',
      'Only the solver status infeasible is counted as a proved infeasibility. A time limit without an incumbent or an inaccurate infeasibility status is unresolved. Feasible incumbents without an optimal status are not called optimal. When a horizon reference has no valid incumbent, its explicitly unprotected reserve-matching fallback is included in execution metrics and flagged. Thus an infeasible reference can still have a reported coverage; this is not the coverage of a solved infeasible model.',
      r'MOSEK uses two threads, a mixed integer search limit of 3 s and a relative optimization gap tolerance of $10^{-4}$. Feasible incumbents undergo an explicit constraint check with numerical service tolerance $2\times10^{-5}$ Mbit. Statements about optimality and feasibility have these numerical solver tolerances.',
      'The equal-bandwidth chance pilot is stored separately in pilot_equal.csv. Its feasible set is stricter than the variable-bandwidth model. It is not a deployment oracle. Perfect-demand cases are labelled information interventions, not deployable forecasts.',
      table(pd.read_csv(OUT/'pilot_equal.csv'),['scenario','scale','solver_classification','total_online_s']),
      '## Load sensitivity',
      'Coverage is useful service capped at each cell demand divided by total demand. Raw service is not capped. Worst-cell violation first averages each named cell over all sampled windows and then takes the largest average; it is not an average of per-window maxima. Minimum cell coverage similarly takes the minimum of per-cell averages. The raw table also preserves the distinct mean per-draw minimum coverage.',
      table(summary[summary.stage=='B_load'],['scenario','scale','policy','pooled_coverage','worst_cell_violation','infeasible_windows','unresolved_windows','unprotected_execution_fraction']),
      '## Forecast horizon comparison',
      table(forecasts[forecasts.partition=='test'].groupby(['scenario','horizon','model'],as_index=False)[['relative_rmse','rmse_mbit','reserve_exceedance']].mean(),['scenario','horizon','model','relative_rmse','rmse_mbit']),
      'Relative RMSE divides each cell cumulative-demand RMSE by that cell mean demand before averaging cells. Absolute errors must be interpreted with window duration. A longer window is not the same violation event as a shorter window. Persistence is the principal reference; historical mean is also included. The ordinary autoregressive reference has order two and uses the history selected for FARIMA; it is not a separately tuned optimum across every autoregressive order.',
      '## Static certification branches and latency',
      table(summary[summary.policy=='certified_static_matching'],['scenario','horizon','scale','condition','windows','matching_direct_fraction','repair_attempt_window_fraction','repair_accepted_fraction','conditional_repair_success','full_call_window_fraction','unprotected_fallback_fraction']),
      'The four final paths are mutually exclusive. A repair attempt can contain two solver calls. Conditional repair success has no value when no repair was attempted. None of these counts measures reuse of a schedule from an earlier completed window.',
      table(summary[summary.stage.str.startswith('C_hybrid')|summary.stage.str.startswith('C_horizon')],['scenario','horizon','scale','policy','pooled_coverage','worst_cell_violation','latency_median_ms','latency_p95_ms','exceeds_window_fraction']),
      'Online latency includes actual prediction filtering, trust-state construction where used, weight construction, matching, validation, all failed and successful repairs, and full optimization. Conditional channel simulation for scientific evaluation is outside online timing. Startup windows are identified by first use of each solver template; steady-state summaries exclude them. Process peak resident memory is a lifetime maximum, not incremental allocation. Hardware configuration signalling was not measured.',
      'Model loading and offline filter construction are outside online latency. The first optimization call for a template includes its construction and compilation. Static trust construction is timed once for each window and charged to each policy that uses it. The reported values represent separate policy execution costs, rather than the elapsed duration of running all policies together.',
      'Logical simulation ignores computation waiting. The exceedance fraction and latency-to-window ratio explicitly show where zero-lead-time real-time operation has not been demonstrated. Solver waiting limits apply to mixed-integer search, not the entire Python call including compilation.',
      '## Uncertainty attribution',
      table(summary[summary.stage=='C_ablation_fixed_reserve'],['condition','policy','pooled_coverage','worst_cell_violation','infeasible_windows','unresolved_windows']),
      'The demand reserve is identical in all four interventions. Perfect-demand interventions replace the point forecast only; deterministic-channel interventions set channel standard deviations to zero. The fixed reserve is deliberately conservative for perfect demand and is not removed, so the intervention does not confound forecast error with reserve policy.',
      '## Statistical limitations',
      'block_interval_sensitivity.csv reports circular moving-block bootstrap intervals using blocks of 2, 4 and 8 sampled windows, with 1000 resamples. The evenly spaced windows are ordered in physical time; these block sizes are a dependence sensitivity exercise, not proof that long-memory dependence is exhausted. The small longer-horizon and hybrid samples cannot precisely establish a low violation probability. Channel draws are paired across strategies and do not count as additional traffic windows.',
      'For example, the following four-window block intervals describe baseline load 1.25. Coverage bounds refer to the mean window coverage; worst cell violation is recomputed by averaging each cell and then taking the maximum within every bootstrap sample.',
      table(selected_intervals,['policy','coverage_lo','coverage_hi','worst_violation_lo','worst_violation_hi']),
      '## Figures',
      *[f'![{v["caption"]}](figures/{v["name"]}.png)\n\n{v["caption"]}' for v in inventory],
      '## Recommendations for the deliverable and the next checkpoint',
      'Replace near-perfect-coverage claims with the explicit load and feasibility context. Separate variable-bandwidth optimization from equal-resource matching. Define reliability at the window duration shown. Describe new matching acceptance and old-plan reuse as different events. State the conditional channel model and empirical nature of demand reserves.',
      'Choose the sequential study only after reviewing these results. A bounded initial design would use the baseline and fixed-hotspot boundary cases at twelve slots, followed by one longer horizon only if its full-solve feasibility and overhead are acceptable. The sequential comparison should record actual old-plan reuse and cost per simulated second, rather than infer either quantity from static branches. No sequential experiment or longer trajectory generation has been started.'
    ]
    report='\n\n'.join(text)+'\n'
    for old,new in {'end-to-end':'overall','cell-specific':'cell specific','per-slot':'per slot','per-cell':'per cell','one-sided':'one sided','zero-lead-time':'zero lead time','worst-case':'worst case','long-memory':'long memory','moving-block':'moving block','twelve-slot':'twelve slot','scenario-specific':'scenario specific','model-specific':'model specific','per-window':'per window','per-draw':'per draw','three-slot':'three slot','cell-averaged':'cell averaged','one-quarter':'one quarter','non-overlap':'nonoverlap','start-up':'startup','long-term':'long term','spectral-efficiency':'spectral efficiency','fixed-hotspot':'fixed hotspot','variable-bandwidth':'variable bandwidth','equal-bandwidth':'equal bandwidth','demand-aware':'demand aware','high-density':'high density','Twelve-slot':'Twelve slot','horizon-specific':'horizon specific','short-term':'short term','per-process':'per process','fixed-cycle':'fixed cycle','real-time':'real time','48-slot':'48 slot','24-slot':'24 slot'}.items():
        report=report.replace(old,new)
    for short,full in [('FARIMA','fractional autoregressive integrated moving average (FARIMA)'),('RMSE','root mean squared error (RMSE)'),('CPU','central processing unit (CPU)'),('GPU','graphics processing unit (GPU)')]:
        report=report.replace(short,full,1)
    (OUT/'checkpoint_report.md').write_text(report)
    write_json(OUT/'checkpoint.json',dict(A_B_C_complete=True,D_started=False,
        total_static_rows=len(raw),total_optimizer_attempts=len(attempts),
        startup_windows=int(raw.first_template_call.sum()),figures=len(inventory)))
    print(json.dumps(json.loads((OUT/'checkpoint.json').read_text()),indent=2))


if __name__=='__main__': main()
