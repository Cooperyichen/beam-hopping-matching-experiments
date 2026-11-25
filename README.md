# Beam Hopping Matching Experiments

本仓库维护 beam hopping illumination pattern 子问题的数值实验代码、报告与阶段性结果。核心目标是验证单时隙 beam cell assignment 可以被建模为二分图 matching，并利用 Total Unimodularity 得到可解释、可复现的线性规划松弛方案；随后在跨时隙 demand coupling 与 learning based scheduling setting 下扩展对比。

## 建模核心

沿用原始 beam hopping 表述中的记号：

- $\mathcal{B}=\{1,\ldots,M\}$ 表示 candidate satellite beams。
- $\mathcal{C}=\{c_i\mid i=1,\ldots,N\}$ 表示 ground cells。
- $\mathcal{T}=\{1,\ldots,T\}$ 表示 time slots。
- $x_t^{b,i}\in\{0,1\}$ 表示 beam $b$ 在 slot $t$ 是否服务 cell $c_i$。
- $T_{\text{slot}}$ 表示 slot duration。
- $D_t^i$ 表示 cell $c_i$ 在 slot $t$ 的 demand rate。
- $Q_i=\sum_{t=1}^{T}T_{\text{slot}}D_t^i$ 表示 cell $c_i$ 在 horizon 内需要满足的 total demand。
- $\bar R_t^{b,i}$ 表示 beam $b$ 到 cell $c_i$ 在 slot $t$ 被选中时的 achievable rate。

单时隙 matching 的 edge weight 定义为理论可交付 traffic volume：

$$
w_t^{b,i}=T_{\text{slot}}\bar R_t^{b,i}.
$$

Demand 不是 edge weight，而是跨时隙 lower bound：

$$
\sum_{t=1}^{T}\sum_{b=1}^{M}T_{\text{slot}}\bar R_t^{b,i}x_t^{b,i}\ge Q_i,\quad \forall c_i\in\mathcal{C}.
$$

## 阶段节点

| 阶段 | 目标 | 当前状态 | 主要产物 |
|---|---|---|---|
| Phase 0 | 建立项目结构、符号规范与报告体系 | 已完成初版 | `beam_hopping_project_status.md` |
| Phase 1 | 单时隙 Total Unimodularity matching 实验 | 已完成本地实验 | `single_slot_matching.py`，`phase_1_single_slot_matching_report.md` |
| Phase 2 | Beam hopping 文献筛选与 illumination pattern 报告 | 已完成初筛和入选理由 | `phase_2_literature_screening_report.md` |
| Phase 3 | 单时隙复杂度与参考算法对比 | 已完成 LP、BILP、greedy、GA 和第一版 DRL/CMADRL prototype | `phase3_benchmarks.py`，`drl_marl_reproduction.py` |
| Phase 4 | 跨时隙 covering constraint 与 online matching prototype | 已完成第一版 prototype | `cross_time_online_matching.py` |

## 目录结构

```text
beam_hopping_experiment_artifacts/
  code/
    single_slot_matching.py
    phase3_benchmarks.py
    drl_marl_reproduction.py
    cross_time_online_matching.py
    test_single_slot_matching.py
    test_phase3_benchmarks.py
    test_drl_marl_reproduction.py
    test_cross_time_online_matching.py
    results/
  reports/
    beam_hopping_project_status.md
    phase_1_single_slot_matching_report.md
    phase_2_literature_screening_report.md
    phase_3_single_slot_comparison_report.md
    phase_4_cross_time_online_matching_report.md
```

## 运行方式

进入代码目录：

```bash
cd beam_hopping_experiment_artifacts/code
```

运行测试：

```bash
python3 -m pytest -q
```

运行 Phase 1 单时隙 matching 实验：

```bash
python3 single_slot_matching.py \
  --solver MOSEK \
  --sizes 4x19x4,10x37x10 \
  --topologies complete sparse geometric \
  --demand-profiles uniform hotspot \
  --repetitions 3 \
  --run-bilp \
  --output results/phase_1_single_slot_results.csv
```

运行 Phase 3 heuristic benchmark：

```bash
python3 phase3_benchmarks.py \
  --solver MOSEK \
  --sizes 4x19x4,10x37x10 \
  --topologies complete sparse geometric \
  --demand-profiles uniform hotspot \
  --repetitions 3 \
  --run-bilp \
  --output results/phase_3_single_slot_comparison.csv
```

运行 Phase 3 Deep Reinforcement Learning 与 Cooperative Multi-Agent Deep Reinforcement Learning prototype：

```bash
python3 drl_marl_reproduction.py \
  --episodes 30 \
  --horizon 60 \
  --eval-horizon 80 \
  --repetitions 5 \
  --output results/phase_3_drl_marl_reproduction.csv
```

运行 Phase 4 跨时隙 online matching prototype：

```bash
python3 cross_time_online_matching.py \
  --T 20 \
  --M 10 \
  --N 37 \
  --K 10 \
  --topology sparse \
  --demand-profile hotspot \
  --prediction-modes accurate noisy adversarial \
  --repetitions 5 \
  --output results/phase_4_cross_time_online_matching.csv
```

## 当前实验数据说明

当前实验使用 synthetic data。Phase 1、Phase 3 matching 与 Phase 4 的单时隙 instance 由 `generate_instance()` 生成，包括 synthetic channel gain、noise power、demand profile 与 feasible edge topology。Phase 3 DRL/CMADRL prototype 使用 queue based environment，traffic arrival rate 使用 uniform sampling，slot arrivals 使用 Poisson sampling。

当前数据适合验证 matching formulation、Total Unimodularity、LP relaxation 与 BILP 的一致性，以及不同 heuristic 和 online matching prototype 的相对行为。当前数据还不能被解释为严格复现 beam hopping DRL 或 CMADRL 论文中的物理层仿真结果。

## 当前限制

- Phase 1 和 Phase 3 matching 中的 $g_t^{b,i}$ 是 synthetic lognormal gain，不是由 antenna pattern、path loss 与 fading model 逐项计算得到。
- Phase 3 DRL/CMADRL prototype 冻结 bandwidth allocation，只复现 illumination pattern decision layer。
- 当前 DRL/CMADRL 训练长度较短，主要用于验证 pipeline，不应解读为论文级性能复现。
- 若需要严格复现文献曲线，需要补充 paper specific channel model、traffic scenario、reward definition、network architecture 与长训练配置。

## 依赖

主要 Python 依赖包括：

- `numpy`
- `cvxpy`
- `pytest`
- `torch`
- `MOSEK`

优化求解默认使用 MOSEK。若 MOSEK 不可用，相关 LP 与 BILP 实验应停止并报告环境问题。

## Forecasting policies and deficit scheduling supplement

The [collaborator reuse guide](beam_hopping_experiment_artifacts/code/synthetic_traffic_validation/README.md) maps the Pareto forecasting PDF to its implementations, documents Yekta Demirci supplied source lineage and subsequent adaptations, and introduces a separate fixed resource deficit controller. See the [PDF policy map](beam_hopping_experiment_artifacts/code/synthetic_traffic_validation/PDF_STRATEGY_MAP.md), [deficit integration guide](beam_hopping_experiment_artifacts/code/synthetic_traffic_validation/DEFICIT_GUIDE.md), [source provenance](beam_hopping_experiment_artifacts/code/synthetic_traffic_validation/SOURCE_PROVENANCE.md), and [project journal](beam_hopping_experiment_artifacts/reports/PROJECT_JOURNAL.md). Historical results are not evidence of deficit stability.
