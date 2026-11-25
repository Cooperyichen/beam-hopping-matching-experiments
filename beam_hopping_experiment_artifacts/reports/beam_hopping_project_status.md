# Beam Hopping Matching 数值实验项目状态

更新时间：2026 年 6 月 30 日

## 总体状态

当前进入第一轮本地实现。核心方向已经固定：单时隙问题使用二分图 maximum weight matching 的线性规划松弛来证明 Total Unimodularity 的有效性；跨时隙问题把 demand satisfaction 作为 covering lower bound，并在后续阶段引入 online matching 与 learning augmented online matching 工具。

## 节点状态

| 阶段 | 内容 | 当前状态 | 主要产物 |
|---|---|---|---|
| Phase 0 | 项目结构与符号规范 | 已启动 | 本状态文件，统一符号约定 |
| Phase 1 | 单时隙 Total Unimodularity matching 实验 | 本地代码已实现，正式本地实验已完成 | `single_slot_matching.py`，`phase_1_single_slot_matching_report.md` |
| Phase 2 | Beam hopping 文献筛选 | 已完成最终筛选 | `phase_2_literature_screening_report.md` |
| Phase 3 | 单时隙复杂度对比 | 已完成文献校准场景汇总，第一版 DRL/CMADRL 训练复现已实现 | `phase_3_single_slot_comparison_report.md` |
| Phase 4 | 跨时隙 online matching 与 learning augmented 方法 | 已完成第一版 online matching prototype | `phase_4_cross_time_online_matching_report.md` |

## 当前关键判断

- 边权定义为链路在当前 slot 被选中时的理论可交付 traffic volume：

$$
w_t^{b,i}=T_{\text{slot}}\bar R_t^{b,i}.
$$

- Demand 不是边权，而是跨时隙必须满足的 lower bound：

$$
\sum_{t=1}^{T}\sum_{b=1}^{M}T_{\text{slot}}\bar R_t^{b,i}x_t^{b,i}\ge Q_i,\quad \forall c_i\in\mathcal{C}.
$$

- 单时隙阶段只证明 matching constraint matrix 的 Total Unimodularity，不把 $Q_i$ 加入硬约束。

## 已生成的本地实验结果

- Phase 1 CSV：`phase_1_single_slot_results.csv`。
- Phase 1 summary：`phase_1_summary.csv`。
- Phase 3 CSV：`phase_3_single_slot_comparison.csv`。
- Phase 3 summary：`phase_3_summary.csv`。
- Phase 3 literature calibrated summary：`phase_3_literature_calibrated_summary.csv`。
- Phase 3 DRL/CMADRL reproduction：`phase_3_drl_marl_reproduction.csv`。
- Phase 4 CSV：`phase_4_cross_time_online_matching.csv`。
- Phase 4 summary：`phase_4_cross_time_online_matching_summary.csv`。
- 图表：`phase_1_runtime_boxplot.png`，`phase_1_integrality_violation.png`，`phase_3_objective_ratio.png`。

## 状态审计说明

Phase 2 已完成最终筛选。Phase 3 已根据 Phase 2 结果生成 DRL/MADRL 风格 $N=19,K=4$ 和 Genetic Algorithm 风格 $N=37,K=10$ 两类文献校准场景。当前新增第一版 DRL/CMADRL training reproduction，已经可以训练、评估并输出 CSV。该版本尚不能视为论文级性能复现。

## 下一步

- 为 Phase 4 增加 exact matching oracle 或 rolling horizon solver。
- 为 Phase 3 增加 specialized maximum weight bipartite matching algorithm。
- 继续扩展 DRL/CMADRL training pipeline，区分 traffic satisfaction reward 与 throughput delay fairness reward。

## 当前阻塞

ThinkPad SSH 认证成功，但远端 shell wrapper 报错：

```text
/bin/bash: line 1: -d Ubuntu --exec bash -lc: command not found
```

用户已确认改为 MacBook 本地工作。ThinkPad 阻塞不再阻止当前项目推进。

## Supplement recorded on 2026-11-24

The forecasting PDF strategies are now packaged for collaborator reuse, with explicit source lineage, historical evidence and a separate deficit matching controller. Implementation checks do not establish the theorem assumptions or chance guarantees. See [PROJECT_JOURNAL.md](PROJECT_JOURNAL.md) and the [reuse guide](../code/synthetic_traffic_validation/README.md). Earlier node connectivity statements above are historical; thinkpad-wsl was reachable during this task.
