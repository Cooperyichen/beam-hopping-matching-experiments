# Phase 3 报告：单时隙复杂度与参考算法对比

## 当前状态

本阶段已经执行本地对比实验。Phase 2 完成后，本报告进一步把结果拆分为两个文献校准场景：DRL/MADRL 风格的 $N=19,K=4$ 场景，以及 Genetic Algorithm 风格的 $N=37,K=10$ 场景。

## 实验目标

比较以下方法在同一单时隙 instance 上的 throughput、runtime、CPU time、memory usage 和解质量：

- Total Unimodularity Linear Programming relaxation；
- Binary Integer Linear Programming benchmark；
- Genetic Algorithm adapted baseline；
- greedy demand based baseline；
- random feasible matching baseline。

## 控制变量

所有方法使用同一个：

- feasible edge set $\mathcal{E}_t$；
- edge weight matrix $w_t^{b,i}$；
- $M,N,K,T_{\text{slot}}$；
- fixed $B_0$、$P_0$、$g_t^{b,i}$、$\sigma_i^2$。

Demand rate $D_t^i$ 只作为评价指标，不作为单时隙硬约束。

## 实现方法

| Method | 含义 | 实现方式 |
|---|---|---|
| Linear Programming relaxation | 利用 Total Unimodularity 的连续松弛 | `cvxpy` 调用 MOSEK |
| Binary Integer Linear Programming benchmark | 二元整数规划对照 | `cvxpy` 调用 MOSEK |
| random feasible matching | 随机可行 matching | 随机扫描 feasible edges 并保持 beam/cell 不冲突 |
| greedy throughput | 贪心最大 throughput | 按 $w_t^{b,i}$ 降序选择不冲突 edge |
| demand priority greedy | 需求加权贪心 | 用 $D_t^i$ 形成 cell priority，只作为 scoring，不作为 hard constraint |
| genetic algorithm | 遗传算法近似 | population、crossover、mutation、repair 后按 throughput fitness 选择 |

完整 DRL/MADRL training pipeline 暂未实现。原因是其 offline training time、state definition、reward function 和 neural network architecture 会引入额外变量。当前阶段只使用其 reported conclusions 和可解释 baseline。

## 全部场景结果摘要

结果文件为：

`phase_3_single_slot_comparison.csv`

| Method | Rows | Mean objective | Mean wall time | Median wall time |
|---|---:|---:|---:|---:|
| Linear Programming relaxation | $36$ | $3.807616\times 10^7$ | $0.005600$ s | $0.004841$ s |
| Binary Integer Linear Programming benchmark | $36$ | $3.807616\times 10^7$ | $0.004912$ s | $0.004923$ s |
| greedy throughput | $36$ | $3.805776\times 10^7$ | $0.000047$ s | $0.000040$ s |
| genetic algorithm | $36$ | $3.790028\times 10^7$ | $0.272947$ s | $0.210070$ s |
| demand priority greedy | $36$ | $3.687459\times 10^7$ | $0.000048$ s | $0.000040$ s |
| random feasible matching | $36$ | $3.629594\times 10^7$ | $0.000040$ s | $0.000033$ s |

## 文献校准场景结果

结果文件为：

`phase_3_literature_calibrated_summary.csv`

### DRL/MADRL 风格场景

该场景来自 DRL scheduling 与 MADRL beam pattern 文献中的常见设置：

$$
N=19,\quad K=4.
$$

| Method | Rows | Mean objective | Median wall time |
|---|---:|---:|---:|
| Linear Programming relaxation | $18$ | $3.856839\times 10^7$ | $0.003326$ s |
| Binary Integer Linear Programming benchmark | $18$ | $3.856839\times 10^7$ | $0.003470$ s |
| greedy throughput | $18$ | $3.853838\times 10^7$ | $0.000017$ s |
| genetic algorithm | $18$ | $3.846011\times 10^7$ | $0.087650$ s |
| demand priority greedy | $18$ | $3.722176\times 10^7$ | $0.000020$ s |
| random feasible matching | $18$ | $3.703234\times 10^7$ | $0.000014$ s |

### Genetic Algorithm 风格场景

该场景来自 Genetic Algorithm dynamic beam hopping 文献：

$$
N=37,\quad K=10.
$$

| Method | Rows | Mean objective | Median wall time |
|---|---:|---:|---:|
| Linear Programming relaxation | $18$ | $3.758393\times 10^7$ | $0.005310$ s |
| Binary Integer Linear Programming benchmark | $18$ | $3.758393\times 10^7$ | $0.005789$ s |
| greedy throughput | $18$ | $3.757714\times 10^7$ | $0.000058$ s |
| genetic algorithm | $18$ | $3.734046\times 10^7$ | $0.410030$ s |
| demand priority greedy | $18$ | $3.652742\times 10^7$ | $0.000060$ s |
| random feasible matching | $18$ | $3.555954\times 10^7$ | $0.000047$ s |

## 当前结论

- Linear Programming relaxation 与 Binary Integer Linear Programming benchmark 在两个文献校准场景中目标值匹配，支持单时隙 Total Unimodularity 分析。
- Greedy throughput 在单时隙 throughput objective 下很接近最优，但没有跨时隙 demand fairness 保证。
- Genetic Algorithm 接近最优，但 median wall time 明显高于 matching oracle 与 greedy baseline。
- Demand priority greedy 在单时隙 throughput objective 下会牺牲 objective value，它更适合 Phase 4 cross time demand coupling。
- 完整 DRL/MADRL 训练仍未复现，因此本阶段不能声称完成相对于 learning based 方法的 full reproduction comparison。

## 图表

已生成：

- `phase_3_objective_ratio.png`。

## 下一步

- 增加 specialized maximum weight bipartite matching algorithm。
- 如果需要完整 learning based baseline，建立 DRL/MADRL training pipeline，并单独报告 offline training time 与 online inference time。
