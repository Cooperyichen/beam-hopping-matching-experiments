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

第一版 DRL/CMADRL training pipeline 已在 `drl_marl_reproduction.py` 中实现。该实现不再只引用文献结论，而是提供可运行的训练、推理与 CSV 输出。当前仍不能称为论文级完整复现，主要限制是训练长度、网络结构细节、随机信道模型和带宽分配模块尚未完全对齐原文。

## DRL/CMADRL 复现范围

本次复现针对 Phase 2 中两个与 illumination pattern 最接近的 learning based 文献。

### 单智能体 DRL scheduling

参考对象为 `Beam_Hopping_Scheduling_Based_on_Deep_Reinforcement_Learning.pdf`。该文献使用 $N=19$ 个 virtual cells，并且每个 slot 最多激活 $K=4$ 个 beams。其核心思想是把 satellite scheduler 看作一个 DRL agent，由 agent 根据当前 traffic demand 与 channel state 选择 beam hopping action。

当前实现中的映射为：

| 文献概念 | 当前实现 |
|---|---|
| environment state | queue state $\mathbf{D}_t\in\mathbb{R}^{N\times T_{\text{ttl}}}$ |
| action | 从 $\mathcal{C}$ 中选择 $K$ 个 cells |
| agent | single Deep Q-Network |
| reward | throughput 与 delay fairness 的加权即时 reward |
| scale | $N=19,K=4$ |

单智能体 action space 为：

$$
|\mathcal{A}_{\text{DQN}}|=\binom{N}{K}.
$$

在 $N=19,K=4$ 时：

$$
|\mathcal{A}_{\text{DQN}}|=\binom{19}{4}=3876.
$$

### Cooperative Multi-Agent DRL illumination reproduction

参考对象为 `Dynamic_Beam_Pattern_and_Bandwidth_Allocation_Based_on_Multi-Agent_Deep_Reinforcement_Learning_for_Beam_Hopping_Satellite_Systems.pdf`。该文献把 agents 分成两个 group：illumination group 与 bandwidth allocation group。由于当前项目要求固定 $B_0$、$P_0$、$g_t^{b,i}$ 与 $\sigma_i^2$ 来控制无关变量，本次只复现 illumination group，并冻结 bandwidth allocation。

当前实现中的映射为：

| 文献概念 | 当前实现 |
|---|---|
| global state | $\mathbf{D}_t\in\mathbb{R}^{N\times T_{\text{ttl}}}$ |
| illumination agents | $K$ 个 independent Q-learning agents |
| action of each illumination agent | 从 $\mathcal{C}$ 中选择 $1$ 个 cell |
| duplicate action handling | sequential masking，保证同一个 slot 内不重复选择 cell |
| shared reward | throughput 与 delay fairness 的加权 reward |
| bandwidth allocation agents | 当前固定资源实验中冻结 |

多智能体版本中每个 illumination agent 的 action space 为：

$$
|\mathcal{A}_{\text{agent}}|=N.
$$

这体现了文献中降低 action space 的主要动机：单智能体直接面对 $\binom{N}{K}$ 级别的组合 action，而多智能体将每个 agent 的选择压缩到 $N$。

## DRL/CMADRL 第一版复现实验结果

结果文件为：

`phase_3_drl_marl_reproduction.csv`

实验配置：

$$
N=19,\quad M=4,\quad K=4,\quad T_{\text{ttl}}=5.
$$

训练配置：

$$
\text{training episodes}=30,\quad \text{training horizon}=60.
$$

评估配置：

$$
\text{evaluation horizon}=80,\quad \text{repetitions}=5.
$$

| Method | Mean throughput Mbit | Mean satisfaction | Mean delay gap slots | Mean inference time |
|---|---:|---:|---:|---:|
| random | $3.1204$ | $0.1045$ | $4.3842$ | $1.6839\times 10^{-6}$ s |
| periodic | $3.5942$ | $0.1045$ | $4.4302$ | $7.4553\times 10^{-7}$ s |
| greedy demand | $4.9401$ | $0.1214$ | $4.4670$ | $5.9826\times 10^{-6}$ s |
| delay fairness | $4.3226$ | $0.1522$ | $3.5726$ | $7.4924\times 10^{-6}$ s |
| single agent DQN | $2.8289$ | $0.0957$ | $4.4652$ | $5.4978\times 10^{-5}$ s |
| CMADRL IQL | $2.9605$ | $0.1043$ | $4.4600$ | $1.2963\times 10^{-4}$ s |

当前结果显示，第一版训练管线已经可运行，但 $30$ episodes 的短训练不足以复现文献中 learning based 方法优于 heuristic baseline 的性能。该结果应该被理解为 pipeline validation，而不是 final algorithmic comparison。

## 当前不能声称论文级完整复现的原因

- 目标论文没有在当前公开检索中提供可直接复用的 GitHub repository。
- DRL 文献中的 fading channel 细节、需求分布实验和 DQN 超参数没有完整代码级披露。
- CMADRL 文献包含 bandwidth allocation agents，而当前实验为了公平控制变量冻结了 bandwidth allocation。
- 当前训练长度只有 $30$ episodes，远低于论文中 offline training 收敛曲线所隐含的训练强度。
- 当前 reward 同时考虑 throughput 与 delay fairness；DRL scheduling 文献主要报告 traffic satisfaction，CMADRL 文献报告 throughput 与 delay fairness，需要后续拆分为两个对应 reward setting。

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
- 第一版 DRL/CMADRL 训练管线已经实现，但当前结果只能证明复现框架可运行，不能证明 learning based 方法已经达到论文级性能。

## 图表

已生成：

- `phase_3_objective_ratio.png`。

## 下一步

- 增加 specialized maximum weight bipartite matching algorithm。
- 增加更长训练的 DRL/CMADRL 实验，并分别报告 offline training time 与 online inference time。
- 为 DRL scheduling 单独设置 traffic satisfaction reward，为 CMADRL 单独设置 throughput delay fairness reward。
- 若后续找到论文作者开源代码，将以开源实现作为优先参考，并保留当前实现作为 controlled reproduction baseline。
