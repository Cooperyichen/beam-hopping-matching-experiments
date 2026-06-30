# Phase 1 报告：单时隙 Beam Hopping Matching 与 Total Unimodularity

## 1. 目标

本阶段只研究单个 time slot $t$ 内的 beam hopping illumination pattern 子问题。目标是证明该子问题可以写成二分图 maximum weight matching，并利用 Total Unimodularity 性质把二元变量松弛为连续变量：

$$
0\le x_t^{b,i}\le 1.
$$

若约束矩阵为 Total Unimodular matrix，且右端项为整数，则线性规划松弛存在整数最优极点。因此，本阶段可以用 Linear Programming relaxation 代替 Binary Integer Linear Programming benchmark。

## 2. 符号约定

沿用原始 beam hopping 模型：

- $\mathcal{B}=\{1,2,\ldots,M\}$ 表示 candidate satellite beams。
- $\mathcal{C}=\{c_i\mid i=1,\ldots,N\}$ 表示 ground cells。
- $K$ 表示单个 slot 内必须 active 的 beam cell pairs 数量。
- $x_t^{b,i}$ 表示 beam $b$ 是否在 slot $t$ 服务 cell $c_i$。
- $T_{\text{slot}}$ 表示 slot duration。
- $\bar R_t^{b,i}$ 表示 edge $(b,c_i)$ 被选择时的 achievable rate。

边权定义为：

$$
w_t^{b,i}=T_{\text{slot}}\bar R_t^{b,i}.
$$

这里 $w_t^{b,i}$ 是理论可交付 traffic volume。它不是 demand，也不是 demand 与 capacity 的 $\min$ 或 $\max$。

Demand 的角色是跨时隙 lower bound：

$$
Q_i=\sum_{t=1}^{T}T_{\text{slot}}D_t^i,
$$

$$
\sum_{t=1}^{T}\sum_{b=1}^{M}T_{\text{slot}}\bar R_t^{b,i}x_t^{b,i}\ge Q_i,\quad \forall c_i\in\mathcal{C}.
$$

因此本阶段不把 $Q_i$ 放入单时隙硬约束。

## 3. 单时隙二分图模型

固定 slot $t$，构造二分图：

$$
G_t=(\mathcal{B},\mathcal{C},\mathcal{E}_t).
$$

若 $(b,c_i)\in\mathcal{E}_t$，则 beam $b$ 可以在 slot $t$ 服务 cell $c_i$。对应决策变量为 $x_t^{b,i}$。

线性规划松弛为：

$$
\max_{\mathbf{x}_t}\quad
\sum_{b=1}^{M}\sum_{i=1}^{N}w_t^{b,i}x_t^{b,i}
$$

subject to

$$
\sum_{i=1}^{N}x_t^{b,i}\le 1,\quad \forall b\in\mathcal{B},
$$

$$
\sum_{b=1}^{M}x_t^{b,i}\le 1,\quad \forall c_i\in\mathcal{C},
$$

$$
\sum_{b=1}^{M}\sum_{i=1}^{N}x_t^{b,i}=K,
$$

$$
0\le x_t^{b,i}\le 1,\quad \forall b\in\mathcal{B},\ c_i\in\mathcal{C}.
$$

Benchmark formulation 把最后一组约束替换为：

$$
x_t^{b,i}\in\{0,1\}.
$$

## 4. Total Unimodularity 解释

二分图的 node edge incidence matrix 是 Total Unimodular matrix。这里每个变量 $x_t^{b,i}$ 对应一条 edge，每列在 beam 侧有一个 $1$，在 cell 侧有一个 $1$。

Beam 侧约束为：

$$
\sum_{i=1}^{N}x_t^{b,i}\le 1.
$$

Cell 侧约束为：

$$
\sum_{b=1}^{M}x_t^{b,i}\le 1.
$$

Cardinality constraint

$$
\sum_{b=1}^{M}\sum_{i=1}^{N}x_t^{b,i}=K
$$

可以由所有 beam 侧 incidence rows 求和得到，因此它不引入新的非 Total Unimodular 结构。

结论是：当 $K$ 和所有右端项为整数时，Linear Programming relaxation 的最优极点解应为整数解。

## 5. 实验设计

当前代码实现以下 topology：

- complete bipartite graph；
- coverage constrained sparse graph；
- random geometric graph。

当前代码实现以下 single slot demand rate distribution，仅用于评价单时隙服务贡献。这里的 $D_t^i$ 是固定 slot $t$ 中 cell $c_i$ 的 demand rate sample，不是跨时隙 arrival sequence，也不是排队模型中的多时隙 arrival process：

- uniform；
- hotspot；
- Poisson sampled single slot demand rate；
- heavy tailed single slot demand rate。

每个实验记录：

- objective value；
- integrality violation；
- MOSEK reported solve time；
- wall time；
- CPU time；
- peak memory；
- variable count；
- approximate constraint count；
- edge density。

## 6. 正确性检查

本地已通过以下测试：

- 边权 $w_t^{b,i}$ 由 achievable rate 生成，而不是由 $D_t^i$ 生成。
- sparse topology 不存在 completely isolated beam 或 completely isolated cell。
- 小规模 complete graph 的 brute force optimum 为有限正值。

## 7. 当前结果状态

MacBook 本地已完成正式第一版实验。实验规模为：

- $M=4,N=19,K=4$；
- $M=10,N=37,K=10$；
- topology 包含 complete、sparse、geometric；
- single slot demand rate distribution 包含 uniform、hotspot；
- 每个配置重复 $3$ 次；
- 每个 instance 同时求解 Linear Programming relaxation 与 Binary Integer Linear Programming benchmark。

结果文件为：

`phase_1_single_slot_results.csv`

汇总结果为：

| Formulation | Rows | Mean objective | Mean wall time | Median wall time | Max integrality violation |
|---|---:|---:|---:|---:|---:|
| Binary Integer Linear Programming | $36$ | $3.807616\times 10^7$ | $0.005929$ s | $0.005393$ s | $6.66\times 10^{-16}$ |
| Linear Programming relaxation | $36$ | $3.807616\times 10^7$ | $0.008059$ s | $0.005116$ s | $3.26\times 10^{-8}$ |

Smoke test 结果摘要如下：

| Instance | Formulation | Objective value | Integrality violation | Status |
|---|---|---:|---:|---|
| $M=4,N=6,K=3$ complete | LP relaxation | $38692507.3168$ | $7.3\times 10^{-13}$ | optimal |
| $M=4,N=6,K=3$ complete | Binary Integer Linear Programming | $38692507.3168$ | $0$ | optimal |
| $M=4,N=6,K=3$ sparse | LP relaxation | $38236026.9333$ | $2.2\times 10^{-10}$ | optimal |
| $M=4,N=6,K=3$ sparse | Binary Integer Linear Programming | $38236026.9364$ | $0$ | optimal |

Linear Programming relaxation 的解在数值精度内为整数。Linear Programming relaxation 与 Binary Integer Linear Programming 的 objective value 差异来自求解器数值容差，relative difference 很小。

## 8. 图表

已生成：

- `phase_1_runtime_boxplot.png`；
- `phase_1_integrality_violation.png`。

## 9. 当前结论

第一版本地实验支持以下判断：

- 单时隙 matching 线性规划松弛产生数值上整数的解。
- Linear Programming relaxation 与 Binary Integer Linear Programming benchmark 目标值一致。
- 当前小中规模下 MOSEK 对 Binary Integer Linear Programming benchmark 也很快，因此单靠小规模实验无法展示明显 runtime advantage。

## 10. 更大规模实验

为检查 relaxation 在更大规模下的表现，额外运行了以下配置：

- $M=32,N=128,K=32$；
- $M=64,N=256,K=64$；
- $M=128,N=512,K=128$；
- topology 包含 complete 和 sparse；
- single slot demand rate distribution 使用 uniform；
- 每个配置运行 $1$ 次 Linear Programming relaxation 和 Binary Integer Linear Programming benchmark。

结果文件为：

`phase_1_large_scale_results.csv`

汇总结果如下：

| Formulation | $M$ | $N$ | $K$ | Variable count | Mean wall time | Mean solver time | Max integrality violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Binary Integer Linear Programming | $32$ | $128$ | $32$ | $4096$ | $0.0437$ s | $0.0338$ s | $4.44\times 10^{-16}$ |
| Linear Programming relaxation | $32$ | $128$ | $32$ | $4096$ | $0.0493$ s | $0.0208$ s | $2.68\times 10^{-11}$ |
| Binary Integer Linear Programming | $64$ | $256$ | $64$ | $16384$ | $0.1511$ s | $0.1235$ s | $2.78\times 10^{-15}$ |
| Linear Programming relaxation | $64$ | $256$ | $64$ | $16384$ | $0.1061$ s | $0.0514$ s | $1.34\times 10^{-9}$ |
| Binary Integer Linear Programming | $128$ | $512$ | $128$ | $65536$ | $0.7325$ s | $0.6070$ s | $6.00\times 10^{-15}$ |
| Linear Programming relaxation | $128$ | $512$ | $128$ | $65536$ | $0.4652$ s | $0.2155$ s | $5.67\times 10^{-10}$ |

此外，额外运行了 $M=256,N=1024,K=256$ 的 Linear Programming relaxation only 扩展实验，变量数为 $262144$。结果文件为：

`phase_1_large_scale_lp_only_results.csv`

| Topology | Variable count | Wall time | Solver time | Integrality violation |
|---|---:|---:|---:|---:|
| complete | $262144$ | $2.3212$ s | $1.3702$ s | $1.08\times 10^{-7}$ |
| sparse | $262144$ | $1.9512$ s | $0.8634$ s | $3.37\times 10^{-9}$ |

## 11. 评估解释

更大规模实验显示，小规模下 Linear Programming relaxation 没有明显优势的现象是合理的，不代表 Total Unimodularity 没有价值。

原因有四点。

第一，当前问题是纯二分图 matching。Binary Integer Linear Programming benchmark 的 root Linear Programming relaxation 已经是 integral polytope，MOSEK 很可能在 root relaxation 处已经得到整数最优解，不需要展开实质性的 branch and bound tree。

第二，小规模实例中 `cvxpy` 建模、矩阵装配、solver 调用等 overhead 与真实 solver time 同量级，会掩盖 Linear Programming relaxation 和 Binary Integer Linear Programming 之间的算法差异。

第三，Binary Integer Linear Programming formulation 没有显式加入 $0\le x_t^{b,i}\le 1$ 这两组连续边界约束，而 Linear Programming relaxation 显式加入了这些约束。因此在通用建模层面，Linear Programming relaxation 的 constraint count 反而更大。

第四，Total Unimodularity 的主要价值不是保证小规模 LP solver 一定快于 MIP solver，而是证明：

$$
\text{Linear Programming relaxation optimum}
=
\text{Binary Integer Linear Programming optimum}.
$$

这允许我们把原本看起来是 integer scheduling 的问题，安全地降为 polynomial time matching 或 Linear Programming 问题。更大规模实验中，solver reported time 已经开始表现出 relaxation 的优势，例如 $M=128,N=512,K=128$ 时，Linear Programming relaxation 的平均 solver time 为 $0.2155$ s，而 Binary Integer Linear Programming benchmark 为 $0.6070$ s。

## 12. 待更新内容

- 加入 runtime scaling 图。
- 加入 specialized maximum weight bipartite matching algorithm，用于展示结构化算法相对于通用 solver 的优势。
