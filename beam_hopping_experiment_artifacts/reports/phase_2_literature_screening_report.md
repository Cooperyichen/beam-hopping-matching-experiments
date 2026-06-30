# Phase 2 报告：Beam Hopping Illumination Pattern 文献筛选

## 1. 目标

本阶段目标是筛选与当前 beam hopping illumination pattern 子问题最接近的参考文章，并判断哪些模型和算法可以进入 Phase 3 与 Phase 4 的对比实验。

当前原始问题的核心变量是：

$$
x_t^{b,i}\in\{0,1\},
$$

其中 $x_t^{b,i}=1$ 表示 beam $b$ 在 slot $t$ 服务 cell $c_i$。本阶段优先选择含有 beam activation、cell selection、time slot assignment、illumination pattern 或 demand satisfaction 结构的文章。

## 2. 筛选标准

文章进入核心对比集合需要满足至少三项：

- 明确建模 beam hopping 或 illumination pattern；
- 含有 binary scheduling variable；
- 目标函数涉及 throughput、traffic satisfaction、delay fairness、load balance 或 demand matching；
- 能在固定 $B_0$、$P_0$、$g_t^{b,i}$、$\sigma_i^2$ 后保留 scheduling 子问题；
- 给出可复现实验参数、复杂度描述、算法结构或运行时间。

## 3. 直接入选文章

### 3.1 Beam Hopping Scheduling Based on Deep Reinforcement Learning

结论：直接入选，用于 Phase 3 的 $N=19,K=4$ 场景和 Phase 4 的 sequential decision 背景。

证据：

- 文章第 1 页明确提出动态匹配 non uniform traffic demands 与 limited satellite beam resources。
- 第 2 页给出 $N$ virtual cells 与每个 slot 最多服务 $K$ cells 的设定。
- 第 3 页解释 constraint $C1$ 限制每个 slot 的最大 beams，$C2$ 定义 binary variable。
- 第 4 页给出仿真设置：$19$ virtual cells，单 slot 最大 $4$ beams，beam bandwidth 为 $500$ MHz。
- 第 4 页使用 Random Beam Hopping、Periodic Beam Hopping 等 benchmark。

与本项目映射：

- 其 cell selection variable 可映射到本项目的 $x_t^{b,i}$，在没有区分 beam index 时可视为每个 active beam 对一个 selected cell 的 assignment。
- 其 traffic satisfaction rate 对应本项目跨时隙评价中的 demand satisfaction ratio。
- Phase 3 中采用 $N=19,K=4$ 作为文献校准场景。

限制：

- 完整 Deep Reinforcement Learning 训练依赖 state、reward、training episodes 和 neural network 细节；当前阶段不复现训练，只复现其 random、periodic、greedy 类可解释 benchmark，并记录其 reported conclusion。

### 3.2 Dynamic Beam Hopping of Multi beam Satellite Based on Genetic Algorithm

结论：直接入选，用于 Phase 3 的 Genetic Algorithm benchmark 与 $N=37,K=10$ 场景。

证据：

- 第 2 页说明使用 Genetic Algorithm 解决 maximizing throughput 与 minimizing delay 的 multi objective problem。
- 第 3 页给出 decision variable $X_t=[x_t^{c_1},\ldots,x_t^{c_N}]$，其中 $x_t^{c_n}\in\{0,1\}$ 表示 cell 是否被选择。
- 第 4 页给出 Genetic Algorithm 流程，包括 population、fitness、crossover、mutation。
- 第 6 页说明数据包由 Poisson distribution 生成，并在 $37$ cells 中选择 $10$ beams。
- 第 6 页报告 Genetic Algorithm 选择了需求最高的 $10$ cells，并与 random scheduling 和 maximum minimum rate control scheduling 比较。

与本项目映射：

- $x_t^{c_n}$ 可视为本项目在 beam 不区分或 beam-cell capacity 同质化下的 cell level version of $x_t^{b,i}$。
- 其 Genetic Algorithm 可转化为在同一个 $\mathcal{E}_t$ 与 $w_t^{b,i}$ 上搜索 $K$ 条不冲突 edge。
- Phase 3 中采用 $N=37,K=10$ 作为文献校准场景。

限制：

- 原文的 GA 是逐个选择 cell，并把已选择 cell 的 traffic demand 置零；本项目适配版需要保持 beam-cell matching feasibility，因此使用 repair operator 保证 beam 与 cell 不冲突。

### 3.3 Dynamic Beam Pattern and Bandwidth Allocation Based on Multi Agent Deep Reinforcement Learning for Beam Hopping Satellite Systems

结论：直接入选，用于 Phase 3 的 complexity comparison 和 Phase 4 的 sequential scheduling 背景。

证据：

- 第 4 页把 dynamic beam hopping 建模为 long term throughput 与 delay fairness 的 multi objective optimization。
- 第 6 页中 agent group $G1$ 负责 beam hopping pattern，action 定义为选择 illuminated cell。
- 第 8 页给出仿真参数：$19$ cells，$4$ beams，$P_{\text{tot}}=33$ dBW，total bandwidth $500$ MHz，bandwidth chunks 为 $4$。
- 第 12 页明确 complexity analysis 是按 one time slot decision 分析。
- 第 13 页给出七种算法的 complexity 与 execution time，并指出 Genetic Algorithm 约 $3.2$ seconds，MADRL online inference 约 $500$ microseconds，DF-BH 与 CKLCQ-BH 约 $200$ microseconds。

与本项目映射：

- 其 beam pattern action 可映射到本项目单 slot selected edge set。
- 其 $N=19,K=4$ 与 Phase 3 小规模场景一致。
- 其 reported execution time 用于解释 learning based method 的 online inference 与 offline training 应分开比较。

限制：

- 原文同时优化 bandwidth allocation。为了公平比较，本项目 Phase 3 固定 $B_0$ 与 $P_0$，只比较 illumination pattern。

### 3.4 An Online Convex Optimization based Joint Beam Hopping and Resource Allocation Scheme for LEO Satellite Networks

结论：直接入选，主要用于 Phase 4。

证据：

- 第 1 页说明该文处理 time varying objectives and constraints，并使用 Online Convex Optimization。
- 第 3 页定义 time varying demand 与 channel 参数，并给出 long term constraint。
- 第 4 页使用 dynamic regret 和 cumulative constraint violation 衡量 online algorithm。
- 第 5 页给出 long term constrained online convex optimization with bandit feedback algorithm。
- 第 6 页展示 per slot throughput、accumulative violation 和 long term dynamic regret。

与本项目映射：

- 其 long term constraint 与本项目跨时隙 demand satisfaction 约束一致：

$$
\sum_{t=1}^{T}\sum_{b=1}^{M}T_{\text{slot}}\bar R_t^{b,i}x_t^{b,i}\ge Q_i.
$$

- 其 cumulative constraint violation 可用于 Phase 4 的 unmet demand 指标。
- 其 dynamic regret 可用于衡量 online matching policy 与 hindsight optimum 的差距。

限制：

- 原文同时处理 association、subcarrier、power allocation，并通过 convexification 处理连续资源。本项目 Phase 4 只借鉴 online constraint handling，不直接复现完整 OCO resource allocation。

### 3.5 The Next Generation of Beam Hopping Satellite Systems: Dynamic Beam Illumination With Selective Precoding

结论：直接入选，用于 Phase 4 的 dynamic illumination、demand matching 和 repair heuristic 背景。

证据：

- 第 3 页说明目标是在给定 time window 中满足 beam demand，并把 dynamic beam illumination 写成 Binary Quadratic Programming。
- 第 9 页给出 Greedy Algorithm，对 relaxed solution 做 binary rounding，并通过 swapping 修复每个 time slot 的 active beam count。
- 第 11 页展示 illumination pattern，并用 per beam demand matching 与 precoding utilization 评价。
- 第 14 页讨论 demand matching 在 channel estimation error 下的变化。

与本项目映射：

- 该文的 illumination matrix 可映射到本项目跨时隙 $x_t^{b,i}$ 的 beam level aggregation。
- 其 greedy repair 与 swapping 方法可用于 Phase 4 的 local repair for unmet demand。
- 其 demand matching 指标可用于评价跨时隙 demand satisfaction。

限制：

- 原文引入 selective precoding 和 inter beam interference penalty，与当前固定 $g_t^{b,i}$ 的单步 matching 不完全一致。Phase 3 不直接用它作单时隙 benchmark。

### 3.6 Precoded Cluster Hopping for Multibeam GEO Satellite Communication Systems

结论：直接入选，用于 Phase 4 的 cluster level illumination pattern 扩展。

证据：

- 第 5 页说明 transmit power per beam 固定且 equally distributed，不优化 power。
- 第 5 页定义目标是获得 optimal illumination pattern，即 snapshots 与 dwelling time，使 demands fairly satisfied。
- 文章关注 cluster hopping，允许 adjacent beams 同时 illuminated。

与本项目映射：

- cluster snapshot 可视为多个 $x_t^{b,i}$ 的 group activation。
- fixed power assumption 与本项目固定 $P_0$ 的控制变量原则一致。

限制：

- 该文以 cluster 为单位，不是单 beam-cell edge matching；因此用于 Phase 4 扩展讨论，不进入 Phase 3 核心 benchmark。

### 3.7 A Framework for Joint Beam Scheduling and Resource Allocation in Beam Hopping Based Satellite Systems

结论：直接入选，用于 Phase 3/4 的 modular algorithm 和 Hungarian matching 参考。

证据：

- 第 7 页给出 binary allocation variable $\alpha_{t,b,f}^i$。
- 第 8 页说明 framework 包含 beam footprint partitioning、time slot based beam scheduling、joint frequency power allocation。
- 第 11 页使用 Hungarian algorithm 解决 user resource assignment。
- 第 15 页和第 16 页给出 computational complexity，并说明 Algorithm C 包含 Hungarian matching、water filling 和 convex optimization。

与本项目映射：

- $\alpha_{t,b,f}^i$ 可视为在加入 frequency index 后的 $x_t^{b,i}$ 扩展变量。
- 其 Hungarian matching 支持本项目 “specialized matching algorithm” 的下一步方向。

限制：

- 原文同时处理 beam placement、frequency、power。本项目需要冻结这些变量，只提取 scheduling 和 assignment 子模块。

## 4. 条件入选文章

### 4.1 Multi Satellite Beam Hopping Based on Load Balancing and Interference Avoidance for NGSO Satellite Communication Systems

结论：条件入选，主要作为多卫星扩展背景。

证据：

- 第 5 页把 load balancing 子问题写成 Mixed Integer Linear Programming，并说明可由 branch and bound 或 MOSEK/CVX 求解。
- 第 6 页把 valid BH pattern 映射为 service vector。
- 第 9 页给出 $206$ cells、每颗 satellite 随机选择 $4$ cells from $37$ cells 等仿真设置。
- 第 12 页给出整体 complexity，并说明复杂度对 satellites 数量近似线性。

限制：

- 多卫星覆盖、interference avoidance、load balancing 与当前单卫星 beam-cell matching 差异较大。后续如研究 multi satellite matching 可纳入。

## 5. 暂不纳入文章

### 5.1 Joint Beam Hopping and Carrier Aggregation in High Throughput Multibeam Satellite Systems

结论：暂不纳入 Phase 3 核心 benchmark。

理由：

- 该文核心变量包括 cluster activation $z_{l,t}$、carrier-user assignment $a_{c,u}^{(l)}$ 和 continuous allocation $\beta_{c,u}^{(l)}$。
- 第 6 页说明问题是 multi objective Mixed Integer Nonlinear Programming。
- 第 7 页需要 linearization product of continuous and integer variables。
- 它适合作为 mixed integer nonlinear extension 背景，不适合作为当前固定 power、固定 bandwidth、单步 matching 的公平 baseline。

### 5.2 Dynamic Resource Allocation for Beam Hopping Satellites Communication System: An Exploration

结论：暂不纳入。

理由：

- 当前文本抽取不足以稳定提取 formulation 和 algorithm details。
- 与当前 matching 子问题的直接变量映射不够清晰。

### 5.3 978-3-319-76571-6.pdf

结论：暂不纳入。

理由：

- 该文件是 proceedings 或书籍合集。
- 当前未定位到直接对应 beam hopping illumination pattern 的章节。

## 6. 对 Phase 3 的最终影响

Phase 3 应采用两类文献校准场景：

1. DRL/MADRL 风格：

$$
N=19,\quad K=4.
$$

2. Genetic Algorithm 风格：

$$
N=37,\quad K=10.
$$

在这两类场景下，本项目比较：

- Linear Programming relaxation from Total Unimodularity；
- Binary Integer Linear Programming benchmark；
- random feasible matching；
- greedy throughput；
- demand priority greedy；
- genetic algorithm。

完整 DRL/MADRL 训练暂不复现，原因是其 offline training cost、network architecture、state definition 和 reward tuning 会引入额外变量。当前只使用其 reported conclusions 和可解释 benchmark。

## 7. 对 Phase 4 的最终影响

Phase 4 采用以下路线：

- 使用 OCO 文献的 dynamic regret 与 cumulative constraint violation 评价跨时隙 online policy；
- 使用 dynamic illumination 文献中的 greedy repair 和 demand matching；
- 使用 learning augmented online algorithms 文献中的 consistency、robustness、smoothness；
- 使用 primal dual learning augmented 方法，把 prediction advice 融入 residual demand price。

## 8. 总结

Phase 2 的最终筛选结果支持当前研究路线：单时隙问题用 Total Unimodular matching oracle；跨时隙问题则不再是纯 matching，需要使用 demand price、rolling horizon、online constraint handling 和 prediction aware policy。
