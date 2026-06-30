# Phase 2 更新日志：Beam Hopping 文献筛选

更新时间：2026-06-30

## 检查清单

- [x] 文献目录是否确认。
- [x] 初始候选文章是否列出。
- [x] 筛选标准是否明确。
- [x] 是否完成初筛。
- [x] 是否完成逐篇 formulation 抽取。
- [x] 是否完成逐篇 algorithm 抽取。
- [x] 是否完成逐篇 simulation setting 抽取。
- [x] 是否完成复杂度证据复核。
- [x] 是否完成关键页面文本复核。
- [x] 是否形成最终入选列表。

## 已完成事项

- 已读取 `/Volumes/Research实验/synthesis defense/beam hopping` 下主要 PDF。
- 已用 `pypdf` 抽取 formulation、algorithm、simulation、complexity 相关页面文本。
- 已形成三类文献结论：直接入选、条件入选、暂不纳入。
- 已将 Phase 3 对比场景校准到两类文献参数：
  - DRL/MADRL 风格：$N=19,K=4$；
  - Genetic Algorithm 风格：$N=37,K=10$。
- 已将 OCO、dynamic illumination、cluster hopping、PDLA 与 learnable prediction 文献纳入 Phase 4 方法设计。

## 最终入选文章

- `Beam_Hopping_Scheduling_Based_on_Deep_Reinforcement_Learning.pdf`。
- `Dynamic_Beam_Hopping_of_Multi-beam_Satellite_Based_on_Genetic_Algorithm.pdf`。
- `Dynamic_Beam_Pattern_and_Bandwidth_Allocation_Based_on_Multi-Agent_Deep_Reinforcement_Learning_for_Beam_Hopping_Satellite_Systems.pdf`。
- `An_Online_Convex_Optimization-based_Joint_Beam_Hopping_and_Resource_Allocation_Scheme_for_LEO_Satellite_Networks.pdf`。
- `The_Next_Generation_of_Beam_Hopping_Satellite_Systems_Dynamic_Beam_Illumination_With_Selective_Precoding.pdf`。
- `frsip-01-721682.pdf`。
- `electronics-14-02887-v2.pdf`。

## 条件入选或暂不纳入

- `Multi-Satellite_Beam_Hopping_Based_on_Load_Balancing_and_Interference_Avoidance_for_NGSO_Satellite_Communication_Systems.pdf`：条件入选，主要用于多卫星扩展背景。
- `Joint_Beam_Hopping_and_Carrier_Aggregation_in_High_Throughput_Multibeam_Satellite_Systems.pdf`：暂不作为 Phase 3 核心 benchmark，主要用于 carrier aggregation 和 mixed integer nonlinear programming 背景。
- `Dynamic_Resource_Allocation_for_Beam_Hopping_Satellites_Communication_System_An_Exploration.pdf`：暂不纳入，文本抽取质量不足且与当前 matching 子问题的直接映射不够清晰。
- `978-3-319-76571-6.pdf`：暂不纳入，当前未定位到直接对应 illumination pattern 子问题的章节。

## 对 Phase 3 的影响

Phase 3 现在分为两层：

- provisional baseline：random、greedy、demand priority greedy、genetic algorithm、Linear Programming relaxation、Binary Integer Linear Programming benchmark；
- literature calibrated comparison：基于 DRL/MADRL 的 $N=19,K=4$ 场景和 Genetic Algorithm 的 $N=37,K=10$ 场景进行汇总。

## 对 Phase 4 的影响

Phase 4 采用以下工具路线：

- 从 OCO 文献借鉴 dynamic regret、cumulative constraint violation、long term constraint 的评价方式；
- 从 dynamic illumination 文献借鉴 demand matching、illumination pattern、greedy repair 和 binary relaxation 思路；
- 从 learning augmented 文献借鉴 consistency、robustness、prediction validation 和 primal dual with advice。

## 剩余事项

- 后续若需要论文级引用，需要对关键页面进行图像渲染复核。
- 后续若需要完全复现 DRL/MADRL，需要单独建立训练数据与训练时间记录；当前不把 offline training time 与 online inference time 混合比较。
