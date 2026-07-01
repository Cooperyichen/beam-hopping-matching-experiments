# Phase 3 更新日志：单时隙复杂度与参考算法对比

更新时间：2026 年 6 月 30 日

## 状态结论

Phase 3 已完成第一版本地实验，并在 Phase 2 完成后生成文献校准场景汇总。本次新增了 Deep Reinforcement Learning（DRL）与 Cooperative Multi-Agent Deep Reinforcement Learning（CMADRL）的第一版训练复现管线。当前版本已经可以训练、评估并输出 CSV，但尚未达到论文级完整复现。

## 检查清单

- [x] 对比目标是否定义。
- [x] 控制变量是否定义。
- [x] Phase 1 真实 CSV 是否生成。
- [x] Phase 2 最终入选算法是否确认。
- [x] Benchmark runner 是否实现。
- [x] 第一版参考算法是否实现或适配。
- [x] 图表是否生成。
- [x] 报告是否更新。
- [x] 是否根据 Phase 2 最终入选列表重设参数环境。
- [x] 是否逐篇说明参考文章算法实现途径。
- [x] 是否完成文献校准场景汇总。
- [x] 是否完成第一版 DRL/CMADRL training reproduction。
- [ ] 是否完成论文级 DRL/CMADRL 性能复现。

## 已完成事项

- 生成 `phase_3_single_slot_comparison.csv`。
- 生成 `phase_3_summary.csv`。
- 生成 `phase_3_literature_calibrated_summary.csv`。
- 校准两类场景：
  - DRL/MADRL 风格：$N=19,K=4$；
  - Genetic Algorithm 风格：$N=37,K=10$。
- 新增 `drl_marl_reproduction.py`：
  - 单智能体 DRL 使用 Deep Q-Network（DQN）选择 $K$ 个 illuminated cells；
  - 多智能体版本使用 $K$ 个 independent Q-learning illumination agents，共享同一个 global state 与 global reward；
  - 在当前项目的固定资源设定下冻结 bandwidth allocation，只复现 illumination pattern 决策层。
- 新增 `test_drl_marl_reproduction.py`，覆盖 action space、environment step 与短训练 smoke test。
- 生成 `phase_3_drl_marl_reproduction.csv` 与 `phase_3_drl_marl_reproduction_smoke.csv`。

## 当前限制

- 第一版 DRL/CMADRL 训练已复现，但训练轮数、网络结构、超参数和随机场景尚未达到论文级完整复现。
- 公开网页检索未发现两个目标论文的可直接使用开源仓库，当前实现按论文中的 state、action、reward、simulation scale 和 algorithm structure 贴近复现。
- OCO 类方法主要用于 Phase 4，不作为单时隙核心 baseline。
- Dynamic illumination with selective precoding 含 precoding/interference penalty，不作为 Phase 3 的直接单步 benchmark。
