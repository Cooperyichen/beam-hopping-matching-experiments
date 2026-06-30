# Phase 1 更新日志：单时隙 Total Unimodularity matching 实验

更新时间：2026-06-30

## 检查清单

- [x] 输入是否确认。
- [x] 符号是否与图片一致。
- [x] 建模是否完成。
- [x] 边权是否定义为 theoretical deliverable traffic volume。
- [x] Demand 是否排除在单时隙 edge weight 之外。
- [x] 本地代码初版是否创建。
- [x] 本地单元测试是否通过。
- [x] 代码是否维护在 MacBook 本地。
- [x] MacBook 本地 MOSEK 环境是否可用。
- [x] MacBook 本地实验是否可复现。
- [x] MacBook 本地 smoke test 结果 CSV 是否生成。
- [x] 报告是否根据本地 smoke test 更新。
- [x] Phase 1 正式本地结果 CSV 是否生成。
- [x] Phase 1 图表是否生成。
- [x] 更大规模 Phase 1 结果 CSV 是否生成。
- [x] 单时隙 demand rate distribution 表述是否修正。

## 已完成事项

- 建立 Phase 1 单时隙模型。
- 实现 `single_slot_matching.py`，包含实例生成、topology 生成、LP relaxation 求解、BILP benchmark 求解、CSV 输出。
- 实现 `test_single_slot_matching.py`，检查边权定义、拓扑可行性和 brute force sanity check。
- 本地测试结果：`3 passed`。
- 修复 complete graph 下 infeasible edge mask 为空时 `cvxpy` 不接受空约束的问题。
- MacBook 本地 MOSEK smoke test 已运行，生成 `phase_1_smoke_results.csv`。
- Smoke test 中 LP relaxation integrality violation 分别约为 $7.3\times 10^{-13}$ 与 $2.2\times 10^{-10}$。
- Phase 1 正式本地实验已运行，生成 `phase_1_single_slot_results.csv`，共 $72$ 行。
- LP relaxation 最大整数性误差为 $3.26\times 10^{-8}$。
- Phase 1 图表已生成。
- 更大规模实验已运行，生成 `phase_1_large_scale_results.csv`，覆盖 $4096$、$16384$、$65536$ 个变量。
- 更大规模 LP only 扩展已运行，生成 `phase_1_large_scale_lp_only_results.csv`，覆盖 $262144$ 个变量。
- 报告中已把 “Poisson arrival induced demand” 修正为 “Poisson sampled single slot demand rate”，避免误解为多时隙 arrival sequence。

## 已知问题

- ThinkPad SSH 认证成功，但远端 OpenSSH 执行命令前触发错误 wrapper：

```text
/bin/bash: line 1: -d Ubuntu --exec bash -lc: command not found
```

- 该问题导致 `ssh`、交互式 `ssh -tt`、`scp` 都无法把代码落盘到 ThinkPad。
- 用户已要求改为 MacBook 本地工作，因此该问题不再阻塞当前阶段。

## 下一步

- 增加图表解释文字。
- 与 Phase 3 baseline 结果联动分析。
