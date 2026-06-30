# Phase 3 更新日志：单时隙复杂度与参考算法对比

更新时间：2026-06-30

## 状态结论

Phase 3 已完成第一版本地实验，并在 Phase 2 完成后生成文献校准场景汇总。当前仍未复现完整 DRL/MADRL training pipeline，因此 learning based 方法只使用文献结论和可解释 baseline。

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
- [ ] 是否完成完整 DRL/MADRL training reproduction。

## 已完成事项

- 生成 `phase_3_single_slot_comparison.csv`。
- 生成 `phase_3_summary.csv`。
- 生成 `phase_3_literature_calibrated_summary.csv`。
- 校准两类场景：
  - DRL/MADRL 风格：$N=19,K=4$；
  - Genetic Algorithm 风格：$N=37,K=10$。

## 当前限制

- 完整 DRL/MADRL 训练未复现。
- OCO 类方法主要用于 Phase 4，不作为单时隙核心 baseline。
- Dynamic illumination with selective precoding 含 precoding/interference penalty，不作为 Phase 3 的直接单步 benchmark。
