# Phase 4 更新日志：跨时隙 covering constraint 与 Learning Augmented Online Matching

更新时间：2026-06-30

## 检查清单

- [x] 跨时隙问题是否定义。
- [x] Demand lower bound 角色是否确认。
- [x] Learning Augmented Online Matching 指标是否定义。
- [x] Learning Augmented Online Matching 文献是否逐篇抽取。
- [x] 跨时隙算法原型是否实现。
- [x] prediction accurate/noisy/adversarial 实验是否实现。
- [x] Phase 4 结果 CSV 是否生成。
- [x] Phase 4 summary 是否生成。
- [ ] 与 beam hopping 文献算法是否完成多步复现对比。

## 已完成事项

- 实现 `cross_time_online_matching.py`。
- 实现 `test_cross_time_online_matching.py`。
- 运行 $T=20$、$M=10,N=37,K=10$ 的 sparse hotspot scenario。
- 测试 accurate、noisy、adversarial 三类 prediction mode。
- 比较 throughput greedy、urgency primal dual、prediction aware 三类 policy。

## 当前限制

- 当前原型使用 greedy matching oracle，不是 exact matching oracle。
- 当前没有实现 full OCO solver。
- 当前没有复现 DRL/MADRL 的 training pipeline。

## 下一步

- 增加 exact matching oracle。
- 增加 rolling horizon solver。
- 增加 OCO style dynamic regret。
- 增加 local repair for unmet demand。
