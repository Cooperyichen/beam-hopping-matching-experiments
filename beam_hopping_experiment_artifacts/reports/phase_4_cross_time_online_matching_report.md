# Phase 4 报告：跨时隙 Demand Coupling 与 Learning Augmented Online Matching

## 当前状态

本阶段已执行第一版 cross time online matching prototype。该原型用于验证 demand coupling 下的 online matching 工具链，不是完整 DRL/MADRL/OCO 复现。

## 跨时隙约束

原始模型中的 total demand 为：

$$
Q_i=\sum_{t=1}^{T}T_{\text{slot}}D_t^i.
$$

跨时隙 demand satisfaction constraint 为：

$$
\sum_{t=1}^{T}\sum_{b=1}^{M}T_{\text{slot}}\bar R_t^{b,i}x_t^{b,i}\ge Q_i,\quad \forall c_i\in\mathcal{C}.
$$

加入该约束后，整体问题不再只是单时隙二分图 matching。单时隙 matching 子问题仍可作为 oracle 使用。

## 文献工具映射

OCO beam hopping 文献提供了 dynamic regret 与 cumulative constraint violation 作为跨时隙评价指标。本项目当前用 total unmet demand 近似 cumulative constraint violation，用 runtime 和 reoptimization count 衡量在线执行开销。

Dynamic illumination with selective precoding 文献提供了 greedy repair、binary rounding 和 demand matching 思路。本项目后续可把 residual demand highest cells 作为 repair target。

Learning augmented online algorithms 文献提供 consistency、robustness 和 smoothness 三个评价方向。本项目当前用 accurate、noisy、adversarial 三类 prediction mode 来观察 prediction aware policy 的退化行为。

Primal Dual Learning Augmented Algorithms 文献说明 prediction advice 可以通过 primal update rate 或 price bias 融入 online covering 问题。本项目当前用 residual demand price 和 predicted residual demand price 构造 edge score。

## 第一版算法

当前实现了三类 policy。

### Throughput greedy

每个 slot 使用原始 edge weight：

$$
w_t^{b,i}=T_{\text{slot}}\bar R_t^{b,i}.
$$

该方法最大化即时 throughput，不主动考虑 residual demand。

### Urgency primal dual

令 $d_i(t)$ 表示 slot $t$ 前 cell $c_i$ 的 residual demand。定义 urgency score：

$$
u_i(t)=\frac{d_i(t)}{T-t+1}.
$$

edge score 为：

$$
s_t^{b,i}=w_t^{b,i}\left(1+\frac{u_i(t)}{\bar u(t)}\right),
$$

其中 $\bar u(t)$ 是所有 cell urgency 的平均值。

### Prediction aware

令 $\widehat d_i(t)$ 表示 predicted residual demand，令 $\theta_t$ 表示 trust weight。当前采用：

$$
\theta_t=\exp\left(-\kappa\frac{E_t}{A_t}\right),
$$

其中 $E_t$ 是累计 prediction error，$A_t$ 是累计 realized demand scale。edge score 混合 realized urgency 和 predicted urgency：

$$
s_t^{b,i}=w_t^{b,i}\left(1+(1-\theta_t)u_i(t)+\theta_t\widehat u_i(t)\right).
$$

## 第一版实验结果

实验设置：

- $T=20$；
- $M=10,N=37,K=10$；
- topology 为 sparse；
- single slot demand rate distribution 为 hotspot；
- prediction mode 包含 accurate、noisy、adversarial；
- 每个 prediction mode 重复 $5$ 次。

结果文件：

`phase_4_cross_time_online_matching.csv`

汇总文件：

`phase_4_cross_time_online_matching_summary.csv`

### Accurate prediction

| Policy | Mean throughput | Mean unmet demand | Mean satisfaction | Mean minimum satisfaction |
|---|---:|---:|---:|---:|
| throughput greedy | $7.474637\times 10^8$ | $1.852912\times 10^8$ | $0.809120$ | $0.178394$ |
| urgency primal dual | $7.316540\times 10^8$ | $6.991955\times 10^7$ | $0.905483$ | $0.704358$ |
| prediction aware | $7.321555\times 10^8$ | $6.945280\times 10^7$ | $0.907763$ | $0.723175$ |

### Noisy prediction

| Policy | Mean throughput | Mean unmet demand | Mean satisfaction | Mean minimum satisfaction |
|---|---:|---:|---:|---:|
| throughput greedy | $7.469545\times 10^8$ | $1.635997\times 10^8$ | $0.823923$ | $0.299131$ |
| urgency primal dual | $7.314415\times 10^8$ | $6.083213\times 10^7$ | $0.915931$ | $0.758104$ |
| prediction aware | $7.310544\times 10^8$ | $6.691900\times 10^7$ | $0.909955$ | $0.712845$ |

### Adversarial prediction

| Policy | Mean throughput | Mean unmet demand | Mean satisfaction | Mean minimum satisfaction |
|---|---:|---:|---:|---:|
| throughput greedy | $7.468378\times 10^8$ | $1.924831\times 10^8$ | $0.802255$ | $0.180304$ |
| urgency primal dual | $7.315113\times 10^8$ | $7.858045\times 10^7$ | $0.890469$ | $0.616630$ |
| prediction aware | $7.311735\times 10^8$ | $7.996870\times 10^7$ | $0.895088$ | $0.665944$ |

## 当前解释

Throughput greedy 获得最高 total throughput，但 unmet demand 最大，minimum satisfaction 很低。这说明只看即时 throughput 会牺牲 demand coverage。

Urgency primal dual 降低了 total throughput，但显著减少 unmet demand，并提高 minimum satisfaction。这符合 Phase 4 处理 cross time coupling 的目标。

Prediction aware 在 accurate prediction 下略优于 urgency primal dual；在 adversarial prediction 下通过 trust decay 触发 reoptimization，性能没有崩溃。这是 learning augmented online matching 中 robustness 和 smoothness 的初步体现。

## 下一步

- 增加 exact matching oracle。
- 增加 rolling horizon comparison。
- 增加 OCO style dynamic regret。
- 与 beam hopping 文献算法做多步复现对比。
