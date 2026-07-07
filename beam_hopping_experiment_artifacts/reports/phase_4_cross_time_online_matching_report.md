# Phase 4 报告：跨时隙 Online Matching、预测增强策略与强化学习对比

## 当前状态

本阶段已经完成两类结果整合。

第一类是跨时隙 online matching 结果，用于验证在恢复 demand satisfaction lower bound 后，如何用 residual demand pricing 和 prediction advice 处理跨时间耦合。

第二类是与 reinforcement learning，简称 RL，和 cooperative multi agent deep reinforcement learning with independent Q learning，简称 CMADRL IQL，相关的统一对比结果。该对比把 `greedy_throughput`、`urgency_primal_dual`、`prediction_aware_accurate`、`periodic`、`greedy_demand`、`delay_fairness`、Deep Q Network，简称 DQN，以及 Dueling DQN 放在同一个 queue based evaluation environment 中进行展示。

本报告中的 beam hopping 记号沿用原始问题定义：

- $\mathcal{B}=\{1,\ldots,M\}$ 表示候选 beams。
- $\mathcal{C}=\{c_i\mid i=1,\ldots,N\}$ 表示 ground cells。
- $\mathcal{T}=\{1,\ldots,T\}$ 表示 scheduling slots。
- $x_t^{b,i}$ 表示 beam $b\in\mathcal{B}$ 在 slot $t\in\mathcal{T}$ 是否服务 cell $c_i\in\mathcal{C}$。
- $T_{\mathrm{slot}}$ 表示 slot duration。
- $\bar R_t^{b,i}$ 表示 beam $b$ 在 slot $t$ 服务 cell $c_i$ 时的 achievable rate。
- $w_t^{b,i}=T_{\mathrm{slot}}\bar R_t^{b,i}$ 表示 edge $(b,c_i)$ 在 slot $t$ 被选择时的可交付 traffic volume。

功率单位已经修正为：

$$
P_{\mathrm{tot}}=10^{33/10}\,\mathrm{W},
$$

即 $33\,\mathrm{dBW}$。此前的 $\frac{10^{33/10}}{1000}\,\mathrm{W}$ 对应 $33\,\mathrm{dBm}$，会显著低估 link rate 与 delivered traffic。

## 跨时隙 demand coupling

cell $c_i$ 在整个 horizon 上的 total demand 定义为：

$$
Q_i=\sum_{t=1}^{T}T_{\mathrm{slot}}D_t^i,
$$

其中 $D_t^i$ 是 cell $c_i$ 在 slot $t$ 的 traffic demand rate。

跨时隙 demand satisfaction lower bound 为：

$$
\sum_{t=1}^{T}\sum_{b=1}^{M}T_{\mathrm{slot}}\bar R_t^{b,i}x_t^{b,i}\ge Q_i,\quad \forall c_i\in\mathcal{C}.
$$

加入该约束后，整体问题不再只是单时隙二分图 matching。单时隙 matching 仍然可以作为 oracle 使用，但跨时间控制需要通过 residual demand、dual price 或 prediction advice 进入每个 slot 的 edge score。

## 跨时隙 online matching 策略

### Throughput greedy

Throughput greedy 直接使用 edge traffic volume 作为 score：

$$
s_t^{b,i}=w_t^{b,i}.
$$

该策略最大化即时可交付 traffic volume，不主动考虑 residual demand。因此它适合用作“只追求当前吞吐”的基线。

### Urgency primal dual

令 $d_i(t)$ 表示 slot $t$ 前 cell $c_i$ 的 residual demand。定义 urgency score：

$$
u_i(t)=\frac{d_i(t)}{T-t+1}.
$$

令 $\bar u(t)$ 表示所有 cells 的 average urgency。edge score 定义为：

$$
s_t^{b,i}=w_t^{b,i}\left(1+\frac{u_i(t)}{\bar u(t)}\right).
$$

该策略可理解为把 residual demand 转换成 online dual price。若某个 cell 的 residual demand 高且剩余 slot 少，它对应的 beam cell edge 会获得更高 priority。

### Prediction aware

令 $\widehat d_i(t)$ 表示 predicted residual demand，令 $\theta_t$ 表示 prediction trust weight。当前实现使用：

$$
\theta_t=\exp\left(-\kappa\frac{E_t}{A_t}\right),
$$

其中 $E_t$ 是累计 prediction error，$A_t$ 是累计 realized demand scale，$\kappa$ 控制 trust weight 的衰减速度。

令 $\widehat u_i(t)$ 表示 predicted urgency。prediction aware score 为：

$$
s_t^{b,i}=w_t^{b,i}\left(1+(1-\theta_t)u_i(t)+\theta_t\widehat u_i(t)\right).
$$

在 accurate prediction 下，$\theta_t$ 较高，策略更依赖 predicted urgency。在 prediction error 增大时，$\theta_t$ 下降，策略逐步回到 realized residual demand pricing。

## 跨时隙结果

实验设置如下：

- $T=20$。
- $M=10$，$N=37$，$K=10$。
- topology 为 sparse feasible beam cell graph。
- demand profile 为 hotspot。
- prediction modes 包括 accurate、noisy 和 adversarial。
- 每个 prediction mode 重复 $5$ 次。

结果文件：

- `phase_4_cross_time_online_matching.csv`
- `phase_4_cross_time_online_matching_summary.csv`

### Accurate prediction

| Policy | Mean throughput | Mean unmet demand | Mean satisfaction | Mean minimum satisfaction |
|---|---:|---:|---:|---:|
| throughput greedy | $9.467794\times 10^8$ | $1.216995\times 10^8$ | $0.877343$ | $0.225980$ |
| urgency primal dual | $9.291981\times 10^8$ | $0$ | $1.000000$ | $1.000000$ |
| prediction aware | $9.287851\times 10^8$ | $0$ | $1.000000$ | $1.000000$ |

### Noisy prediction

| Policy | Mean throughput | Mean unmet demand | Mean satisfaction | Mean minimum satisfaction |
|---|---:|---:|---:|---:|
| throughput greedy | $9.462702\times 10^8$ | $9.405220\times 10^7$ | $0.901600$ | $0.378862$ |
| urgency primal dual | $9.285650\times 10^8$ | $0$ | $1.000000$ | $1.000000$ |
| prediction aware | $9.276082\times 10^8$ | $6.017696\times 10^4$ | $0.999930$ | $0.997413$ |

### Adversarial prediction

| Policy | Mean throughput | Mean unmet demand | Mean satisfaction | Mean minimum satisfaction |
|---|---:|---:|---:|---:|
| throughput greedy | $9.461535\times 10^8$ | $1.230661\times 10^8$ | $0.876963$ | $0.228315$ |
| urgency primal dual | $9.275838\times 10^8$ | $4.427175\times 10^3$ | $0.999996$ | $0.999864$ |
| prediction aware | $9.263470\times 10^8$ | $7.208428\times 10^4$ | $0.999892$ | $0.996012$ |

### 跨时隙结果解释

Throughput greedy 的 total throughput 最高，但 unmet demand 明显最大，minimum satisfaction 明显较低。这说明只最大化即时 throughput 会牺牲 horizon level demand coverage。

Urgency primal dual 在 accurate 和 noisy prediction modes 下完全满足 demand lower bound，在 adversarial prediction mode 下也几乎完全满足 demand lower bound。这说明 residual demand price 是当前跨时隙设置中最稳定的非学习策略。

Prediction aware 在三类 prediction modes 下都接近完全满足 demand lower bound。不过在功率修正后的当前设置中，系统容量相对充足，urgency primal dual 已经足够强，因此 prediction advice 的边际收益不明显。下一步若要展示 prediction aware 的优势，应提高 demand load、降低 capacity demand ratio、移动 hotspot，或者引入 temporal lag prediction error。

Noisy prediction 和 adversarial prediction 是 synthetic stress tests。Noisy prediction 对真实 demand 加入 multiplicative Gaussian noise。Adversarial prediction 通过反转 cell index 故意错配 hotspot location，用于测试 prediction aware policy 在空间预测错误下的 robustness。

## RL 与 heuristic 统一对比

为了和 beam hopping 文献中的 scheduling baselines 对齐，当前又在 queue based RL environment 中整合了以下策略：

- `periodic`：按固定顺序周期性服务 cells，不使用 demand、delay 或 channel 状态。
- `greedy_demand`：优先服务当前 queue volume 更大的 cells。
- `delay_fairness`：优先服务 average delay 更高的 cells。
- `greedy_throughput`：优先服务 per cell instantaneous capacity 最大的 cells。
- `urgency_primal_dual`：使用 residual unmet demand price 调整 throughput score。
- `prediction_aware_accurate`：使用真实 arrival rate vector 作为 accurate expected demand prediction。
- `single_agent_dqn`：单智能体 DQN，动作是一次选择 $K$ 个 cells。
- `single_agent_dueling_dqn`：单智能体 Dueling DQN，使用 expanded state、prioritized experience replay 和 unmet demand reward terms。
- `cmadrl_iql`：CMADRL IQL，每个 active beam 对应一个 independent Q learning agent。
- `cmadrl_iql_improved`：在 CMADRL IQL 上加入 expanded state、Dueling Q Network、prioritized experience replay 和 unmet demand reward terms。
- `sequential_dueling_dqn`：sequential Dueling DQN，每次选择一个 cell，连续选择 $K$ 次。

注意：本节中的 `mean_satisfaction` 是 queue based immediate service ratio：

$$
\frac{\min(S_t^i,q_t^i)}{q_t^i},
$$

其中 $S_t^i$ 表示 slot $t$ 中 cell $c_i$ 获得的 service volume，$q_t^i$ 表示 slot $t$ 前 cell $c_i$ 的 queue volume。该指标不同于上文的 horizon level demand satisfaction ratio。

统一对比结果文件：

- `rl_online_matching_unified_comparison.csv`
- `rl_online_matching_unified_summary.csv`

| Method | Mean throughput Mbit | Mean satisfaction | Mean delay gap slots | Training episodes |
|---|---:|---:|---:|---:|
| `delay_fairness` | $4.333710$ | $0.150642$ | $3.689896$ | $0$ |
| `urgency_primal_dual` | $4.840929$ | $0.128402$ | $4.569853$ | $0$ |
| `greedy_demand` | $4.900583$ | $0.117989$ | $4.577311$ | $0$ |
| `single_agent_dueling_dqn` | $3.955560$ | $0.116427$ | $4.453314$ | $500$ |
| `cmadrl_iql` | $3.876750$ | $0.112476$ | $4.503205$ | $500$ |
| `prediction_aware_accurate` | $3.484328$ | $0.111724$ | $4.610429$ | $0$ |
| `periodic` | $3.628205$ | $0.106071$ | $4.459801$ | $0$ |
| `random` | $3.175564$ | $0.105938$ | $4.474999$ | $0$ |
| `cmadrl_iql_improved` | $2.819727$ | $0.096788$ | $4.541345$ | $500$ |
| `single_agent_dqn` | $2.466777$ | $0.093121$ | $4.515633$ | $500$ |
| `sequential_dueling_dqn` | $1.570190$ | $0.074033$ | $4.636476$ | $500$ |
| `greedy_throughput` | $1.267043$ | $0.059806$ | $4.637921$ | $0$ |

### 统一对比解释

`delay_fairness` 在 mean satisfaction 和 delay gap 上最好，说明在 queue based environment 中，直接控制等待时间比单纯追求 instantaneous throughput 更有利于服务公平性。

`urgency_primal_dual` 的 throughput 排名第二，mean satisfaction 排名第二，是当前最有展示价值的 non learning online matching baseline。它不需要训练，却能明显超过当前 DQN 和 CMADRL IQL 策略。

`greedy_demand` 的 throughput 最高，但 mean satisfaction 低于 `delay_fairness` 和 `urgency_primal_dual`。这说明只看当前 queue volume 可以提高服务量，但不能很好地控制 delay fairness。

`greedy_throughput` 在该环境中表现最差。原因是该策略只看 link capacity，不看当前 queue 是否有 traffic，因此会频繁照亮高容量但当前 queue volume 不足的 cells。

`single_agent_dueling_dqn` 是当前 RL 方法中表现最好的一个，500 episodes 后 throughput 和 satisfaction 都接近 heuristic baselines，但仍未超过 `urgency_primal_dual` 和 `delay_fairness`。

## RL training episode 长度的影响

当前有两组可比较的 training results：80 episodes 和 500 episodes。二者使用相同的 evaluation horizon 和 repetitions。结果文件包括：

- `rl_improvement_80ep_mild.csv`
- `rl_improvement_80ep_mild_training_log.csv`
- `rl_improvement_500ep_mild.csv`
- `rl_improvement_500ep_mild_training_log.csv`

| Method | Episodes | Mean throughput Mbit | Mean satisfaction | Final training reward | Final training loss |
|---|---:|---:|---:|---:|---:|
| `single_agent_dqn` | $80$ | $3.086064$ | $0.104139$ | $-22.145200$ | $0.001112$ |
| `single_agent_dqn` | $500$ | $2.466777$ | $0.093121$ | $-27.060536$ | $0.005322$ |
| `cmadrl_iql` | $80$ | $3.828379$ | $0.109169$ | $-21.057559$ | $0.012561$ |
| `cmadrl_iql` | $500$ | $3.876750$ | $0.112476$ | $-17.971701$ | $0.011745$ |
| `single_agent_dueling_dqn` | $80$ | $2.634053$ | $0.096603$ | $-27.172683$ | $0.015177$ |
| `single_agent_dueling_dqn` | $500$ | $3.955560$ | $0.116427$ | $-22.572660$ | $0.001233$ |
| `sequential_dueling_dqn` | $80$ | $1.778325$ | $0.078323$ | $-26.855932$ | $71.782499$ |
| `sequential_dueling_dqn` | $500$ | $1.570190$ | $0.074033$ | $-41.067264$ | $0.438049$ |
| `cmadrl_iql_improved` | $80$ | $2.769470$ | $0.097813$ | $-20.523413$ | $0.003545$ |
| `cmadrl_iql_improved` | $500$ | $2.819727$ | $0.096788$ | $-27.525307$ | $0.002960$ |

训练日志的窗口平均结果如下。

| Method | Episodes | First window mean reward | Last window mean reward | First window mean loss | Last window mean loss |
|---|---:|---:|---:|---:|---:|
| `single_agent_dqn` | $80$ | $-22.357980$ | $-23.108072$ | $0.002333$ | $0.001684$ |
| `single_agent_dqn` | $500$ | $-28.744348$ | $-29.266509$ | $0.002001$ | $0.005467$ |
| `cmadrl_iql` | $80$ | $-22.263257$ | $-20.895316$ | $0.009242$ | $0.012922$ |
| `cmadrl_iql` | $500$ | $-27.379737$ | $-19.103142$ | $0.008791$ | $0.011838$ |
| `single_agent_dueling_dqn` | $80$ | $-23.940960$ | $-23.968493$ | $0.000724$ | $0.002008$ |
| `single_agent_dueling_dqn` | $500$ | $-30.186502$ | $-21.392574$ | $0.000684$ | $0.001112$ |
| `sequential_dueling_dqn` | $80$ | $-24.015509$ | $-23.800426$ | $0.000254$ | $11.255412$ |
| `sequential_dueling_dqn` | $500$ | $-29.056048$ | $-41.010440$ | $0.000267$ | $0.544291$ |
| `cmadrl_iql_improved` | $80$ | $-22.543756$ | $-22.187053$ | $0.002955$ | $0.003225$ |
| `cmadrl_iql_improved` | $500$ | $-28.299238$ | $-28.880708$ | $0.002796$ | $0.003267$ |

### RL training 解释

`single_agent_dueling_dqn` 是 episode length 增加后改善最明显的方法。从 80 episodes 到 500 episodes，mean throughput 从 $2.634053$ 提高到 $3.955560$，mean satisfaction 从 $0.096603$ 提高到 $0.116427$。训练 reward 的 last window 也从 $-30.186502$ 改善到 $-21.392574$。这说明 expanded state、Dueling DQN 和 prioritized experience replay 在较长训练下开始发挥作用。

`cmadrl_iql` 随训练延长也有改善，但幅度较小。mean satisfaction 从 $0.109169$ 提高到 $0.112476$，last window reward 从 $-27.379737$ 改善到 $-19.103142$。

`single_agent_dqn` 没有从长训练中受益。500 episodes 后 evaluation performance 低于 80 episodes，训练 loss 也没有稳定下降。这说明原始 DQN 的 state 和 reward design 仍不足以支撑稳定学习。

`sequential_dueling_dqn` 当前不稳定。虽然 500 episodes 后 final loss 比 80 episodes 的异常大 loss 小，但 reward 明显恶化，evaluation throughput 和 satisfaction 也下降。该方法需要重新设计 credit assignment 或改用 actor critic 类方法。

`cmadrl_iql_improved` 的 loss 维持较低，但 evaluation reward 和 satisfaction 未改善。这说明低 temporal difference loss 不必然对应更好的 scheduling policy，当前 reward shaping 可能改变了学习目标但没有提升最终调度质量。

## 当前结论

本阶段最清晰的结论有三点。

第一，跨时隙 demand coupling 需要 residual demand aware policy。只看即时 throughput 的策略可能获得更高 total throughput，但会留下明显 unmet demand。

第二，`urgency_primal_dual` 是当前最强的可解释 online matching baseline。它不需要训练，在跨时隙 demand coverage 和 queue based comparison 中都表现稳定。

第三，RL 方法需要更谨慎地设计 state、reward 和 action structure。当前 `single_agent_dueling_dqn` 在长训练后取得明显改善，但仍没有超过 `urgency_primal_dual` 和 `delay_fairness`。因此下一阶段应优先把 RL policy 接入 horizon level demand coverage 指标，而不是只看 queue based immediate satisfaction。

## 下一步

- 在 RL evaluation 中增加 horizon level demand coverage：

$$
\mathrm{Coverage}=\frac{\sum_i\min(Y_i,Q_i)}{\sum_iQ_i},
$$

其中 $Y_i$ 表示 cell $c_i$ 在整个 horizon 上获得的 total delivered traffic。

- 在同一个 horizon level environment 中同时比较 `urgency_primal_dual`、`prediction_aware`、`periodic`、`greedy_demand`、`delay_fairness`、DQN 和 CMADRL IQL。
- 提高 demand load 或降低 capacity demand ratio，使 prediction aware policy 的边际价值更容易被观察。
- 对 `sequential_dueling_dqn` 重新设计 credit assignment，或尝试 proximal policy optimization，简称 PPO，等 actor critic 方法。
