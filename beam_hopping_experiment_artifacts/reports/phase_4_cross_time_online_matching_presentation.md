# Across Slot Online Matching and Learning Based Scheduling Results

## Key message

The across slot beam hopping problem should be evaluated through two complementary views.

First, horizon level demand coverage checks whether each cell $c_i\in\mathcal{C}$ receives enough service over the full scheduling horizon $\mathcal{T}=\{1,\ldots,T\}$.

Second, queue based reinforcement learning, abbreviated as RL, experiments show how learning based scheduling policies compare with simple traffic and delay heuristics in an online queue environment.

The main result is that residual demand pricing is currently the strongest interpretable online matching baseline. Longer RL training improves the best Dueling Deep Q Network, abbreviated as Dueling DQN, but it still does not dominate urgency based online matching.

---

## Matching oracle and across slot control

At slot $t$, beam $b\in\mathcal{B}$ serving cell $c_i\in\mathcal{C}$ has edge traffic volume

$$
w_t^{b,i}=T_{\mathrm{slot}}\bar R_t^{b,i},
$$

where $T_{\mathrm{slot}}$ is the slot duration and $\bar R_t^{b,i}$ is the achievable rate.

The horizon level demand of cell $c_i$ is

$$
Q_i=\sum_{t=1}^{T}T_{\mathrm{slot}}D_t^i.
$$

The cross time lower bound is

$$
\sum_{t=1}^{T}\sum_{b=1}^{M}T_{\mathrm{slot}}\bar R_t^{b,i}x_t^{b,i}\ge Q_i,\quad \forall c_i\in\mathcal{C}.
$$

Single slot matching remains the fast oracle. The across slot policy changes the edge score before calling the oracle.

---

## Online matching policies

Throughput greedy uses only the instantaneous edge volume:

$$
s_t^{b,i}=w_t^{b,i}.
$$

Urgency primal dual uses residual demand as an online price. Let $d_i(t)$ be the residual demand before slot $t$ and define

$$
u_i(t)=\frac{d_i(t)}{T-t+1}.
$$

With $\bar u(t)$ denoting average urgency, the score is

$$
s_t^{b,i}=w_t^{b,i}\left(1+\frac{u_i(t)}{\bar u(t)}\right).
$$

Prediction aware scheduling mixes realized urgency and predicted urgency:

$$
s_t^{b,i}=w_t^{b,i}\left(1+(1-\theta_t)u_i(t)+\theta_t\widehat u_i(t)\right),
$$

where $\widehat u_i(t)$ is predicted urgency and $\theta_t$ is the prediction trust weight.

---

## Prediction trust weight

The trust weight is

$$
\theta_t=\exp\left(-\kappa\frac{E_t}{A_t}\right),
$$

where $E_t$ is cumulative prediction error, $A_t$ is cumulative realized demand scale, and $\kappa$ controls how quickly the policy discounts inaccurate prediction advice.

Prediction modes:

- Accurate prediction: the forecast follows the realized demand profile.
- Noisy prediction: multiplicative Gaussian noise perturbs the realized demand.
- Adversarial prediction: hotspot location is intentionally mismatched by reversing the cell index order.

This setup tests consistency under accurate prediction, robustness under harmful prediction, and smooth degradation between the two.

---

## Horizon level results after power unit correction

The corrected transmit power is

$$
P_{\mathrm{tot}}=10^{33/10}\,\mathrm{W},
$$

which corresponds to $33\,\mathrm{dBW}$.

Experiment setting:

- $T=20$, $M=10$, $N=37$, $K=10$.
- Sparse feasible beam cell graph.
- Hotspot demand profile.
- Five repetitions for each prediction mode.

| Prediction mode | Policy | Throughput | Unmet demand | Mean satisfaction | Minimum satisfaction |
|---|---|---:|---:|---:|---:|
| Accurate | throughput greedy | $9.467794\times 10^8$ | $1.216995\times 10^8$ | $0.877343$ | $0.225980$ |
| Accurate | urgency primal dual | $9.291981\times 10^8$ | $0$ | $1.000000$ | $1.000000$ |
| Accurate | prediction aware | $9.287851\times 10^8$ | $0$ | $1.000000$ | $1.000000$ |
| Noisy | throughput greedy | $9.462702\times 10^8$ | $9.405220\times 10^7$ | $0.901600$ | $0.378862$ |
| Noisy | urgency primal dual | $9.285650\times 10^8$ | $0$ | $1.000000$ | $1.000000$ |
| Noisy | prediction aware | $9.276082\times 10^8$ | $6.017696\times 10^4$ | $0.999930$ | $0.997413$ |
| Adversarial | throughput greedy | $9.461535\times 10^8$ | $1.230661\times 10^8$ | $0.876963$ | $0.228315$ |
| Adversarial | urgency primal dual | $9.275838\times 10^8$ | $4.427175\times 10^3$ | $0.999996$ | $0.999864$ |
| Adversarial | prediction aware | $9.263470\times 10^8$ | $7.208428\times 10^4$ | $0.999892$ | $0.996012$ |

Observation: throughput greedy maximizes delivered traffic but leaves substantial unmet demand. Urgency primal dual almost completely satisfies demand across all prediction modes.

---

## Queue based unified comparison

The RL comparison uses a queue based environment with $N=19$, $M=4$, $K=4$, evaluation horizon $120$, and five repetitions.

The satisfaction metric in this table is the immediate queue service ratio

$$
\frac{\min(S_t^i,q_t^i)}{q_t^i},
$$

where $S_t^i$ is service volume and $q_t^i$ is queue volume before service. This is not the same metric as horizon level demand coverage.

| Method | Mean throughput Mbit | Mean satisfaction | Mean delay gap slots | Training episodes |
|---|---:|---:|---:|---:|
| delay fairness | $4.333710$ | $0.150642$ | $3.689896$ | $0$ |
| urgency primal dual | $4.840929$ | $0.128402$ | $4.569853$ | $0$ |
| greedy demand | $4.900583$ | $0.117989$ | $4.577311$ | $0$ |
| single agent Dueling DQN | $3.955560$ | $0.116427$ | $4.453314$ | $500$ |
| CMADRL IQL | $3.876750$ | $0.112476$ | $4.503205$ | $500$ |
| prediction aware accurate | $3.484328$ | $0.111724$ | $4.610429$ | $0$ |
| periodic | $3.628205$ | $0.106071$ | $4.459801$ | $0$ |
| random | $3.175564$ | $0.105938$ | $4.474999$ | $0$ |
| improved CMADRL IQL | $2.819727$ | $0.096788$ | $4.541345$ | $500$ |
| single agent DQN | $2.466777$ | $0.093121$ | $4.515633$ | $500$ |
| sequential Dueling DQN | $1.570190$ | $0.074033$ | $4.636476$ | $500$ |
| greedy throughput | $1.267043$ | $0.059806$ | $4.637921$ | $0$ |

Observation: delay fairness gives the best queue satisfaction and delay gap. Urgency primal dual is the best throughput and satisfaction tradeoff among non learning online matching policies.

---

## Heuristic baseline interpretation

Periodic scheduling serves cells in a fixed cyclic order. It is simple and stable, but it ignores queue length, delay, and channel capacity.

Greedy demand serves cells with larger current queue volume. It obtains the highest throughput in the queue based comparison, but it does not minimize delay gap.

Delay fairness serves cells with larger average waiting time. It gives the best mean satisfaction and the smallest delay gap.

Greedy throughput serves cells with the largest instantaneous capacity. In the queue based environment it performs poorly, as it may illuminate high capacity cells that currently have little traffic to serve.

Urgency primal dual combines capacity and residual unmet demand. It is the most useful interpretable online matching baseline for comparing against RL policies.

---

## Training episode length matters

The table compares 80 episode and 500 episode training under the same evaluation setup.

| Method | Episodes | Throughput Mbit | Satisfaction | Final reward | Final loss |
|---|---:|---:|---:|---:|---:|
| single agent DQN | $80$ | $3.086064$ | $0.104139$ | $-22.145200$ | $0.001112$ |
| single agent DQN | $500$ | $2.466777$ | $0.093121$ | $-27.060536$ | $0.005322$ |
| CMADRL IQL | $80$ | $3.828379$ | $0.109169$ | $-21.057559$ | $0.012561$ |
| CMADRL IQL | $500$ | $3.876750$ | $0.112476$ | $-17.971701$ | $0.011745$ |
| single agent Dueling DQN | $80$ | $2.634053$ | $0.096603$ | $-27.172683$ | $0.015177$ |
| single agent Dueling DQN | $500$ | $3.955560$ | $0.116427$ | $-22.572660$ | $0.001233$ |
| sequential Dueling DQN | $80$ | $1.778325$ | $0.078323$ | $-26.855932$ | $71.782499$ |
| sequential Dueling DQN | $500$ | $1.570190$ | $0.074033$ | $-41.067264$ | $0.438049$ |
| improved CMADRL IQL | $80$ | $2.769470$ | $0.097813$ | $-20.523413$ | $0.003545$ |
| improved CMADRL IQL | $500$ | $2.819727$ | $0.096788$ | $-27.525307$ | $0.002960$ |

Observation: single agent Dueling DQN benefits most from longer training. Original single agent DQN and sequential Dueling DQN do not improve under the current reward and action design.

---

## Training loss and reward windows

For 500 episode training, the first window uses episodes $1$ to $20$ and the last window uses episodes $451$ to $500$.

| Method | First reward | Last reward | First loss | Last loss |
|---|---:|---:|---:|---:|
| single agent DQN | $-28.744348$ | $-29.266509$ | $0.002001$ | $0.005467$ |
| CMADRL IQL | $-27.379737$ | $-19.103142$ | $0.008791$ | $0.011838$ |
| single agent Dueling DQN | $-30.186502$ | $-21.392574$ | $0.000684$ | $0.001112$ |
| sequential Dueling DQN | $-29.056048$ | $-41.010440$ | $0.000267$ | $0.544291$ |
| improved CMADRL IQL | $-28.299238$ | $-28.880708$ | $0.002796$ | $0.003267$ |

Observation: reward improvement is more informative than loss alone. The improved CMADRL IQL loss remains small, but its evaluation performance does not improve. Sequential Dueling DQN shows unstable credit assignment.

---

## Takeaways for the next experiment

The next comparison should use one horizon level evaluator for all policies:

$$
\mathrm{Coverage}=\frac{\sum_i\min(Y_i,Q_i)}{\sum_iQ_i},
$$

where $Y_i$ is the total delivered traffic to cell $c_i$.

Priorities:

- Keep urgency primal dual as the main interpretable online matching baseline.
- Report periodic, greedy demand, delay fairness, and greedy throughput together with RL policies.
- Use the 500 episode single agent Dueling DQN as the strongest current RL result.
- Increase load or reduce capacity to make prediction advice more discriminative.
