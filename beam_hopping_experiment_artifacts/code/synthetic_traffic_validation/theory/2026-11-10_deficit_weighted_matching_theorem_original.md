# A Conditional Theorem for Matching with Service Deficit Weights

## 1. Purpose and scope

This note establishes a sufficient condition for bounded time-average expected virtual service deficit. The scheduler uses information available before service is realized. It need not observe the forthcoming realization of the channel or demand.

The result combines a conditional quadratic drift argument with a deficit-weighted scheduling objective. With fixed resources and no additional coupling between slots, the scheduling objective separates into maximum-weight bipartite matching problems. Their linear programming relaxations admit integral optimal solutions through Total Unimodularity (TU).

The conclusion concerns cumulative service accounting across consecutive cycles. It does not assert that every individual cycle satisfies its demand, that a prescribed chance constraint holds, or that raw throughput is maximized. The theorem is a sufficient-condition adaptation of the MaxWeight drift method, rather than a claim of a new general stability principle.

## 2. Time, cells, and information

Let $\mathcal C=\{c_i\mid i=1,\ldots,N\}$ be the set of cells and $\mathcal B=\{1,\ldots,M\}$ the set of candidate beams. The positive integers $N$ and $M$ denote their respective cardinalities. Each scheduling cycle contains $T$ slots, indexed by $\mathcal T=\{1,\ldots,T\}$, where $T$ is a positive integer.

Cycles are indexed by $j=1,2,\ldots$. They are consecutive intervals of the actual time axis. Skipped intervals must not be silently removed when interpreting a deficit process as the deficit of the physical system.

All random quantities are defined on a probability space $(\Omega,\mathcal A,\mathbb P)$, and $\mathbb E$ denotes expectation under $\mathbb P$. Let $(\mathcal F_j)_{j\ge1}$ be an increasing sequence of information sigma-algebras. The information $\mathcal F_j$ is available at the beginning of cycle $j$. It may include historical demand and service observations, current deficit values, channel statistics, and demand forecasts.

Let $D_{i,j}\ge0$ denote the true aggregate demand of cell $c_i$ during cycle $j$. It is not required to be known at the beginning of the cycle. Its forecast, when used, is denoted by $\widehat D_{i,j}$ and must be $\mathcal F_j$-measurable.

The expression $\mathbb E[D_{i,j}\mid\mathcal F_j]$ denotes the true conditional mean demand. It is not identified with $\widehat D_{i,j}$ unless a separate assumption justifies that identification. If demand is independent of $\mathcal F_j$, this conditional mean reduces to the unconditional mean. No independence assumption is imposed in the theorem below.

All demand, service, and deficit quantities use bits as their common unit.

## 3. Matching decisions and nominal service

### 3.1 Feasible edges and decisions

For cycle $j$ and slot $t$, let $\mathcal E_{j,t}\subseteq\mathcal B\times\mathcal C$ be the feasible edge set. It is known at the beginning of the cycle and is therefore $\mathcal F_j$-measurable.

Let $K$ be an integer satisfying $1\le K\le\min\{M,N\}$. Assume that every graph under consideration has a matching of cardinality $K$, almost surely.

The variable $x_{j,t}^{b,i}\in\{0,1\}$ indicates whether beam $b$ serves cell $c_i$ in slot $t$ of cycle $j$. The matrix of these variables is $\mathbf x_{j,t}$. Define the finite set of integral feasible matchings

$$
\mathcal X_{j,t}=\left\{
\mathbf x_{j,t}:\begin{array}{l}
\sum_i x_{j,t}^{b,i}\le1,\quad \forall b\in\mathcal B,\\
\sum_b x_{j,t}^{b,i}\le1,\quad \forall c_i\in\mathcal C,\\
\sum_{b,i}x_{j,t}^{b,i}=K,\\
x_{j,t}^{b,i}=0\quad\text{if }(b,c_i)\notin\mathcal E_{j,t},\\
x_{j,t}^{b,i}\in\{0,1\}
\end{array}
\right\}.
$$

A cycle plan is the collection $\mathbf u=(\mathbf x_{j,1},\ldots,\mathbf x_{j,T})$. In the fixed-resource model, its feasible set is

$$
\mathcal U_j=\prod_{t=1}^{T}\mathcal X_{j,t}.
$$

This product structure excludes additional constraints connecting decisions in different slots. The selected plan $\mathbf u_j$ must be $\mathcal F_j$-measurable: it cannot use future realized demand or channel errors. All statements involving random feasible sets are interpreted conditional on the realized information $\mathcal F_j$.

### 3.2 Nominal service coefficients

Let $w_{j,t}^{b,i}\ge0$ denote the nominal service volume, in bits, associated with selecting edge $(b,c_i)$ in slot $t$. The coefficients are finite and $\mathcal F_j$-measurable. They are fixed during optimization of the cycle plan. Demand priorities are not included in $w_{j,t}^{b,i}$.

For a plan $\mathbf u$, define the nominal cumulative service of cell $c_i$ by

$$
m_{i,j}(\mathbf u)
=\sum_{t=1}^{T}\sum_{b\in\mathcal B}
w_{j,t}^{b,i}x_{j,t}^{b,i}.
$$

Let $S_{i,j}(\mathbf u)\ge0$ denote the realized service volume allocated to that cell under the plan. It becomes available for feedback by the beginning of the next cycle.

Here, $S_{i,j}$ is allocated service capacity before capping it by the demand of the current cycle. It may exceed $D_{i,j}$. The useful service for that cycle would instead be $\min\{S_{i,j},D_{i,j}\}$. This distinction matters: the virtual accounting below allows later surplus capacity to compensate for earlier deficits. It is not a model of nontransferable deadline satisfaction.

## 4. Virtual deficit and scheduling score

Let $Z_i^j\ge0$ denote the virtual service deficit of cell $c_i$ at the beginning of cycle $j$, and let $\mathbf Z^j=(Z_1^j,\ldots,Z_N^j)^{\mathsf T}$. Assume that $\mathbf Z^j$ is $\mathcal F_j$-measurable.

Starting from a specified $\mathbf Z^1\ge\mathbf0$, update the deficit using true demand and realized service:

$$
Z_i^{j+1}
=\left[Z_i^j+D_{i,j}-S_{i,j}(\mathbf u_j)\right]^+,
\qquad [z]^+=\max\{z,0\}.
$$

The next information set $\mathcal F_{j+1}$ includes the observations needed for this update. The deficit is not reset at cycle boundaries and is not updated using forecasts in place of observed outcomes.

Define the deficit-weighted score of a candidate plan by

$$
F_j(\mathbf u)=\sum_{i=1}^{N}Z_i^j m_{i,j}(\mathbf u).
$$

Equivalently, the edge weight supplied to the matching problem is

$$
\boxed{\widetilde w_{j,t}^{b,i}=Z_i^j w_{j,t}^{b,i}.}
$$

Thus,

$$
F_j(\mathbf u)
=\sum_{t=1}^{T}\sum_{b\in\mathcal B}\sum_{i=1}^{N}
\widetilde w_{j,t}^{b,i}x_{j,t}^{b,i}.
$$

The nominal service $m_{i,j}$ does not include the deficit multiplier. Applying the multiplier again inside $m_{i,j}$ would count it twice. The score has units of squared bits; it is a scheduling criterion, not a traffic volume.

## 5. Assumptions

All conditional inequalities below hold almost surely for every cycle $j$.

### A1. Valid conditional service lower bound

For every admissible, information-measurable cycle plan,

$$
\mathbb E[S_{i,j}(\mathbf u)\mid\mathcal F_j]
\ge m_{i,j}(\mathbf u),
\qquad i=1,\ldots,N.
$$

This assumption concerns the true conditional distribution given all decision-time information. An unconditional mean or a forecast labelled as a mean does not establish A1 by itself.

### A2. Uniform conditional second-moment bound

There is a deterministic finite constant $C_2\ge0$ such that, for every admissible selected plan,

$$
\frac12\sum_{i=1}^{N}
\mathbb E\!\left[
D_{i,j}^2+S_{i,j}(\mathbf u)^2
\mid\mathcal F_j
\right]\le C_2.
$$

In particular, nonnegativity implies $(D-S)^2\le D^2+S^2$, so the same constant bounds one half of the conditional sum of squared demand–service differences. The separate moment bounds also ensure integrability when the proof separates demand and service expectations. Uniform deterministic bounds on per-cycle demand and service are sufficient for A2.

Also assume that the initial deficit has finite quadratic expectation:

$$
\mathbb E[L(\mathbf Z^1)]<\infty,
\qquad L(\mathbf z)=\frac12\sum_{i=1}^{N}z_i^2.
$$

### A3. Strict conditional service slack

There is a deterministic constant $\varepsilon>0$ such that, for every cycle, there exists an $\mathcal F_j$-measurable probability distribution $p_j$ over the finite plan set $\mathcal U_j$, satisfying

$$
p_j(\mathbf u)\ge0,
\qquad\sum_{\mathbf u\in\mathcal U_j}p_j(\mathbf u)=1,
$$

and

$$
\sum_{\mathbf u\in\mathcal U_j}
p_j(\mathbf u)m_{i,j}(\mathbf u)
\ge
\mathbb E[D_{i,j}\mid\mathcal F_j]+\varepsilon,
\qquad i=1,\ldots,N.
$$

The distribution describes a hypothetical randomized comparison rule. It is chosen using current information and does not select a plan after seeing future outcomes. It is a proof device; the actual algorithm need not execute or explicitly compute it.

Randomization is allowed here so that the comparison can share service opportunities between cells. A3 does not require one deterministic plan to provide a positive service margin to every cell.

Nevertheless, A3 is a strong assumption. It requires slack conditional on every relevant history, rather than only an unconditional average load below capacity. Long correlation, prolonged bursts, or overload can invalidate it. A demand forecast alone does not establish this assumption.

### A4. Uniform additive optimization accuracy

There is a deterministic finite constant $C\ge0$, independent of the current deficit size, such that the chosen feasible plan satisfies

$$
F_j(\mathbf u_j)
\ge
\max_{\mathbf u\in\mathcal U_j}F_j(\mathbf u)-C.
$$

The same objective, coefficients, feasible set, and frozen deficit values must be used when measuring this error. A fixed relative accuracy is not the same as a fixed additive error when deficits can grow.

The constants $C_2$ and $C$ have units of squared bits, and $\varepsilon$ has units of bits per cycle.

## 6. Main theorem

**Theorem.** Under the model and assumptions A1–A4, for every positive integer $J$,

$$
\boxed{
\frac1J\sum_{j=1}^{J}
\mathbb E\!\left[\sum_{i=1}^{N}Z_i^j\right]
\le
\frac{C_2+C}{\varepsilon}
+\frac{\mathbb E[L(\mathbf Z^1)]}{\varepsilon J}.
}
$$

Consequently, the virtual deficit process is strongly stable in the time-average expected sense:

$$
\limsup_{J\to\infty}
\frac1J\sum_{j=1}^{J}
\mathbb E\!\left[\sum_{i=1}^{N}Z_i^j\right]
\le\frac{C_2+C}{\varepsilon}<\infty.
$$

This is a bound on the time average of expected deficit. It does not assert a uniform deterministic bound on all sample paths or a stationary distribution for the process.

## 7. Proof

### Step 1. A scalar inequality

For any $z\ge0$ and any real $v$,

$$
([z+v]^+)^2\le z^2+2zv+v^2.
$$

If $z+v\ge0$, both sides equal $(z+v)^2$. If $z+v<0$, the left side is zero and the right side is the nonnegative quantity $(z+v)^2$.

Substituting $z=Z_i^j$ and $v=D_{i,j}-S_{i,j}(\mathbf u_j)$, summing over cells, and dividing by two gives

$$
\begin{aligned}
L(\mathbf Z^{j+1})-L(\mathbf Z^j)
\le{}&
\frac12\sum_i(D_{i,j}-S_{i,j}(\mathbf u_j))^2\\
&+\sum_i Z_i^j(D_{i,j}-S_{i,j}(\mathbf u_j)).
\end{aligned}
$$

The required expectations are finite. In particular, the scalar inequality together with A2 implies finiteness of the next quadratic deficit expectation from finiteness of the current one; the cross terms are integrable by the Cauchy–Schwarz inequality. This establishes integrability recursively from the initial condition.

### Step 2. Conditional drift

Define the conditional drift by

$$
\Delta_j=
\mathbb E[L(\mathbf Z^{j+1})-L(\mathbf Z^j)\mid\mathcal F_j].
$$

Because the deficits are $\mathcal F_j$-measurable, A2 gives

$$
\Delta_j
\le C_2+
\sum_i Z_i^j\mathbb E[D_{i,j}\mid\mathcal F_j]
-\sum_i Z_i^j\mathbb E[S_{i,j}(\mathbf u_j)\mid\mathcal F_j].
$$

Using A1 and nonnegativity of the deficits,

$$
\Delta_j
\le C_2+
\sum_i Z_i^j\mathbb E[D_{i,j}\mid\mathcal F_j]
-F_j(\mathbf u_j).
$$

No independence between demand, service, and past observations was used in this step.

### Step 3. Compare with the slack rule

The maximum score is at least the score averaged under any probability distribution over feasible plans. Therefore, A4 implies

$$
\begin{aligned}
F_j(\mathbf u_j)
&\ge\max_{\mathbf u\in\mathcal U_j}F_j(\mathbf u)-C\\
&\ge\sum_{\mathbf u\in\mathcal U_j}p_j(\mathbf u)F_j(\mathbf u)-C\\
&=\sum_i Z_i^j
\sum_{\mathbf u\in\mathcal U_j}p_j(\mathbf u)m_{i,j}(\mathbf u)-C\\
&\ge\sum_i Z_i^j
\left(\mathbb E[D_{i,j}\mid\mathcal F_j]+\varepsilon\right)-C,
\end{aligned}
$$

where the last inequality uses A3 and $Z_i^j\ge0$.

Substituting into the drift bound yields

$$
\boxed{\Delta_j\le C_2+C-\varepsilon\sum_iZ_i^j.}
$$

### Step 4. Sum the drift inequalities

The tower property of conditional expectation gives

$$
\mathbb E[\Delta_j]
=\mathbb E[L(\mathbf Z^{j+1})]-\mathbb E[L(\mathbf Z^j)].
$$

Summing for $j=1,\ldots,J$ cancels every intermediate term. Hence

$$
\mathbb E[L(\mathbf Z^{J+1})]-\mathbb E[L(\mathbf Z^1)]
\le J(C_2+C)-\varepsilon\sum_{j=1}^{J}\mathbb E\!\left[\sum_iZ_i^j\right].
$$

Rearranging and using $L\ge0$ gives

$$
\varepsilon\sum_{j=1}^{J}\mathbb E\!\left[\sum_iZ_i^j\right]
\le J(C_2+C)+\mathbb E[L(\mathbf Z^1)].
$$

Division by $\varepsilon J$ proves the finite-$J$ bound. Taking the upper limit as $J$ tends to infinity proves the stated strong stability conclusion. $\square$

## 8. Exact matching and the role of TU

**Corollary.** Suppose the feasible cycle plans have the product structure in Section 3, the nominal service is the stated linear sum, and $Z_i^j$ is held fixed throughout construction of cycle $j$. Solving, for each slot,

$$
\max_{\mathbf x_{j,t}\in\mathcal X_{j,t}}
\sum_{b,i}Z_i^j w_{j,t}^{b,i}x_{j,t}^{b,i}
$$

exactly produces a cycle plan satisfying A4 with $C=0$.

**Proof.** There are no constraints connecting different slots, so maximizing the sum of their objectives equals the sum of their separate maximum values. Thus the assembled plan maximizes $F_j$ over $\mathcal U_j$. $\square$

Each slot problem can be solved through an integral network-flow formulation. Introduce a source, beam nodes, cell nodes, and a sink. Give each source-to-beam, beam-to-cell, and cell-to-sink arc capacity one, retain only feasible beam-to-cell arcs, and require a flow of value $K$. Give beam-to-cell arcs rewards $Z_i^j w_{j,t}^{b,i}$ and all other arcs reward zero.

The directed node–arc incidence matrix is TU. Integer capacities and integer flow balances imply integral extreme points of this bounded flow polytope. Beam-to-cell flows correspond exactly to the slot matching variables; the other arc flows are determined by their row and column sums. Consequently, the matching linear programming relaxation with cardinality $K$ has integral extreme points and admits an integral optimum for these arbitrary real objective coefficients.

This argument uses the integral fixed-flow formulation. It does not rely on the generally invalid claim that adding any linear combination of rows to a TU matrix preserves TU.

An integral optimum exists, but not every optimal point must be integral: convex combinations of tied optimal matchings can also be optimal. The scheduling algorithm must return an integral optimum, for example through an integral matching algorithm or an appropriate extreme-point solution.

TU addresses the inner matching computation. It does not establish A1–A3, and it does not establish the same structure after arbitrary resource or cross-slot constraints are added.

## 9. Further consequences and limitations

### 9.1 Mean rate stability and cumulative net shortfall

Dropping the negative term in the boxed drift inequality and summing gives

$$
\mathbb E[L(\mathbf Z^{J+1})]
\le \mathbb E[L(\mathbf Z^1)]+J(C_2+C).
$$

For every cell, the Cauchy–Schwarz inequality therefore gives

$$
\mathbb E[Z_i^{J+1}]
\le\sqrt{2\mathbb E[L(\mathbf Z^1)]+2J(C_2+C)}.
$$

In particular, $\mathbb E[Z_i^{J+1}]/J\to0$.

The deficit update also implies, on every sample path,

$$
Z_i^{J+1}\ge Z_i^1+\sum_{j=1}^{J}(D_{i,j}-S_{i,j}(\mathbf u_j)).
$$

As $Z_i^1\ge0$, it follows that

$$
\frac1J\mathbb E\!\left[
\left(\sum_{j=1}^{J}D_{i,j}-\sum_{j=1}^{J}S_{i,j}(\mathbf u_j)\right)^+
\right]
\le
\frac{\sqrt{2\mathbb E[L(\mathbf Z^1)]+2J(C_2+C)}}{J}.
$$

This bounds cumulative net shortfall per cycle. It is not a bound on the sum of individual-cycle shortfalls: taking a positive part after summing permits compensation between cycles.

### 9.2 No individual-cycle chance guarantee

The theorem does not imply a bound of the form $\mathbb P(S_{i,j}<D_{i,j})\le\epsilon_i$, where $\epsilon_i\in(0,1)$ is a prescribed violation tolerance.

For example, consider one beam, two cells, one slot per cycle, and exactly one selected edge. Each cell has deterministic demand of one bit per cycle. Either selected edge offers three bits of service. Starting from zero deficits, choose the first cell when the initial weights tie. The deficit sequence then alternates between $(0,1)^{\mathsf T}$ and $(1,0)^{\mathsf T}$, and exact deficit-weighted matching alternates service between the cells.

The deficits remain bounded, while each cell is unserved in approximately half of the cycles. A randomized comparison selecting either cell with probability one half provides average nominal service of $1.5$ bits to each, so A3 holds with $\varepsilon=0.5$ bits per cycle. All quantities are bounded, and A4 holds with $C=0$. This example satisfies the theorem while failing any individual-cell violation target strictly below one half in the long run.

### 9.3 No raw-throughput optimality claim

The algorithm maximizes deficit-weighted service, not unweighted raw throughput. Stabilizing service obligations is a different objective from maximizing the number of bits offered in each cycle. A separate analysis is needed to quantify any raw-throughput tradeoff.

### 9.4 Forecasts, correlated processes, and resource optimization

Forecasts may be included in $\mathcal F_j$, used to propose plans, or used to choose between tied plans. Arbitrarily replacing $Z_i^j$ with a different urgency coefficient is not covered unless A4 for the original score is established.

Correlation across cycles is allowed by the proof, but the conditional slack and moment assumptions must still hold. They are not consequences of correlation being weak on average or of a predictor having a small empirical error. If A3 fails for individual cycles, an analysis over larger windows may be appropriate; the present theorem does not supply that extension automatically.

Joint continuous resource allocation can be included in an enlarged plan definition if A1–A4 are established for that enlarged problem. However, the exact matching corollary only applies to the separable fixed-resource model stated here.

A certified approximate solver can establish A4 by producing a feasible score $\underline F_j$ and a valid upper bound $\overline F_j$ on the optimum of the same frozen problem, with $\overline F_j-\underline F_j\le C$. Solver termination alone, or a score computed for a different objective, does not establish A4.

## 10. Status and theoretical context

The proof above establishes the stated implication from A1–A4. Application to a particular model requires separate verification of those assumptions. The note does not claim independent review or machine-checked formal verification.

The relevant background is the MaxWeight and Lyapunov drift literature. These references motivate the method; the exact conditional assumptions and proof used in this note are stated explicitly above.

1. L. Tassiulas and A. Ephremides, “Stability Properties of Constrained Queueing Systems and Scheduling Policies for Maximum Throughput in Multihop Radio Networks,” IEEE Transactions on Automatic Control, 1992. [Research report and source information](https://drum.lib.umd.edu/items/571fda52-aefb-4497-9a2d-69d8c7c907b9).
2. M. J. Neely, “Notes on Backpressure Routing,” 2011. [Author's lecture notes](https://ee.usc.edu/stochastic-nets/docs/backpressure.pdf).
3. S. Mondal and V. Sharma, “Scheduling Under Fading and Partial Channel Information,” arXiv:1210.7837. [Author manuscript](https://arxiv.org/abs/1210.7837).
