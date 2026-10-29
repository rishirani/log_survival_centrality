# SI Hitting-Time Centrality in Continuous Time

*Working document (new theory direction). Pseudo-proofs only: each proof lists the key steps, not every detail.*

---

## Part A. Main idea

### A.1 What we are doing

The discrete-time paper defines a centrality from the survival probability of a macro-infection hitting time:

- Take an SI epidemic, started at a seed $v$.
- Declare a "macro-infection" once a fraction $\gamma$ of the nodes is infected.
- Define $\delta_t(v)=\mathbb P_v(\text{macro-infection not yet reached by time }t)$ and $C_t(v)=-\log\delta_t(v)$.

A seed with a larger $C_t$ has a survival probability that decays faster, so it reaches macro-infection sooner and is "more central" as a spreader. A seed with a small $C_t$ lingers in small-exit-rate (bottleneck) sets and is slow to cause a macro-infection.

**The replacement.** We replace the discrete-time chain by a **continuous-time Markov chain (CTMC)** on the set of infected sets, with pairwise exponential infection clocks. Everything else (state space, macro set, survival, centrality) keeps its meaning.

### A.2 Motivation: removing $\alpha$

In the discrete paper, per-step infection probabilities are $A_{ij}\in[0,1)$, and the staying probability of an infected set $I$ is
$$P(I,I)=\prod_{i\in I}\prod_{j\notin I}(1-A_{ij}).$$
Taking $-\log$ gives $\sum\sum-\log(1-A_{ij})$, which is **not** the cut weight $\sum\sum A_{ij}$. The paper's limit theorems (degree at the lowest threshold, min-cut at long times) therefore carry error bounds of the form $\tfrac{|V|\alpha^2}{2(1-\alpha)}$ and $\tfrac{|V|^2\alpha^2}{8(1-\alpha)}$, which require $\max A_{ij}\le\alpha$ small.

In continuous time, the probability of "nothing happens" over a window is $e^{-(\text{cut weight})\cdot\text{time}}$, so the exponent is **exactly** the cut weight. Consequently:

- the degree result becomes an **identity**, valid at every horizon;
- the min-cut result becomes an **exact limit**;
- no assumption on small $\alpha$ is needed. Only the product (rate) $\times$ (time) matters.

This also answers the reviewer concern "small $\alpha$ together with fixed $|V|$": small $\alpha$ was a statement about time resolution, not about a low infection rate in a small system.

### A.3 Roadmap

Steps 1-8 below rebuild the discrete paper's theory (closed form, reachability reduction, spectrum, degree limit, min-cut limit) in the CTMC setting, with one new structural lemma (Step 6) that shows the hitting time is a sum of independent exponentials. Step 9 gives exact finite-time formulas (chain form and quasi-polynomial form). Step 10 gives non-asymptotic two-sided bounds and an explicit ranking crossover time. Step 11 gives the second-order asymptotics: the tie correction and the seed-specific constant.

---

## Part B. Variables and assumptions

### B.1 Model

- **Nodes:** $V=\{1,\dots,n\}$.
- **Contact weights:** $A_{ij}\ge0$ (directed, weighted). $A_{ij}>0$ means $i$ can contact $j$. These are weights, not probabilities, so no $A_{ij}<1$ restriction.
- **Self-loop weights:** $A_{ii} = 0$ 
- **Infectivity:** $\beta>0$. Pairwise infection rate $\lambda_{ij}=\beta A_{ij}$.
- **Pairwise clocks (assumption):** while $i$ is infected and $j$ is susceptible, $i$ infects $j$ at constant hazard $\lambda_{ij}$, independently of all other pairs and of the past.
- **SI:** infected nodes never recover, so the infected set only grows.
- **Single stepping (consequence):** independent exponential clocks never ring simultaneously (probability zero), so each event infects **exactly one** node. Multi-node growth over a time window happens only as a sequence of single events.
- **Seed:** the process starts at $X(0)=\{v\}$.

### B.2 Standard facts used

- **Fact A (competing exponentials).** If independent clocks with rates $a_1,\dots,a_r$ run in parallel, the first ring happens after $\mathrm{Exp}(\sum a_k)$ time, clock $k$ rings first with probability $a_k/\sum a$, and the identity of the winner is independent of the time.
- **Fact B (memorylessness / Markov property).** A clock that has not yet rung has the same remaining-time law as a new one. So the holding time in each new state is a fresh exponential, independent of the past.

### B.3 Notation

| Symbol | Meaning |
|---|---|
| $X(s)\subseteq V$ | infected set at time $s$ |
| $\mathcal X=2^V$ | state space |
| $r_j(I)=\sum_{i\in I}\lambda_{ij}$ | rate at which $j\notin I$ gets infected from $I$ |
| $q(I)=\sum_{j\notin I}r_j(I)=\sum_{i\in I}\sum_{j\notin I}\lambda_{ij}$ | total exit rate of $I$ = cut weight of $I$ |
| $M$ | generator in column convention: $M(J,I)=$ rate $I\to J$ |
| $s_\gamma=\lceil\gamma n\rceil$, $m=s_\gamma-1$ | macro size, largest transient size |
| $\mathcal X_\gamma=\{I:\lvert I\rvert\ge s_\gamma\}$ | macro-infection set |
| $\mathcal T_\gamma=\mathcal X\setminus\mathcal X_\gamma$ | transient set |
| $\tau_\gamma$ | macro-hitting time |
| $\delta_t(v)=\mathbb P_v(\tau_\gamma>t)$ | survival probability |
| $C_t(v)=-\log\delta_t(v)$ | log-survival centrality |
| $\mathcal R(\gamma,v)$ | transient sets reachable from $\{v\}$ |
| $M_{\mathcal T},M_{\mathcal R}$ | restrictions of $M$ to $\mathcal T_\gamma$, $\mathcal R$ |
| $\kappa_{\gamma,v}=\min_{I\in\mathcal R}q(I)$ | reachability-constrained min-cut |
| $d_{\rm out}(v)=\sum_jA_{vj}$ | weighted out-degree on contact weights |

### B.4 Standing assumptions

1. **No traps:** every reachable transient set has $q(I)>0$ (the epidemic can always continue). Equivalently $\kappa_{\gamma,v}>0$.
2. **Nontrivial threshold:** $m\ge1$, i.e. $\gamma>1/n$ (otherwise the seed is already macro-infected and $\tau_\gamma=0$).
3. **Column convention:** $M=G^\top$ acts on probability column vectors, matching the paper's $Q^ke_{\{v\}}$. In ascending-size order, $M$ is lower triangular.

---

## Part C. Results, Steps 1-11

---

### Step 1. The chain and its generator

**Definition 1 (infection chain and generator).** The state is the infected set $I\in\mathcal X=2^V$. The generator $M\in\mathbb R^{\mathcal X\times\mathcal X}$ is
$$M(I\cup\{j\},I)=r_j(I)\ \ (j\notin I),\qquad M(I,I)=-q(I),\qquad M(J,I)=0\ \text{otherwise.}$$

*In words:* from $I$, each susceptible $j$ is infected at rate $r_j(I)$, the sum of the clocks of all infected nodes pointing at $j$. The diagonal is minus the total rate of leaving $I$.

**Lemma 1 (structure of $M$).**
1. Off-diagonal entries are $\ge0$ and every column sums to $0$.
2. $M(J,I)\ne0$ with $J\ne I$ implies $J=I\cup\{j\}$, so only single-node jumps occur.
3. In ascending-size order, $M$ is lower triangular, with diagonal $\{-q(I)\}$.
4. $q(I)$ is the cut weight $\sum_{i\in I}\sum_{j\notin I}\lambda_{ij}$.

*What is happening:* because the infected set only grows by one node at a time, every transition moves to the next layer (size $+1$). That is what makes the matrix triangular, and the diagonal carries the exit rates.

*Pseudo-proof.*
1. Each column lists the rates out of one state. The diagonal is defined as minus their sum, so columns sum to zero.
2. Independent exponential clocks ring at distinct times almost surely, so only one node is added per event.
3. Every nonzero off-diagonal entry goes from size $s$ to $s+1$, so it lies strictly below the diagonal in size order.
4. Expand $q(I)=\sum_{j\notin I}\sum_{i\in I}\lambda_{ij}$. $\square$

---

### Step 2. Macro set, hitting time, survival, centrality

**Definition 2 (macro-infection quantities).**
$$\mathcal X_\gamma=\{I:\lvert I\rvert\ge s_\gamma\},\quad \tau_\gamma=\inf\{s\ge0:X(s)\in\mathcal X_\gamma\},$$
$$\delta_t(v)=\mathbb P_v(\tau_\gamma>t),\qquad C_t(v)=-\log\delta_t(v).$$

*In words:* the macro hitting time is the first moment the infection reaches $s_\gamma$ nodes. Survival is the probability that it has not yet happened by time $t$. The centrality is the negative log of that probability.

**Lemma 2 (survival is "still transient"; no overshoot).**
1. $\{\tau_\gamma>t\}=\{X(t)\in\mathcal T_\gamma\}$.
2. $X(\tau_\gamma)$ has size exactly $s_\gamma$.
3. Stopping the chain at $\tau_\gamma$ does not change $\delta_t$.

*What is happening:* since the infected set never shrinks, once the macro set is entered the process stays there. And since each jump adds one node, the macro set is entered exactly at size $s_\gamma$.

*Pseudo-proof.*
1. Monotone growth: if $X(s)\in\mathcal X_\gamma$ for some $s\le t$, then $X(t)\supseteq X(s)$ is also in $\mathcal X_\gamma$. The converse is immediate.
2. Sizes increase by one per jump, so the first size $\ge s_\gamma$ is $s_\gamma$.
3. The event $\{\tau_\gamma>t\}$ depends only on the path before $\tau_\gamma$. $\square$

---

### Step 3. Closed form of survival

**Theorem 1 (closed-form survival).** Let $M_{\mathcal T}$ be $M$ restricted to $\mathcal T_\gamma$ (diagonal keeps the full exit rates $-q(I)$). Then for every seed $v$ and $t\ge0$,
$$\delta_t(v)=\mathbf 1^\top e^{M_{\mathcal T}t}\,e_{\{v\}},\qquad C_t(v)=-\log\bigl(\mathbf 1^\top e^{M_{\mathcal T}t}e_{\{v\}}\bigr).$$

*In words:* the survival probability is the total mass still sitting in transient sets, and the transient mass evolves under a sub-generator, so it is a matrix exponential. This is the paper's Theorem 1 with $Q_\gamma^t$ replaced by $e^{M_{\mathcal T}t}$.

*Pseudo-proof.*
1. Let $p(s)$ be the law of $X(s)$ as a column vector. The forward equation is $p'=Mp$.
2. Order states as (transient, macro). Nothing flows from macro back to transient, so $M=\begin{bmatrix}M_{\mathcal T}&0\\ \ast&\ast\end{bmatrix}$.
3. Hence $p_{\mathcal T}'=M_{\mathcal T}p_{\mathcal T}$ and $p_{\mathcal T}(t)=e^{M_{\mathcal T}t}p_{\mathcal T}(0)$, with $p_{\mathcal T}(0)=e_{\{v\}}$.
4. By Lemma 2, $\delta_t(v)=\mathbf 1^\top p_{\mathcal T}(t)$. $\square$

*Remarks.*
- $e^{M_{\mathcal T}t}$ is nonnegative with column sums $\le1$ (sub-stochastic).
- Connection to the discrete paper: sampling at times $k\Delta$ gives $\delta_{k\Delta}(v)=\mathbf 1^\top(e^{M_{\mathcal T}\Delta})^ke_{\{v\}}$, which is the paper's formula with $Q_\gamma:=e^{M_{\mathcal T}\Delta}$. Its diagonal is $e^{-q(I)\Delta}$. Its off-diagonals are not the paper's frozen-rate synchronous kernel.

---

### Step 4. Reachability reduction

**Definition 3 (reachable transient set).**
$$\mathcal R(\gamma,v)=\{I\in\mathcal T_\gamma:\ \mathbb P_v(X(s)=I\text{ for some }s)>0\}.$$

*In words:* the transient sets that the epidemic can physically pass through starting from $v$. This depends only on the support of $A$, not on $\beta$ or on any time discretization.

**Lemma 3 (graph characterization and closure).**
1. $I\in\mathcal R(\gamma,v)$ iff $\lvert I\rvert\le m$, $v\in I$, and every node of $I$ is reachable from $v$ by a directed path using only nodes of $I$.
2. $\mathcal R$ is closed under transitions inside $\mathcal T_\gamma$: if $I\in\mathcal R$ and $J\in\mathcal T_\gamma$ is reached from $I$ at positive rate, then $J\in\mathcal R$.

*Pseudo-proof.*
1. ($\Rightarrow$) Each added node was infected through an edge from an earlier infected node. Follow these edges back to $v$ inside $I$. ($\Leftarrow$) Order $I$ by a breadth-first search from $v$. Each node has an earlier in-neighbor, so adding nodes in this order is a valid growth path at positive rates.
2. Append the step $I\to J$ to a growth path reaching $I$. $\square$

**Lemma 4 (reachability reduction).** With $M_{\mathcal R}$ the restriction of $M$ to $\mathcal R$ (diagonal still the full $-q(I)$),
$$\delta_t(v)=\mathbf 1^\top e^{M_{\mathcal R}t}e_{\{v\}},\qquad C_t(v)=-\log\bigl(\mathbf 1^\top e^{M_{\mathcal R}t}e_{\{v\}}\bigr).$$

*What is happening:* unreachable transient sets can never receive mass, so they can be deleted from the matrix. This is where seed dependence of the long-time behavior enters.

*Pseudo-proof.*
1. By closure (Lemma 3.2), $M(J,I)=0$ for $I\in\mathcal R$, $J\in\mathcal T_\gamma\setminus\mathcal R$. So $M_{\mathcal T}=\begin{bmatrix}M_{\mathcal R}&\ast\\0&\ast\end{bmatrix}$ in the order $(\mathcal R,\ \mathcal T_\gamma\setminus\mathcal R)$.
2. The first block column of $e^{M_{\mathcal T}t}$ is $\begin{bmatrix}e^{M_{\mathcal R}t}\\0\end{bmatrix}$.
3. $e_{\{v\}}$ lies in the first block, so apply Theorem 1. $\square$

*Remarks.* $M_{\mathcal R}$ is still lower triangular. For $m=1$, $\mathcal R=\{\{v\}\}$. The reachable set is the same notion in the discrete and continuous settings because it only depends on the support of $A$.

---

### Step 5. Spectrum

**Lemma 5 (spectrum is the diagonal).**
$$\operatorname{spec}(M_{\mathcal R})=\{-q(I):I\in\mathcal R(\gamma,v)\}\quad\text{(with multiplicity)}.$$
All eigenvalues are real and $\le0$. Under no traps, they are $<0$.

*What is happening:* every reachable set contributes one decay mode, at its own exit rate. The slowest mode has rate $\kappa_{\gamma,v}=\min_{\mathcal R}q$. This is the continuous replacement of "the spectral radius is the largest diagonal entry".

*Pseudo-proof.*
1. $M_{\mathcal R}$ is lower triangular (Lemma 1.3, restricted).
2. $\det(M_{\mathcal R}-\lambda\mathrm{Id})=\prod_{I\in\mathcal R}(-q(I)-\lambda)$.
3. The roots are $-q(I)\le0$. $\square$

*Remarks.*
- Equal values $q(I)=q(I')$ on sets lying on a common growth path usually produce Jordan blocks, hence factors like $t\,e^{-qt}$. "Spectrum = diagonal" is a statement about eigenvalues, not diagonalizability.
- Unreachable sets would add extra modes to $M_{\mathcal T}$, but $e_{\{v\}}$ never excites them.
- Only monotonicity (the absorbing property of SI) was used.

---

### Step 6. Sum-of-exponentials structure

**Lemma 6 (growth-path decomposition).** Let $\pi=(I_1\subset I_2\subset\dots\subset I_m)$ be the sequence of transient sets visited, $I_1=\{v\}$, $I_{k+1}=I_k\cup\{j_k\}$. Then:
1. $\pi$ has probability $\Pr(\pi)=\prod_{k=1}^{m-1}\dfrac{r_{j_k}(I_k)}{q(I_k)}$.
2. Given $\pi$, $\tau_\gamma=\sum_{k=1}^mE_k$ with independent $E_k\sim\mathrm{Exp}(q(I_k))$.

In particular there are **exactly $m$** holding periods, and
$$\delta_t(v)=\sum_\pi\Pr(\pi)\,\mathbb P\Bigl(\sum_{k=1}^mE_k>t\Bigr).$$

*What is happening:* the epidemic climbs through the layers of sizes $1,2,\dots,m$, spending an exponential time in each. The rate at each layer is the exit rate of the current infected set, so the bottleneck is whichever visited set has the smallest exit rate.

*Pseudo-proof.*
1. Build the chain as a jump chain plus holding times. By Fact A, from $I_k$ the next node is $j$ with probability $r_j(I_k)/q(I_k)$, independently of the holding time.
2. By Fact B, the holding time at $I_k$ is $\mathrm{Exp}(q(I_k))$, independent of the past.
3. Sizes rise by one per jump (Lemma 1.2), so there are exactly $m$ transient holding periods before size $m+1$. The final jump from $I_m$ always lands in $\mathcal X_\gamma$, so contributes no factor to $\Pr(\pi)$.
4. Sum the holding times and average over paths. $\square$

*Remarks.*
- This is the probabilistic mirror of Theorem 1 and Lemma 5: the matrix form groups by state, the path form groups by trajectory, and the rates appearing in the hypoexponential tails are the diagonal entries.
- No traps ensures every $q(I_k)>0$, so all holding times are finite.
- Discrete analogue: holding times are geometric and, because a step may add several nodes, the number of holding periods is random and at most $m$.

---

### Step 7. Lowest threshold: weighted out-degree

**Theorem 2 (degree identity).** Suppose $s_\gamma=2$ (i.e. $\gamma\in(1/n,2/n]$, including $\gamma=2/n$) and $d_{\rm out}(v)>0$. Then for every $t>0$,
$$\delta_t(v)=e^{-\beta d_{\rm out}(v)\,t},\qquad C_t(v)=\beta\,d_{\rm out}(v)\,t,\qquad \frac{C_t(v)}{t}=\beta\,d_{\rm out}(v).$$

*What is happening:* with $s_\gamma=2$ the epidemic is macro-infected as soon as the seed infects anyone. The survival probability is the probability that the seed has not yet fired any of its outgoing clocks, which is an exponential with the seed's total outgoing rate. This is the "first secondary infection" interpretation.

*Pseudo-proof.*
1. $m=1$. The only transient set reachable from $\{v\}$ is $\{v\}$ (every jump leads to size 2, which is macro). So $\mathcal R=\{\{v\}\}$.
2. By Lemma 4, $M_{\mathcal R}=[-q(\{v\})]$, hence $\delta_t=e^{-q(\{v\})t}$. (Equivalently by Lemma 6: one holding period.)
3. $q(\{v\})=\sum_{j\ne v}\lambda_{vj}=\beta\,d_{\rm out}(v)$. $\square$

*Remarks.*
- **No approximation and no $\alpha$.** This replaces the paper's bound $\tfrac{|V|\alpha^2}{2(1-\alpha)}$ by an identity, for every $t$, not only in a limit.
- The ranking of seeds at this threshold is by $d_{\rm out}$ and does not depend on $t$ or $\beta$.
- Only the seed's own row of $A$ matters here.
- In the discrete paper the same computation is exact for the quantity $-\sum_j\log(1-A_{vj})$. The $\alpha$ error only appeared when comparing with $\sum_jA_{vj}$. With rates as the primitives, the comparison is exact.

---

### Step 8. Long-time limit: reachability-constrained min-cut

**Definition 4 (reachability-constrained min-cut).**
$$\kappa_{\gamma,v}=\min_{I\in\mathcal R(\gamma,v)}q(I)=\beta\min_{I\in\mathcal R(\gamma,v)}\ \sum_{i\in I}\sum_{j\notin I}A_{ij}.$$

*In words:* among all infected sets the epidemic can physically pass through, the one with the smallest total contact weight leaving it. A set with a small exit rate is a bottleneck that the infection lingers in.

**Theorem 3 (exact long-time rate).** Under no traps,
$$\lim_{t\to\infty}\frac{C_t(v)}{t}=\kappa_{\gamma,v}.$$
At $s_\gamma=2$, $\kappa_{\gamma,v}=\beta d_{\rm out}(v)$, which agrees with Theorem 2.

*What is happening:* at long times the survival probability is governed by the slowest decaying mode of $M_{\mathcal R}$, which is the smallest exit rate among reachable sets. The proof is the continuous-time version of the discrete paper's "norm bound plus stay-at-the-bottleneck" argument.

*Pseudo-proof.*
1. *Upper bound on survival.* $M_{\mathcal R}$ is triangular with eigenvalues $-q(I)\le-\kappa$ (Lemma 5). Hence $\lVert e^{M_{\mathcal R}t}\rVert\le c\,(1+t)^{|\mathcal R|-1}e^{-\kappa t}$ (the polynomial factor covers Jordan blocks). So $\delta_t\le c'(1+t)^{|\mathcal R|-1}e^{-\kappa t}$, and $\liminf C_t/t\ge\kappa$. (Alternatively, use Lemma 6 and compare each holding time with an $\mathrm{Exp}(\kappa)$ variable.)
2. *Lower bound on survival.* Let $I^\star\in\mathcal R$ attain $\kappa$. Since a positive-rate growth path leads to $I^\star$, the chain is at $I^\star$ at a fixed time $s>0$ with probability $b>0$. Staying there for the remaining time has probability $e^{-\kappa(t-s)}$ (Fact B). So $\delta_t\ge b\,e^{-\kappa(t-s)}$, and $\limsup C_t/t\le\kappa$.
3. Combine. $\square$

*Remarks.*
- **Corollary 1 of the discrete paper is absorbed.** The $-\log(1-A_{ij})$ weights are exactly the rates, so no separate statement is needed. The paper's $A$-weighted min-cut is recovered only approximately, with the $\alpha$ bound, and exactly under $\lambda_{ij}=\beta A_{ij}$ with $A$ as contact weights.
- $\kappa_{\gamma,v}$ is non-increasing in $\gamma$, and $\kappa_{\gamma,v}\le\beta d_{\rm out}(v)$ because $\{v\}$ is always reachable.
- Seed dependence enters only through which sets are reachable. At high $\gamma$ many seeds share nearly the same $\kappa$. The finer seed-specific information is then in the constant term of the asymptotics (Step 11).
- Several minimizers on one growth path change the subexponential correction ($t^{k-1}$ factors, Step 11) but not this limit.
- $\beta$ is a pure time scale: $C_t(v;\beta)=C_{\beta t}(v;1)$, so the $\kappa$-ranking is independent of $\beta$.

---

### Step 9. Exact finite-time formulas

Steps 3 to 6 already give exact formulas for the survival probability. This step adds two more, and puts all four side by side. All four hold for **every** $t\ge0$, with no approximation and no limit.

| Form | Where | Shape |
|---|---|---|
| E1, matrix | Theorem 1, Lemma 4 | $\mathbf 1^\top e^{M_{\mathcal R}t}e_{\{v\}}$ |
| E3, full path | Lemma 6 | $\sum_\pi\Pr(\pi)\,\mathbb P\bigl(\sum_{k=1}^mE_k>t\bigr)$ |
| E2, chain | Lemma 7 (below) | $\sum_cw(c)\,h_c(t)$, all terms positive |
| E4, quasi-polynomial | Theorem 4 (below) | $\sum_ae^{-\theta_at}P_a(t)$, explicit |

**Definition 5 (growth chain, weight, chain kernel).** A *growth chain* is a sequence of sets
$$c=(K_0\subset K_1\subset\dots\subset K_l),\qquad K_0=\{v\},\quad K_{i+1}=K_i\cup\{j_i\},\quad r_{j_i}(K_i)>0,\quad |K_l|\le m,$$
with $0\le l\le m-1$. Write $q_i=q(K_i)$. Its **weight** is $w(c)=\prod_{i=0}^{l-1}r_{j_i}(K_i)>0$ (with $w(c)=1$ if $l=0$). Its **kernel** is the convolution on $[0,\infty)$
$$h_c(t)=\bigl(e^{-q_0\,\cdot}*e^{-q_1\,\cdot}*\dots*e^{-q_l\,\cdot}\bigr)(t).$$
Every $K_i$ lies in $\mathcal R$ (Lemma 3), and there are finitely many chains.

*In words:* a chain is a possible partial history of the epidemic, listing the infected sets in the order they appear, stopping anywhere below the macro threshold. The weight multiplies the rates of the infections used. The kernel $h_c$ accounts for the time spent in the sets of the chain.

**Lemma 7 (chain representation).**
1. For every chain $c$ and every $t\ge0$, $w(c)\,h_c(t)=\mathbb P_v\bigl(\text{the sets visited so far form }c\text{ and }X(t)=K_l\bigr)$.
2. For every $I\in\mathcal R$, the visit probability is $p_I:=\mathbb P_v(X\text{ visits }I)=\sum_{c:\,K_l=I}\dfrac{w(c)}{\prod_{i<l}q_i}$.
3. For every $t\ge0$,
$$\delta_t(v)=\sum_cw(c)\,h_c(t).$$

*What is happening:* at time $t$ the epidemic sits in exactly one transient set. Group by the history that led there. The weight records which infections happened, and the convolution records how the time was spent.

*Proof.*
1. Fix $c$. By Facts A and B, the first $l$ jumps follow $c$ with probability $\prod_{i<l}\frac{r_{j_i}(K_i)}{q_i}=\frac{w(c)}{\prod_{i<l}q_i}$. Given this, the holding times $E_0,\dots,E_l$ at $K_0,\dots,K_l$ are independent with $E_i\sim\mathrm{Exp}(q_i)$.
2. The chain is followed and $X(t)=K_l$ iff $E_0+\dots+E_{l-1}\le t<E_0+\dots+E_l$. The probability of this, given the chain, is
$$\int_{s_0+\dots+s_{l-1}\le t}\ \prod_{i<l}q_ie^{-q_is_i}\cdot e^{-q_l(t-\sum_{i<l}s_i)}\,ds=\Bigl(\prod_{i<l}q_i\Bigr)h_c(t).$$
3. Multiply the two factors. The $\prod_{i<l}q_i$ cancel, leaving $w(c)h_c(t)$. This is part 1.
4. Part 2: the epidemic visits $I$ iff it follows some chain ending at $I$. Chains are distinct histories, so these events are disjoint. Add the probabilities from step 1.
5. Part 3: by Lemma 4, $\delta_t=\sum_{J\in\mathcal R}\mathbb P_v(X(t)=J)$. Each term splits over the chains ending at $J$, and part 1 applies. $\square$

*Remark.* Part 3 is the Neumann series of $(s\,\mathrm{Id}-M_{\mathcal R})^{-1}$ in the time domain: write $M_{\mathcal R}=D+N$ with $D$ diagonal and $N$ strictly lower, so $N$ is nilpotent and the series is finite. In the Laplace domain, $\hat\delta(s)=\sum_cw(c)\prod_i\frac1{s+q_i}$.

**Theorem 4 (exact exponential-polynomial form).** Let $\theta_1<\theta_2<\dots<\theta_r$ be the distinct values of $q$ on $\mathcal R$. Then $\theta_1=\kappa=\kappa_{\gamma,v}$, and there are real polynomials $P_1,\dots,P_r$ such that for **every** $t\ge0$
$$\delta_t(v)=\sum_{a=1}^re^{-\theta_at}P_a(t).$$
Moreover:
1. *Degrees.* $\deg P_a\le n_a-1$, where $n_a$ is the largest number of sets with $q=\theta_a$ on a single chain. In particular $\deg P_a\le m-1$.
2. *Coefficients.* For a chain $c$ let $\mu_{c,b}=\#\{i:\ q_i=\theta_b\}$. Then
$$P_a(t)=\sum_cw(c)\sum_{j=1}^{\mu_{c,a}}A^{(c)}_{a,j}\,\frac{t^{j-1}}{(j-1)!},\qquad A^{(c)}_{a,j}=\frac1{(\mu_{c,a}-j)!}\,\frac{d^{\,\mu_{c,a}-j}}{ds^{\,\mu_{c,a}-j}}\Bigl[\prod_{b\ne a}(s+\theta_b)^{-\mu_{c,b}}\Bigr]_{s=-\theta_a},$$
where the sum over $c$ runs over chains with $\mu_{c,a}\ge1$ and the product over values $b$ that occur on $c$.
3. *Distinct rates.* If the rates $q_0,\dots,q_l$ of a chain are pairwise distinct, then $h_c(t)=\sum_{i=0}^le^{-q_it}\prod_{j\ne i}\frac1{q_j-q_i}$.
4. *Slowest rate.* $n_1=d_v:=\max_ck_c$, where $k_c=\mu_{c,1}$ is the number of minimizers on $c$. The polynomial $P_1$ has degree exactly $d_v-1$ and leading coefficient
$$a_v=\frac1{(d_v-1)!}\sum_{c:\,k_c=d_v}w(c)\,\pi_c>0,\qquad\pi_c=\prod_{i:\,q_i>\kappa}\frac1{q_i-\kappa}.$$

*What is happening:* the survival probability is a finite sum of decaying exponentials, one for each distinct exit rate, each multiplied by a polynomial. This is the spectral picture of Step 5 made explicit: the exponents are the eigenvalues $-q(I)$, and the polynomial degrees are the Jordan-block sizes seen by the seed.

*Proof.*
1. By Lemma 7.3, $\delta_t=\sum_cw(c)h_c(t)$, a finite sum. It suffices to expand each $h_c$.
2. Fix a chain $c$ with distinct rate values $\theta_b$ of multiplicity $\mu_b=\mu_{c,b}\ge1$, so $\sum_b\mu_b=l+1$. The function $e^{-qt}$ has Laplace transform $\frac1{s+q}$, so by the convolution theorem $\hat h_c(s)=\prod_i\frac1{s+q_i}=\prod_b(s+\theta_b)^{-\mu_b}$ for $\mathrm{Re}\,s>-\kappa$.
3. This is a proper rational function with poles at $-\theta_b$ of order $\mu_b$. Partial fractions give $\hat h_c(s)=\sum_b\sum_{j=1}^{\mu_b}\frac{A_{b,j}}{(s+\theta_b)^j}$, with $A_{b,j}=\frac1{(\mu_b-j)!}\frac{d^{\mu_b-j}}{ds^{\mu_b-j}}\bigl[(s+\theta_b)^{\mu_b}\hat h_c(s)\bigr]_{s=-\theta_b}$. Note $(s+\theta_b)^{\mu_b}\hat h_c(s)=\prod_{b'\ne b}(s+\theta_{b'})^{-\mu_{b'}}$.
4. The function $\frac{t^{j-1}}{(j-1)!}e^{-\theta t}$ has transform $\frac1{(s+\theta)^j}$. The function $h_c$ is continuous, and a continuous function is determined by its Laplace transform (Lerch's theorem). Hence for all $t\ge0$
$$h_c(t)=\sum_b\sum_{j=1}^{\mu_b}A_{b,j}\frac{t^{j-1}}{(j-1)!}e^{-\theta_bt}.$$
5. Multiply by $w(c)$, sum over $c$, and collect terms with the same exponential $e^{-\theta_at}$. This defines $P_a$ and gives part 2. The degree of the $c$-term is at most $\mu_{c,a}-1\le n_a-1$. A chain has at most $m$ sets, so $n_a\le m$.
6. Part 3: all $\mu_b=1$ and $A_{b,1}=\prod_{b'\ne b}\frac1{\theta_{b'}-\theta_b}$.
7. Part 4. The minimum value of $q$ on a chain is at least $\kappa=\theta_1$, so $n_1=\max_c\mu_{c,1}=\max_ck_c=d_v$. For a chain with $k=k_c\ge1$, set $G(s)=\prod_{i:\,q_i>\kappa}\frac1{s+q_i}$. Then $A_{1,k}=G(-\kappa)=\pi_c>0$, so this chain contributes $w(c)\pi_c\frac{t^{k-1}}{(k-1)!}$ as its top-degree term to $P_1$. Chains with $k_c<d_v$ have degree $<d_v-1$ and do not affect the coefficient of $t^{d_v-1}$. Chains with $k_c=d_v$ contribute **positive** amounts to it, so nothing cancels, and the coefficient is $a_v>0$. $\square$

**Worked check (four nodes, seed 1).** Take $\lambda_{12}=1,\lambda_{13}=2,\lambda_{24}=0.5,\lambda_{34}=1$ and $\gamma=0.75$, so $m=2$. Then $q(\{1\})=3$, $q(\{1,2\})=2.5$, $q(\{1,3\})=2$. The chains are:

| chain | $w$ | rates | $h_c$ |
|---|---|---|---|
| $(\{1\})$ | $1$ | $3$ | $e^{-3t}$ |
| $(\{1\},\{1,2\})$ | $1$ | $3,\,2.5$ | $2e^{-2.5t}-2e^{-3t}$ |
| $(\{1\},\{1,3\})$ | $2$ | $3,\,2$ | $e^{-2t}-e^{-3t}$ |

So $\delta_t=2e^{-2t}+2e^{-2.5t}-3e^{-3t}$, with $P_1\equiv2=a_v$, $d_v=1$.

---

### Step 10. Finite-time bounds and ranking

**Definition 6 (minimizers, visit probabilities, gap).** Let $\kappa=\kappa_{\gamma,v}$.
- $\mathcal M=\{I\in\mathcal R:\ q(I)=\kappa\}$ is the set of **minimizers**.
- $p_I=\mathbb P_v(X\text{ visits }I)>0$ for $I\in\mathcal R$ (positive by Definition 3), given exactly by Lemma 7.2. Put $p^\star_v=\max_{I\in\mathcal M}p_I$.
- The **gap** is $g=\min\{q(I)-\kappa:\ I\in\mathcal R,\ q(I)>\kappa\}>0$, with $g=+\infty$ if there is no such $I$.
- $S_m(x)=\sum_{k=0}^{m-1}\frac{x^k}{k!}$.

**Theorem 5 (sandwich).** Under no traps, for every $t\ge0$:
1. For every $I\in\mathcal R$: $\ \delta_t(v)\ge p_I\,e^{-q(I)t}$. In particular $\delta_t(v)\ge p^\star_v\,e^{-\kappa t}$.
2. $\delta_t(v)\le e^{-\kappa t}S_m(\kappa t)$.

*What is happening:* the lower bound says that if the epidemic happens to enter a bottleneck, it stays there for an exponential time at rate $\kappa$ at least with the stated probability. The upper bound says that every holding stage is at least as fast as a rate-$\kappa$ stage, so the total time is stochastically smaller than an Erlang time of $m$ rate-$\kappa$ stages.

*Proof of part 1.*
1. Fix $I\in\mathcal R$. Let $T_I$ be the arrival time at $I$ ($=\infty$ if never visited) and $H_I$ the holding time there. On $\{X\text{ visits }I\}$, the process is in a set contained in $I$ at all times before $T_I$ (monotone growth and every earlier state on the way to $I$ is a subset of $I$), and equals $I$ on $[T_I,T_I+H_I)$.
2. All these sets have size $\le|I|\le m$, so they are transient. By Lemma 2.1, $X(t)\in\mathcal T_\gamma$ means $\tau_\gamma>t$. Hence $\{X\text{ visits }I,\ T_I+H_I>t\}\subseteq\{\tau_\gamma>t\}$.
3. Since $T_I\ge0$, $\{H_I>t\}\subseteq\{T_I+H_I>t\}$ on the event of visiting $I$.
4. $T_I$ is a stopping time. By the strong Markov property and Fact B, given the visit, $H_I\sim\mathrm{Exp}(q(I))$ independent of the past. So $\mathbb P(\text{visit }I,\ H_I>t)=p_Ie^{-q(I)t}$.
5. Combine steps 2 to 4. Take $I$ a minimizer with $p_I=p^\star_v$ for the last claim.

*Proof of part 2.*
1. By Lemma 6, $\delta_t=\sum_\pi\Pr(\pi)\,\mathbb P\bigl(\sum_{k=1}^mE_k>t\bigr)$, where given $\pi$ the $E_k\sim\mathrm{Exp}(q(I_k))$ are independent, and $\sum_\pi\Pr(\pi)=1$ (each step selects one node with probability $r_j/q$, and no traps ensures $q>0$).
2. Fix $\pi$. Every $I_k\in\mathcal R$, so $q(I_k)\ge\kappa$. Let $E'_1,\dots,E'_m$ be i.i.d. $\mathrm{Exp}(\kappa)$ and set $E_k=\frac{\kappa}{q(I_k)}E'_k$. If $E'\sim\mathrm{Exp}(\kappa)$ then $cE'\sim\mathrm{Exp}(\kappa/c)$, so $E_k\sim\mathrm{Exp}(q(I_k))$, independent over $k$. Since $\frac\kappa{q(I_k)}\le1$, $E_k\le E'_k$ pointwise.
3. Hence $\sum_kE_k\le\sum_kE'_k\sim\mathrm{Erlang}(m,\kappa)$, and $\mathbb P(\sum_kE_k>t)\le\mathbb P(\mathrm{Erlang}(m,\kappa)>t)$.
4. The Erlang tail is the probability that a rate-$\kappa$ Poisson process has fewer than $m$ events by time $t$, namely $e^{-\kappa t}S_m(\kappa t)$.
5. The bound does not depend on $\pi$. Average over $\pi$. $\square$

**Corollary 1 (log bounds, slope, order).** Under no traps:
1. For all $t\ge0$: $\ \kappa t-\log S_m(\kappa t)\ \le\ C_t(v)\ \le\ \kappa t-\log p^\star_v$.
2. For $\kappa t\ge1$: $\ -\log m-(m-1)\log(\kappa t)\ \le\ C_t(v)-\kappa t\ \le\ -\log p^\star_v$.
3. $\displaystyle\lim_{t\to\infty}\frac{C_t(v)}{t}=\kappa$, and $C_t(v)=\kappa t+O(\log t)$.
4. If $m=1$, then $S_1\equiv1$, $p^\star_v=1$, and $C_t(v)=\kappa t$ exactly, which is Theorem 2.

*Proof.* (1) Take $-\log$ of Theorem 5. (2) For $x\ge1$ and $0\le k\le m-1$, $x^k/k!\le x^{m-1}$, so $S_m(x)\le mx^{m-1}$. Apply this with $x=\kappa t$. (3) Divide (2) by $t$ and let $t\to\infty$. (4) With $m=1$, the only reachable set is $\{v\}$, $\kappa=q(\{v\})$ and both bounds equal $e^{-\kappa t}$. $\square$

*Remark.* Part 3 is a second proof of Theorem 3, with the explicit logarithmic error. The log-slack on the lower side is real but conservative: Step 11 shows that for a unique minimizer the true offset is $O(1)$. The upper bound is attained when every reachable set is a minimizer (the directed path with equal rates gives $\delta_t=e^{-\kappa t}S_m(\kappa t)$ exactly).

**Corollary 2 (ranking crossover).** Let $u,v$ be two seeds with the same $\gamma$ (hence the same $m$), with $\kappa_v>\kappa_u>0$. Put $\Delta\kappa=\kappa_v-\kappa_u$ and
$$\varphi(t)=\Delta\kappa\,t-\log S_m(\kappa_vt)+\log p^\star_u .$$
Then $\varphi(t)>0$ implies $C_t(v)>C_t(u)$. Moreover $\varphi(t)\to\infty$, so there is a finite $t_0$ with $C_t(v)>C_t(u)$ for all $t>t_0$ (take $t_0=\sup\{t:\varphi(t)\le0\}$). For $m\ge2$ an explicit sufficient time is
$$t_0=\max\Bigl\{\frac1{\kappa_v},\ \frac2{\Delta\kappa}\Bigl[\log\frac m{p^\star_u}-(m-1)\Bigl(1+\log\frac{\Delta\kappa}{2(m-1)\kappa_v}\Bigr)\Bigr]\Bigr\}.$$
For $m=1$ the inequality holds for every $t>0$.

*What is happening:* the more central seed (larger $\kappa$) wins once its guaranteed linear gain $\Delta\kappa\,t$ beats the logarithmic slack in the two bounds.

*Proof.*
1. Corollary 1.1 for $v$ (lower bound on $C_t$) gives $C_t(v)\ge\kappa_vt-\log S_m(\kappa_vt)$. Corollary 1.1 for $u$ (upper bound) gives $C_t(u)\le\kappa_ut-\log p^\star_u$.
2. Subtract: $C_t(v)-C_t(u)\ge\varphi(t)$.
3. $\varphi$ is continuous. For $\kappa_vt\ge1$, $\log S_m(\kappa_vt)\le\log m+(m-1)\log(\kappa_vt)$, so $\varphi(t)\ge\Delta\kappa\,t-(m-1)\log(\kappa_vt)-\log(m/p^\star_u)\to\infty$.
4. Explicit $t_0$ ($m\ge2$): use $\log y\le ay-1-\log a$ for $y,a>0$ (because $\log(ay)\le ay-1$), with $y=\kappa_vt$ and $a=\frac{\Delta\kappa}{2(m-1)\kappa_v}$. Then $(m-1)\log(\kappa_vt)\le\frac{\Delta\kappa}2t-(m-1)(1+\log a)$. So for $\kappa_vt\ge1$,
$$\varphi(t)\ge\frac{\Delta\kappa}2\,t-\log\frac m{p^\star_u}+(m-1)(1+\log a),$$
which is positive when $t$ exceeds the bracketed term of the formula.
5. $m=1$: $C_t=\kappa t$ for both seeds (Corollary 1.4), so $C_t(v)-C_t(u)=\Delta\kappa\,t>0$. $\square$

*Remarks.*
- The condition is **sufficient**, not sharp. It uses only the worst-case rate $\kappa$ and the single-bottleneck probability $p^\star_u$.
- $p^\star_u$ can be replaced by any lower bound, for instance the probability of one explicit chain to a minimizer, $\prod_{i<l}r_{j_i}(K_i)/q_i$ (a single term of Lemma 7.2).
- If $\kappa_u=\kappa_v$ this corollary says nothing. The comparison is then decided by the tie depth and the constant of Step 11.
- At short horizons the order can differ from the $\kappa$ order. The crossover bound quantifies when the long-time order is guaranteed.

---

### Step 11. Second-order asymptotics: tie depth and seed-specific constant

Theorem 3 and Corollary 1 give the slope $\kappa$ with a logarithmic error. This step identifies the next two terms exactly, which is where seeds with equal slopes are told apart. It is the continuous-time, single-node version of the "second-order survival asymptotics" statement. It follows from the exact formula of Theorem 4, so it is the leading term of a finite-time formula.

**Definition 7 (tie depth).** For a chain $c$ let $k_c=\#\{i:\ K_i\in\mathcal M\}$ be the number of minimizers on it. The **tie depth** is $d_v=\max_ck_c$. It is $\ge1$, because some minimizer is reachable and Lemma 3.1 gives a chain ending at it.

**Lemma 8 (tie depth is the height of the minimizer poset).** $d_v$ equals the largest $d$ for which there are minimizers $I_1\subsetneq I_2\subsetneq\dots\subsetneq I_d$ in $\mathcal M$.

*What is happening:* any strictly nested family of reachable minimizers can be strung onto a single growth chain, so ties on one history are exactly nested minimizers. Reachable sets can be climbed one node at a time without leaving the reachable set.

*Proof.*
1. ($\le$) The minimizers on a chain are strictly nested sets of $\mathcal M$. So $k_c\le$ height.
2. ($\ge$) *Climbing claim:* if $I\subsetneq J$ are both in $\mathcal R$, there is $u\in J\setminus I$ with an in-neighbor in $I$, and $I\cup\{u\}\in\mathcal R$. Proof: pick any $u'\in J\setminus I$. By Lemma 3.1 there is a directed path $v=u_0\to u_1\to\dots\to u_r=u'$ inside $J$. Let $u_s$ be the first node on it not in $I$ ($s\ge1$ since $v\in I$). Then $u_{s-1}\in I$ has an edge to $u_s\in J\setminus I$. Every node of $I\cup\{u_s\}$ is reachable from $v$ inside $I\cup\{u_s\}$ (nodes of $I$ because $I\in\mathcal R$, and $u_s$ via $u_{s-1}$), and $|I\cup\{u_s\}|\le|J|\le m$. So $I\cup\{u_s\}\in\mathcal R$ by Lemma 3.1.
3. Given nested minimizers $I_1\subsetneq\dots\subsetneq I_d$: take a growth chain from $\{v\}$ to $I_1$ (Lemma 3.1), then apply the climbing claim repeatedly to go from $I_a$ to $I_{a+1}$. This gives a chain, with all rates positive, passing through all $I_a$. Hence $k_c\ge d$. $\square$

**Theorem 6 (second-order asymptotics).** Under no traps, let $a_v>0$ be the constant of Theorem 4.4. Then for $t\ge1$:
1. If $d_v\ge2$:
$$\delta_t(v)=a_v\,t^{d_v-1}e^{-\kappa t}\bigl(1+O(1/t)\bigr),\qquad C_t(v)=\kappa t-(d_v-1)\log t-\log a_v+O(1/t).$$
2. If $d_v=1$: the correction is exponentially small,
$$\delta_t(v)=a_v\,e^{-\kappa t}\bigl(1+O(t^{m-1}e^{-gt})\bigr),\qquad C_t(v)=\kappa t-\log a_v+O(t^{m-1}e^{-gt}).$$

*What is happening:* if the epidemic can climb through $d_v$ nested bottlenecks in a row, it spends $d_v$ consecutive $\mathrm{Exp}(\kappa)$ stages in them, an Erlang time, and this adds the polynomial factor $t^{d_v-1}$. The non-bottleneck stages are faster and only rescale it by constants. The constant $a_v$ depends on the seed through the chain weights, so it is the seed-specific information that the slope alone misses.

*Proof.*
1. Theorem 4 gives, for all $t\ge0$, $e^{\kappa t}\delta_t=P_1(t)+\sum_{a\ge2}e^{-(\theta_a-\kappa)t}P_a(t)$.
2. By Theorem 4.4, $P_1(t)=a_vt^{d_v-1}+(\text{terms of degree}\le d_v-2)$, and $P_1\equiv a_v$ if $d_v=1$.
3. For $a\ge2$, $\theta_a-\kappa\ge g$ and $\deg P_a\le m-1$ (Theorem 4.1). So for $t\ge1$, $\bigl|\sum_{a\ge2}e^{-(\theta_a-\kappa)t}P_a(t)\bigr|\le C\,t^{m-1}e^{-gt}$ for a constant $C$.
4. Hence $e^{\kappa t}\delta_t=a_vt^{d_v-1}+O(t^{d_v-2})$ for $d_v\ge2$ (the exponentially small term is absorbed), and $e^{\kappa t}\delta_t=a_v+O(t^{m-1}e^{-gt})$ for $d_v=1$.
5. Divide by $a_vt^{d_v-1}$ to obtain the relative error. For $t$ large enough that $|r|\le\frac12$ in $\delta=a_vt^{d_v-1}e^{-\kappa t}(1+r)$, $|\log(1+r)|\le2|r|$. Take $-\log$, and use $C_t=-\log\delta_t$. $\square$

**Remarks.**

**1. Worked checks.**
- *Four nodes, seed 1* (Step 9). $\mathcal M=\{\{1,3\}\}$, $\kappa=2$, $d_v=1$, $g=0.5$, and the only chain with $k_c=1$ is $(\{1\},\{1,3\})$ with $w=2$, $\pi_c=1/(3-2)=1$, so $a_v=2$. This matches $\delta_t=2e^{-2t}+2e^{-2.5t}-3e^{-3t}$, whose relative correction decays like $e^{-0.5t}=e^{-gt}$.
- *Directed path $1\to2\to3\to4$, all rates $1$, $\gamma=1$* ($m=3$). Every reachable set has $q=1$, so $\kappa=1$ and $d_v=3$. The chains of lengths $1,2,3$ have kernels $e^{-t},\,te^{-t},\,\frac{t^2}2e^{-t}$ and weights $1$. So $\delta_t=e^{-t}(1+t+\frac{t^2}2)$, $P_1=1+t+\frac{t^2}2$, $a_v=\frac12$, and $C_t=t-2\log t+\log2+O(1/t)$.

**2. Consequences for ranking.** At equal $\kappa$,
$$C_t(v)-C_t(u)=-(d_v-d_u)\log t-\log\frac{a_v}{a_u}+o(1).$$
So at long horizons the seed with the larger tie depth, and then the larger $a_v$, has the larger survival probability and hence the **smaller** centrality: it lingers longer in nested bottlenecks. The hierarchy of information is $\kappa_{\gamma,v}$, then $d_v$, then $a_v$.

**3. A proof without partial fractions (explicit remainders).** Step 1: $h_c(t)e^{\kappa t}=g_c(t)$ is the convolution of the functions $e^{-\varepsilon_it}$, with $\varepsilon_i=q_i-\kappa\ge0$. Step 2: the $k=k_c$ minimizer stages give $\varepsilon_i=0$, so the factor is the constant $1$, whose $k$-fold convolution is $t^{k-1}/(k-1)!$. Step 3: the remaining $p=l+1-k$ stages have $\varepsilon_i\ge g>0$. Their convolution $E\ge0$ satisfies $\int_0^\infty E=\prod\varepsilon_i^{-1}=\pi_c$ (Tonelli) and, by monotonicity of convolution of nonnegative functions, $E(u)\le\frac{u^{p-1}}{(p-1)!}e^{-gu}$. So $E$ has an exponentially small tail and finite moment $\mu_1=\int uE(u)\,du$. Step 4: $g_c(t)=\int_0^tE(u)\frac{(t-u)^{k-1}}{(k-1)!}du$. For $k=1$, $g_c=\pi_c-\int_t^\infty E$. For $k\ge2$, the mean value theorem gives $|(t-u)^{k-1}-t^{k-1}|\le(k-1)ut^{k-2}$, hence $\bigl|g_c(t)-\pi_c\frac{t^{k-1}}{(k-1)!}\bigr|\le\frac{t^{k-1}}{(k-1)!}\int_t^\infty E+\frac{t^{k-2}}{(k-2)!}\mu_1=O(t^{k-2})$. Step 5: chains with $k_c=0$ have all rates $\ge\kappa+g$, so $h_c\le\frac{t^l}{l!}e^{-(\kappa+g)t}$. Summing with Lemma 7.3 gives Theorem 6 again.

**4. The Erlang route.** Use Lemma 6 instead. On a full path with $k\ge1$ minimizers, the minimizer stages add up to exactly an $\mathrm{Erlang}(k,\kappa)$ time $T$, and the other stages add up to an independent time $Y$ with all rates $\ge\kappa+g$. The needed fact is:

> **Lemma 9 (Erlang plus a light tail).** Let $T\sim\mathrm{Erlang}(k,\kappa)$, $k\ge1$, and $Y\ge0$ independent with $\mathbb E[e^{\kappa'Y}]<\infty$ for some $\kappa'>\kappa$. Then
> $$\mathbb P(T+Y>t)=\mathbb E[e^{\kappa Y}]\ \frac{(\kappa t)^{k-1}}{(k-1)!}\,e^{-\kappa t}\,(1+o(1)).$$

*Proof.* (i) $\mathbb P(T>x)=e^{-\kappa x}S_k(\kappa x)$ for $x\ge0$ and $=1$ for $x<0$. (ii) Conditioning on $Y$: $\mathbb P(T+Y>t)=\mathbb P(Y>t)+e^{-\kappa t}\mathbb E[\mathbf 1_{Y\le t}\,e^{\kappa Y}S_k(\kappa(t-Y))]$. (iii) By Markov's inequality, $\mathbb P(Y>t)\le e^{-\kappa't}\mathbb E[e^{\kappa'Y}]=o(t^{k-1}e^{-\kappa t})$. (iv) Divide the second term by $t^{k-1}e^{-\kappa t}$. Pointwise, $S_k(\kappa(t-y))/t^{k-1}\to\kappa^{k-1}/(k-1)!$. The ratio is bounded by a constant for $t\ge1$, uniformly in $y\le t$, because $S_k$ is increasing. The dominating function $Ce^{\kappa Y}$ is integrable, so dominated convergence applies. $\square$

Since $Y$ is a sum of independent $\mathrm{Exp}(q_i)$ with $q_i>\kappa$, $\mathbb E[e^{\kappa Y}]=\prod_{q_i>\kappa}\frac{q_i}{q_i-\kappa}$. Paths with fewer than $d_v$ minimizers are lower order, and paths with none are exponentially smaller (Erlang domination at rate $\kappa+g$). Every chain with $k_c=d_v$ extends to a full path without losing minimizers (no traps give a positive-rate continuation), so the maximum over full paths is also $d_v$. This gives the same leading term with
$$a_v=\frac{\kappa^{d_v-1}}{(d_v-1)!}\sum_{\pi:\,k_\pi=d_v}\Pr(\pi)\prod_{i:\,q_i>\kappa}\frac{q_i}{q_i-\kappa}.$$
It agrees with the chain formula because the leading coefficient is unique. Check on the four-node example: $\Pr(\pi)=\frac23$, product $=3$, $a_v=2$. This route yields $o(1)$ quickly, and $O(1/t)$ needs one more moment condition.

**5. Caveats.**
- The constants hidden in $O(\cdot)$ are graph dependent and not tracked, so Theorem 6 is an asymptotic statement. A finite-time ranking time is the job of Corollary 2.
- No traps was used so that $\kappa>0$ and every chain can be continued. With a trap, $\kappa=0$, no chain can contain two traps, so $d_v=1$ and $\delta_t\to a_v$, the probability of ever reaching a trap.
- The chain structure uses single-node jumps. For the discrete multi-node chain an analogous statement holds with chains of arbitrary increments and geometric holding times, but it is not proved here.
- Computing $a_v$ and $d_v$: both are sums over chains, exponential in general. A state-by-state recursion on the reachable sets (propagating the pair (degree, coefficient)) avoids enumerating chains, but is not developed here.

---

## Part D. Summary map to the discrete paper

| Discrete paper | Here |
|---|---|
| Closed form $\mathbf 1^\top Q_\gamma^te_{\{v\}}$ | Theorem 1 |
| Reachability reduction | Lemma 4 (with Lemma 3) |
| Spectrum = diagonal | Lemma 5 |
| Degree limit with $\alpha$-error | Theorem 2, exact |
| Spectral radius limit | Theorem 3 proof, step 1 |
| Min-cut limit with $\alpha$-error | Theorem 3, exact |
| Corollary with $-\log(1-A_{ij})$ | Absorbed (rates are the primitives) |
| (none) | Lemma 6: sum-of-exponentials structure |
| (none) | Lemma 7, Theorem 4: exact chain and quasi-polynomial forms (Step 9) |
| (none) | Theorem 5, Corollaries 1-2: sandwich, log-order, ranking crossover (Step 10) |
| Second-order survival asymptotics (literature note, Prop. 2) | Theorem 6 (Step 11), from Theorem 4 and Lemma 8 |

**Caveats to keep visible.**
- The continuous chain and the paper's synchronous discrete chain have the same diagonal under $A^{\rm disc}_{ij}=1-e^{-\beta A_{ij}\Delta}$, but different off-diagonals (single-node jumps versus synchronous multi-node steps). Steps 3, 4, 5, 7, 8 use only the diagonal and triangularity, so they transfer. Step 6 uses single-node jumps.
- If the symbol $A$ is reused with the new meaning (contact weights), say so explicitly, since the discrete paper uses $A_{ij}$ as a probability.
- The model is Markovian by assumption (memoryless clocks). Fixed infectious delays would break the exponential structure.

%%## Part E. Not covered here (next)

Explicit constants in the Step 11 remainders, a state-by-state recursion for $d_v$ and $a_v$ without enumerating chains, the discrete multi-node analogues of Theorems 4 to 6, and the extensions to SIR and SIS.%%
