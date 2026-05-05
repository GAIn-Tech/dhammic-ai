# Branch-Factor Budget and Interface Slack: An Incremental Progress Framework for Luck, Murphy Events, and Intent-Driven State-Space Traversal

**Draft type:** incremental progress paper, not a final conclusion  
**Version:** 2026-05-05-v0.1  
**Status:** conceptual formalization + live-event pilot dossier + preregisterable empirical agenda  
**Author note:** This paper deliberately does **not** propose new physical laws. It reframes the motivating “fictional resource” as an operational accounting framework over finite task-state spaces, complex-system slack, and computational effort.

---

## Abstract

We introduce **branch-factor budget** and **interface slack** as an operational framework for studying events colloquially described as luck, Murphy's law, or improbable failure clusters. The framework treats an agent's situation as a finite task-state space whose feasible trajectories depend on action affordances, resource margins, information, irreversibility, and the topology of success variants. In this view, intention is not a new physical force: it is a constraint specification that prunes, weights, or expands reachable state trajectories, while effort is the computational and physical cost of selecting and realizing such trajectories. We separate **prepared luck**, in which opportunity meets prior branch-expansion or slack creation, from **random tail luck**, in which exogenous events improve outcomes without observed preparation. We formalize Murphy-like events as high-entropy adverse branch selection under low slack and repeated exposure. A pilot event sweep across lottery payout clustering, AI-provider incidents, CAPTCHA/auth gates, social-to-physical meme propagation, geomagnetic activity, payment/market rails, and infrastructure outages motivates the framework but does not establish causal claims. We propose a reproducible methodology using timestamped radars, event coding, entropy/dispersion/cascade branch-factor indices, null models, Hawkes/self-exciting processes, permutation tests, and controlled toy systems. The contribution is incremental: it turns a speculative intuition about “luck as a resource” into falsifiable measurements over state-space traversal, interface centrality, slack, and irreversibility.

---

## 1. Introduction

A small input can produce a large consequence when it crosses the right interface. A lottery slip's visual geometry can concentrate many players on the same number pattern. A prompt can launch recursive model inference and tool use. A CAPTCHA provider issue can block login flows across unrelated sites. A viral meme can move crowds into physical space. A single overprivileged AI-agent action can mutate a production database. A cable cut can disable emergency communications. These events differ in domain, but they share an architecture: a low-cost trigger is translated by a high-dependency interface into many downstream state changes.

This paper develops a modest formalism for that architecture. We call the usable quantity at stake **branch-factor budget**: the effective number, quality, reversibility, and accessibility of viable next states available to an agent or system under constraints. We call the protective reserve **interface slack**: the capacity, redundancy, queue tolerance, observability, reversibility, and safety margin that prevents a perturbation from collapsing viable branches.

The framework is motivated by recent pilot observations, including:

- a Powerball draw in which the white-ball numbers `3, 19, 35, 51, 67` yielded 91 high-tier matching tickets, plausibly because the numbers formed a vertical pattern on some play slips;
- same-day AI-provider incidents affecting Claude model families and a ChatGPT web surface;
- an official Cloudflare Turnstile challenge incident affecting challenge solve attempts;
- a multi-city “Scientology speedrunning” trend in which a social-video template produced physical crowd actions;
- reported AI-agent database deletion and recovery failure;
- telecom, passport, airline, emergency-call, payment, and crypto incidents with narrow triggers and broad dependency surfaces;
- a measured May 4 G2 geomagnetic storm/Kp 6.33, treated as a real physical background perturbation but not as a proven causal explanation for the above.

These observations do **not** prove a hidden field, global causal coupling, or probability violation. They motivate a measurable question: when do small triggers become large visible anomalies because the affected interface has high centrality, high synchronization, low slack, and irreversible downstream commitments?

### Contributions

1. **Formal model:** We define finite task-state spaces, success-variant coverage, ideal progress trajectories as constrained shortest-path / generalized traveling-salesperson problems, branch-factor indices, slack, irreversibility, intention, effort, luck, and Murphy exposure.
2. **Conceptual separation:** We separate prepared luck from random tail luck and distinguish semantic intent from computational/physical effort.
3. **Pilot taxonomy:** We classify recent events as operational examples of branch-factor amplification, while preserving confidence tiers and alternative explanations.
4. **Empirical methodology:** We specify radars, coding schemas, BFI estimators, null models, Granger/cross-correlation/Hawkes tests, permutation controls, toy experiments, and reproducibility criteria.
5. **Research posture:** We frame the paper as incremental progress. The goal is not to conclude that luck is a physical resource, but to produce a rigorous scaffold that can be falsified, reproduced, and extended.

---

## 2. Related Work

The framework connects literatures that are usually treated separately.

### 2.1 Search, planning, and state spaces

Heuristic search and shortest-path planning ground the idea of progress as traversal through a graph of possible states. A* formalizes how heuristic information reduces search burden in path problems [@Hart_1968]. Markov decision processes model stochastic transitions, policies, and rewards [@Puterman_1994]. Simon's bounded rationality emphasizes that rational choice depends on the structure of the environment, not only on internal cognition [@Simon_1956]. Rugged fitness landscapes show that the topology of alternatives matters: many branches are not equivalent if the success landscape is epistatic and irregular [@Kauffman_1989].

### 2.2 Affordances, opportunity, and prepared luck

The opportunity side of luck is close to the ecological notion of affordances: possibilities for action generated by an actor-environment relation [@Gibson_2013]. Technology affordances extend this to interfaces and hidden/perceptible action possibilities [@Gaver_1991]. Boundary objects show how shared interface artifacts coordinate heterogeneous actors [@Star_1989]. Planned happenstance theory explicitly treats chance career events as constructible opportunities [@Mitchell_1999]. Weak ties expand access to nonredundant opportunities [@Granovetter_1973], and cumulative advantage can convert early lucky events into persistent access to future opportunities [@Merton_1968].

### 2.3 Resilience, accidents, and slack

Resilience engineering studies adaptive capacity under pressure [@Woods_2017]. Woods distinguishes robustness, rebound, graceful extensibility, and sustained adaptability [@Woods_2015; @Woods_2018]. Normal accident theory explains how tight coupling and interactive complexity make failures hard to isolate [@PERROW_2011]. Reason's accident models describe latent conditions and defensive layers [@Reason_1990; @Reason_2016], while Vaughan's Challenger analysis shows how normalization of deviance consumes slack while operations appear normal [@Vaughan_1997].

### 2.4 Complex systems, queues, and cascades

Highly optimized tolerance and robust-yet-fragile design explain why systems can be resilient to expected perturbations while fragile to rare ones [@Carlson_2000; @Doyle_2005]. Cascade models show how local overload or targeted disruption can propagate through networks [@Motter_2002; @Lai_2004]. Watts shows that local threshold behavior can generate global cascades [@Watts_2002]. Kingman's queueing approximation captures why waiting and instability rise sharply as utilization approaches one [@Kingman_1962].

### 2.5 Information, irreducibility, and risk

Algorithmic information and inductive inference provide tools for surprise, compressibility, and prior-weighted search [@Kolmogorov_1968; @Solomonoff_1964]. Computational irreducibility and complex cellular automata remind us that some trajectories cannot be compressed into cheap prediction [@Wolfram_1984; @Wolfram_1985]. Edge-of-chaos work connects adaptability to regimes between rigidity and disorder [@Langton_1990]. Prospect theory and investment under uncertainty ground loss asymmetry, option value, and irreversibility [@Kahneman_1979; @Dixit_1994].

---

## 3. Core Concepts

### 3.1 Interface

An **interface** is a translation layer where a small local input becomes many downstream state changes. Examples include lottery slips, prompts, API gateways, CAPTCHA providers, payment rails, telecom cables, social platforms, and passport systems. Interfaces are important because they concentrate interpretation and commitment.

### 3.2 Branch-factor budget

Branch-factor budget is the effective volume of viable futures available under constraints. It is not raw choice count. A nominal action is not effective if it is too costly, too late, unobservable, irreversible in the wrong direction, or blocked by dependencies.

### 3.3 Interface slack

Slack is residual feasibility margin. It includes spare capacity, fallback paths, queue tolerance, rollback ability, resource reserve, monitoring, and recovery independence. Slack is not always optimal to maximize; it has carrying costs. Its value rises near irreversible commitments and high-coupling interfaces.

### 3.4 Intention

An intention is a constraint specification. It defines acceptable terminal states, prohibited states, required variants, budgets, and preferences. The semantic word “intent” is not thermodynamic by itself. However, when an intention is realized through computation, communication, actuation, or institutional response, it incurs physical and computational costs.

### 3.5 Luck and Murphy

We separate:

- **Prepared luck:** prior effort expanded branches, observability, reversibility, or option value such that an opportunity becomes usable.
- **Random tail luck:** exogenous residual outcome improvement not explained by observed opportunity/preparation variables.
- **Murphy event:** perturbation intersects low slack and high branch entropy, making adverse successors likely across repeated exposure.

---

## 4. Formal Framework

### 4.1 Finite task-state space

A task instance is

```math
\mathcal{T}=(S,s_0,A,P,C,G,\Phi),
```

where `S` is a finite set of task states, `s_0` is the initial state, `A(s)` is the admissible action set, `P(s'|s,a)` is the transition kernel, `C(s,a,s') >= 0` is generalized cost, `G subset S` is the terminal success set, and `Phi:S -> 2^V` maps states to covered success variants.

The induced directed graph is

```math
\Gamma=(S,E), \qquad E=\{(s,s'): \exists a \in A(s) \text{ with } P(s'|s,a)>0\}.
```

A progress topology can be generated by coverage. Define

```math
s \preceq_\Phi t \iff \Phi(s) \subseteq \Phi(t).
```

The Alexandrov topology generated by upper sets of this preorder treats states with greater coverage as lying in more advanced neighborhoods.

### 4.2 Success variants and ideal trajectory

Let `V_* subset V` denote required success variants. A feasible trajectory

```math
\gamma=(s_0,s_1,\ldots,s_T)
```

covers

```math
\Phi(\gamma)=\bigcup_{t=0}^{T}\Phi(s_t).
```

It succeeds when

```math
\Phi(\gamma) \supseteq V_* \, , \quad s_T \in G \text{ if terminal success is required.}
```

Given resource budgets `B_j` and resource costs `c_j`, the ideal incremental trajectory solves

```math
\gamma^* \in \arg\min_\gamma \sum_{t=0}^{T-1}w(s_t,s_{t+1})
```

subject to

```math
\gamma_0=s_0,\quad \Phi(\gamma)\supseteq V_*,\quad \sum_{e\in\gamma}c_j(e)\le B_j \quad \forall j.
```

This is a constrained shortest-path problem with group coverage constraints. If every variant corresponds to one or more states `S_v={s:v in Phi(s)}`, the agent must visit at least one state from every required group. This generalizes traveling-salesperson path, orienteering, set cover, and group Steiner problems.

**Proposition 1.** The ideal trajectory problem is NP-hard in general.

**Proof sketch.** Reduce TSP path by mapping each city to a required variant, each city-state to coverage of that variant, and edge weights to distances. A minimum successful trajectory visits all variants with minimum distance. Alternatively, reduce set cover by mapping states to sets of variants and minimizing visited set-states. Thus the general class contains NP-hard problems. This does not imply every real task is hard; it says general progress planning requires exploiting structure.

### 4.3 Branch-factor index

At state `s`, define feasible successors under constraints `I` and remaining budget `b`:

```math
N_I(s;b)=\{s'\in S:(s,s')\in E,\ s'\text{ satisfies active constraints},\ c(s,s')\le b\}.
```

The local branch factor is

```math
\beta(s;b,I)=|N_I(s;b)|.
```

With belief distribution `q(s'|s)`, define entropy

```math
H(s)=-\sum_{s'\in N(s)}q(s'|s)\log q(s'|s),
```

and effective branch factor

```math
\beta_{eff}(s)=\exp(H(s)).
```

The cumulative pathwise branch-factor index is

```math
\mathcal{B}(\gamma)=\sum_{t=0}^{T-1}\log \beta_{eff}(s_t).
```

This is the log-volume of plausible choices encountered along the path. Under uniform branching, selecting one trajectory among alternatives requires approximately `mathcal{B}` nats of path-specifying information.

### 4.4 Slack

For resource `j`, remaining budget after prefix `gamma_{0:t}` is

```math
r_j(t)=B_j-\sum_{i=0}^{t-1}c_j(s_i,s_{i+1}).
```

Let `d_j^*(s_t)` be minimal required future cost in resource `j` to complete the task. Resource slack is

```math
\sigma_j(s_t)=r_j(t)-d_j^*(s_t).
```

A conservative aggregate is

```math
\sigma_{min}(s_t)=\min_j \sigma_j(s_t).
```

**Proposition 2.** With nonrenewable resources and nonnegative transition costs, remaining budget is nonincreasing. Slack increases only if a transition reduces minimal remaining completion cost by more than the resource consumed.

**Proof sketch.** Since `c_j >= 0`, `r_j(t+1)=r_j(t)-c_j <= r_j(t)`. Then

```math
\sigma_j(s_{t+1})-\sigma_j(s_t)=-c_j(s_t,s_{t+1})-[d_j^*(s_{t+1})-d_j^*(s_t)].
```

This is positive only if `d_j^*(s_t)-d_j^*(s_{t+1}) > c_j`.

### 4.5 Irreversibility

Graph-theoretic irreversibility of edge `(s,t)` is recovery cost:

```math
\rho(s,t)=\inf_{p:t\leadsto s}\sum_{e\in p}w(e)-w(s,t),
```

with censoring rather than infinity when recovery paths are unobserved. Reachability loss is

```math
\Delta_R(s,t)=|R(s)\setminus R(t)|,\qquad R(s)=\{u:u\text{ reachable from }s\}.
```

Informational irreversibility can be represented by reduction in possible worlds:

```math
I_t=\log|\Omega_t|-\log|\Omega_{t+1}|.
```

This is decision-theoretic information, not automatically thermodynamic entropy production.

**Proposition 3.** Irreversible commitments increase the value of information when hidden variables affect which branch is optimal.

```math
VoI(X)=\mathbb{E}_X[\max_a U(a,X)]-\max_a\mathbb{E}_X[U(a,X)]\ge 0.
```

The inequality is Jensen's inequality over a maximum. Irreversibility raises the cost of wrong actions, increasing payoff dispersion and therefore the value of resolving uncertainty before commitment.

### 4.6 Intention and effort

An intention is

```math
I=(V_*,G,K,B,\lambda),
```

where `V_*` are required variants, `G` terminal acceptable states, `K` invariants/prohibitions, `B` budgets, and `lambda` preferences over cost/risk/time/reversibility/quality.

Intention-induced constraint information can be estimated as

```math
\mathcal{C}_I(s)=D_{KL}(q_I(\cdot|s)\Vert q_0(\cdot|s)),
```

where `q_0` is pre-intention behavior and `q_I` is behavior under stated intention. A support-reduction estimate is

```math
R_I(s)=1-\frac{|A_I(s)|}{|A(s)|}.
```

Computational effort for algorithm `M` on instance `x` can be represented as

```math
E_{comp}(M,x)=T_M(x)+\eta M_M(x)+Q_M(x),
```

where `T` is runtime, `M_M` memory, and `Q` query/experiment count. Physical thermodynamic accounting is included only when the implementation is specified; e.g., Landauer-type erasure bounds apply to logically irreversible bit erasure,

```math
E_{erase}\ge k_BT\ln2\cdot n_{erased},
```

but this is not a general law of psychological effort.

### 4.7 Luck decomposition

Let `Y=1` denote success. Let `O` denote external opportunity variables and `P_r` preparation variables. A logistic model separates opportunity, preparation, and their interaction:

```math
Pr(Y=1|O,P_r)=\sigma(\theta_0+\theta_O^\top O+\theta_P^\top P_r+\theta_{OP}^\top(O\otimes P_r)).
```

- `theta_O`: opportunity effect.
- `theta_P`: preparation effect.
- `theta_OP`: prepared-luck interaction: preparation increases opportunity conversion.
- residual: random tail luck after observed covariates.

Tail luck is estimated out-of-sample as calibrated residual surprise:

```math
L_{tail}=Y-\mathbb{E}[Y|O,P_r,X].
```

**Proposition 4.** If

```math
\frac{\partial^2 Pr(Y=1|O,P_r)}{\partial O\partial P_r}>0,
```

then preparation amplifies the marginal value of opportunity. This formalizes “luck is where preparation meets opportunity” without collapsing it into random tail events.

### 4.8 Murphy exposure

Define bad successors relative to intention and slack threshold `kappa`:

```math
N_{bad}(s)=\{s'\in N(s):\sigma_{min}(s')<\kappa\ \text{or}\ s'\text{ violates }I\}.
```

Murphy risk is

```math
M(s)=Pr(s_{t+1}\in N_{bad}(s)|s).
```

Under maximum-entropy uncertainty,

```math
M(s)=\frac{|N_{bad}(s)|}{|N(s)|}.
```

Pathwise exposure is approximated by

```math
\mathcal{M}(\gamma)=1-\prod_{t=0}^{T-1}(1-M(s_t)),
```

or bounded by

```math
Pr(\exists t:s_{t+1}\in N_{bad}(s_t))\le\sum_t M(s_t).
```

If transition-cost noise `Z` causes failure when `Z > sigma`, then

```math
Pr(failure)=1-F_Z(\sigma),
```

so failure probability decreases with slack and rises sharply under heavy-tailed disturbances.

---

## 5. Pilot Event Topography

We treat the pilot sweep as a qualitative, hypothesis-generating map. It is not a causal proof.

### 5.1 Powerball vertical-column event

The Apr. 29, 2026 Powerball draw produced white-ball numbers `3, 19, 35, 51, 67` and Powerball `15`. The official draw page lists two jackpot winners, 27 Match 5 + Power Play winners, and 62 Match 5 winners: 91 tickets matching the five white balls. Reports suggested the numbers aligned as a vertical column on some play slips. The event is a clean example of a random draw intersecting a nonrandom human choice distribution. The anomaly lies less in the RNG and more in the interface that shaped human number selection.

### 5.2 AI-provider and model-surface incidents

Anthropic reported multiple same-day elevated-error incidents affecting Claude model families, and OpenAI reported a ChatGPT website issue. These incidents support the idea that probabilistic AI interfaces are high-fanout dependency surfaces. The pilot evidence does not establish shared root cause or hidden coupling.

### 5.3 AI-agent irreversible mutation

Reports of a Claude/Cursor-driven agent deleting a production database and backups illustrate intention-to-state amplification. A short command sequence crossed an overprivileged tool interface and produced irreversible or hard-to-recover state loss. The framework treats this as high irreversibility, low recovery-path independence, and poor permission slack.

### 5.4 Cloudflare Turnstile challenge failure

Cloudflare reported Turnstile challenge issues on May 4, 2026. CAPTCHA/auth gates sit in front of many unrelated workflows. A small provider-side defect can appear to users as scattered failures across independent sites because the same interface sits under many applications.

### 5.5 Scientology “speedrunning” meme

A viral social challenge reportedly produced physical incidents in San Diego, New York, and Vancouver. This is an example of meme-to-motion branch expansion: a small symbolic template coordinates many actors across geography.

### 5.6 Geomagnetic storm as background perturbation

NOAA/NASA confirmed a May 4 G2 geomagnetic storm with Kp 6.33. This is a real physical perturbation with global geospace reach. However, no verified grid/satellite outage cluster was found. In the framework it is a background variable that may widen disturbance tails for certain systems, not a primary explanation for all observed anomalies.

### 5.7 Infrastructure, finance, and crypto lanes

JetBlue, passport, telecom, Square, Chase, Coinbase, Nasdaq, FanDuel Racing, Ethereum wallet, and MEV reports illustrate narrower branch-factor motifs: centralized queues, payment/app dependency, simultaneous instrument halts, event-day betting sensitivity, dormant-key exposure, and DeFi mechanism amplification. Several are secondary-source or bounded; all require stricter coding before statistical use.

---

## 6. Empirical Methodology

### 6.1 Radars

Each radar is a timestamped append-only collector with raw payload retention, parser versioning, source-health logging, and source-volume controls.

- **AI status radar:** provider incident feeds, API/model status, release notes.
- **Outage radar:** cloud, DNS, CDN, telecom, SaaS, payment incidents.
- **Lottery radar:** draw results, jackpot size, ticket sales estimates, winner count by prize tier, jurisdiction.
- **Social trend radar:** public trend APIs, news indices, Wikipedia pageviews, search-interest proxies.
- **Geomagnetic radar:** NOAA Kp/Ap, solar flare, CME, proton flux, storm alerts.
- **Finance radar:** trading halts, volatility, returns, volume, market sessions.
- **Auth/CDN radar:** identity, CAPTCHA, TLS, DNS, CDN and routing incidents.
- **Crypto-exploit radar:** exploit disclosures, on-chain flows, bridge/DeFi incidents, transaction hashes.

Each radar records source availability `A_{r,t}` and source volume `V_{r,t}` to control reporting and scraping artifacts.

### 6.2 Event schema

Each event is coded as

```math
e_i=\{id,t_{start},t_{detect},t_{end},c,sc,s,g,d,u,q,src\},
```

where `c` is class, `sc` subclass, `s` severity, `g` geography, `d` duration, `u` uncertainty interval, `q` source quality, and `src` raw references. Related articles about the same incident are linked to a parent event rather than treated as independent observations.

Inter-coder reliability is reported using Cohen's `kappa` or Krippendorff's `alpha`; severity uses intra-class correlation. Disagreements are preserved pre- and post-adjudication.

### 6.3 Branch-factor index estimators

**Entropy BFI:**

```math
B^H_t=\exp\left(-\sum_{j=1}^{J}p_{j,t}\log p_{j,t}\right).
```

**Dispersion BFI:** residualize outcomes against expected dynamics,

```math
\epsilon_{k,t}=Y_{k,t}-\widehat{E}(Y_{k,t}|X_t,Y_{k,t-1:t-p}),
```

then define

```math
B^D_t=1+\frac{1}{K}\sum_{k=1}^{K}w_k\left|\frac{\epsilon_{k,t}}{\widehat\sigma_{k,t}}\right|.
```

**Cascade BFI:**

```math
B^C_i(\Delta)=\sum_{j\ne i}1(0<t_j-t_i\le\Delta)P(i\rightarrow j),
```

where linkage probability uses timing, semantic similarity, shared entities, and domain rules.

A composite preregistered index is

```math
B_t=\sum_{m\in\{H,D,C\}}\omega_m z(B^m_t),\quad \sum_m\omega_m=1.
```

### 6.4 Event-study and causal controls

The primary hypothesis is

```math
H_1:E[B_{t+h}|E_t=1,X_t] > E[B_{t+h}|E_t=0,X_t].
```

A baseline event-study model is

```math
B_t=\alpha+\sum_{h=-H_0}^{H_1}\beta_hE_{t-h}+\gamma^\top X_t+\phi(L)B_t+\eta_{d(t)}+\varepsilon_t.
```

Controls include calendar effects, source availability, source volume, media attention, lagged outcomes, exposure denominators, scheduled events, and global shocks. Evidence requires post-event coefficients at preregistered horizons without dominant pre-event effects.

Where treated and untreated geographies/providers/instruments exist, use difference-in-differences:

```math
Y_{g,t}=\alpha_g+\lambda_t+\delta(Treated_g\times Post_t)+\theta^\top X_{g,t}+\varepsilon_{g,t}.
```

### 6.5 Time-series and cascade tests

Prewhitened cross-correlation:

```math
\rho(\ell)=corr(\tilde E_t,\tilde B_{t+\ell}).
```

Granger prediction compares autoregressive models with and without event lags, requiring out-of-sample improvement and corrected joint tests.

Multivariate Hawkes processes model event-type intensities:

```math
\lambda_k(t)=\mu_k(t)+\sum_j\sum_{t_i^j<t}\alpha_{jk}\exp[-\beta_{jk}(t-t_i^j)].
```

The branching ratio

```math
n_{jk}=\alpha_{jk}/\beta_{jk}
```

measures expected offspring. Stationarity requires the spectral radius of the branching matrix below one.

### 6.6 Nulls and falsification

Null models:

1. calendar-preserving permutations;
2. source-volume-matched permutations;
3. phase-randomized time series;
4. block bootstrap;
5. synthetic controls;
6. negative-control events;
7. negative-control outcomes.

A finding is rejected or narrowed if comparable effects appear in negative controls, pre-event effects dominate, results vanish under source-volume controls, or out-of-sample predictions fail.

### 6.7 Controlled experiments

- **LLM prompt branching:** vary ambiguity, temperature, context length, and tool availability; measure semantic branch entropy using frozen embeddings plus human validation.
- **CAPTCHA mock experiment:** benign simulated challenge gate; vary friction/error messages/queue delay; measure retries, abandonment, escalation.
- **Queue simulation:** generate arrivals, bursts, retries, and cascading failures with known ground-truth branching ratios.
- **Lottery-slip choice study:** randomize play-slip geometry, jackpot information, quick-pick availability, salience primes; measure number entropy and duplicate rates without encouraging real gambling.

---

## 7. Falsifiable Predictions

1. **Branch burden:** holding task size and skill constant, larger `mathcal{B}(gamma)` predicts longer completion time and higher failure probability.
2. **Slack protection:** failure probability decreases monotonically with estimated slack, with sharper effects under heavy-tailed shocks.
3. **Intention constraint:** intentions that reduce action entropy without removing successful paths improve completion probability relative to vague goals.
4. **Irreversibility-information:** before irreversible transitions, information acquisition reduces regret more when outcome variance is high.
5. **Prepared luck:** opportunity variables predict success more strongly among prepared agents than unprepared agents; `O x P_r` terms are positive where preparation enables opportunity capture.
6. **Murphy exposure:** under low slack, adverse-event probability scales with `1-prod_t(1-M_t)` or the conservative sum `sum_t M_t`.
7. **Effort asymmetry:** tasks where successful states occupy a smaller fraction of high-progress states require more guidance, computation, or corrective action.
8. **Coverage path:** agents approximating constrained coverage shortest paths outperform greedy local-progress agents when success variants are complementary or order-dependent.

---

## 8. Reproducibility Protocol

1. Freeze a time range and source manifest.
2. Run collectors in append-only mode with raw-payload checksums.
3. Generate canonical event tables using versioned parser containers.
4. Double-code a stratified event sample; compute reliability.
5. Produce BFI streams from preregistered scripts.
6. Run primary models and null tests with one command.
7. Compare output hashes, coefficient signs, confidence intervals, and permutation p-values.
8. Conduct blinded replication on a later time period or held-out geography/provider.
9. Release deviations as a machine-readable audit log.

Minimum replication criteria: same coefficient direction at primary horizons, compatible intervals, passed negative controls, and no undocumented exclusions.

---

## 9. Limitations

- The pilot event dossier is exploratory and vulnerable to reporting bias.
- Public incidents have uncertain start times and incomplete impact estimates.
- Branch-factor budget depends on abstraction quality; poor state definitions can produce formal nonsense.
- More branches are not always better; effective viable value-weighted branches matter.
- Entropy in this paper is combinatorial or epistemic unless a physical stochastic substrate is specified.
- Geomagnetic activity is a real physical input but currently lacks evidence as a primary causal driver of the observed cluster.
- Hawkes and Granger tests are predictive, not automatically causal.
- Luck decomposition is model-relative; residual tail luck may shrink with better observability.
- The framework can be useful even if no cross-domain global coupling is found; it then becomes a local reliability/opportunity accounting tool.

---

## 10. Incremental Conclusion

The current evidence does not justify concluding that probability itself is being depleted or that a new physical law is needed. It does justify a research program around **symbol-to-state interfaces**. Luck and Murphy-like clustering can be treated as changes in reachable trajectories under uncertainty: preparation expands or preserves branches, opportunity exposes valuable branches, random tails select residual outcomes, and low slack makes adverse branches more likely under repeated exposure.

The next step is not more speculation. The next step is a preregistered measurement system: timestamped radars, source-volume controls, coded event tables, branch-factor indices, null models, and controlled toy systems. If the framework is real, it should predict which small triggers will produce disproportionate downstream fanout. If it fails, the failure modes will still identify where the metaphor breaks: poor abstraction, source bias, insufficient cross-domain linkage, or ordinary independent incidents.

In short:

```text
Luck = usable branch budget exposed by opportunity and made actionable by preparation.
Murphy = adverse branch exposure under low slack and irreversible constraints.
Intent = constraint specification over state space.
Effort = the computational and physical cost of selecting, preserving, and realizing a trajectory.
```

This is an incremental topography of the task-state space, not a final map.
