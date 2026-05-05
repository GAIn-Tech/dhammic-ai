# Related-work map and incremental contribution: branch-factor budget / interface slack as operational luck

## Recursion round 1 -- candidate contribution and core construct

**Contribution claim.** Treat "luck" not as a metaphysical residue but as an operational property of an agent--environment interface: the number, quality, reversibility, and temporal accessibility of next actions available under uncertainty. This reframes random luck, Murphy events, effort, preparation, opportunity, slack, and fragility as quantities over reachable state/action spaces.

Let state be `s_t`, feasible actions `A(s_t)`, environmental events `E_t`, transition `T`, and viability predicate `V`. Define an **effective branch-factor budget**:

```tex
B_epsilon(s_t) = sum_{a in A(s_t)} Pr_e[V(T(s_t,a,e))=1] 1[c(a)<=C_t] 1[Delta I(a)<=I_t]
B_tau(s_t) = sum_{a in A(s_t)} 1{ E_e[U(T(s_t,a,e))] - lambda Risk(a) >= tau }.
```

**Interface slack** is the reserve that prevents small perturbations from collapsing this effective branch set:

```tex
S = min_{k in K} (K_k - L_k)/sigma_k,
```

where `K_k` is capacity, `L_k` is load, and `sigma_k` is disturbance scale. Low slack means many affordances are nominal but not executable. "Murphy" is a distributional statement: under high coupling and low slack, routine variance causes `B_tau` to drop discontinuously.

**Luck taxonomy.**
- Random luck: exogenous event increases payoff without prior change in `A(s)` or `S`.
- Opportunity--preparation luck: prior action increased `B_tau` or observation bandwidth, so a later event becomes actionable.
- Intent effort: costly operations that expand, preserve, or prune branches: `A(s)`, `C_t`, information `I_t`, reversibility, or search depth.
- Bad luck / Murphy: exogenous event intersects low slack/tight coupling so branches collapse faster than replanning.
- Fragility: negative convexity of feasible branch count with respect to perturbation: `d^2 B/d delta^2 < 0`.

## Recursion round 2 -- verified adjacent literatures and claim relevance

### Search, planning, and branch factors
- Hart, Nilsson, Raphael (1968), "A Formal Basis for the Heuristic Determination of Minimum Cost Paths," DOI `10.1109/TSSC.1968.300136`. Grounds branching, path cost, and heuristic information.
- Puterman (1994), *Markov Decision Processes*, DOI `10.1002/9780470316887`. Provides stochastic actions, transitions, rewards, policies.
- Simon (1956), "Rational Choice and the Structure of the Environment," DOI `10.1037/h0042769`. Bounded rationality depends on environmental structure.
- Kauffman and Weinberger (1989), "The NK Model of Rugged Fitness Landscapes...", DOI `10.1016/S0022-5193(89)80019-0`. Shows branch count is insufficient without landscape topology.

### Affordances, opportunity, prepared luck
- Gibson (1979; DOI-verified Routledge edition 2013), *The Ecological Approach to Visual Perception*, DOI `10.4324/9780203767764`. Affordances are actor-environment possibilities for action.
- Gaver (1991), "Technology Affordances," DOI `10.1145/108844.108856`. Perceptible and hidden affordances map to observation bandwidth.
- Star and Griesemer (1989), "Institutional Ecology, 'Translations' and Boundary Objects...", DOI `10.1177/030631289019003001`. Interfaces and boundary objects coordinate heterogeneous actors.
- Mitchell, Levin, Krumboltz (1999), "Planned Happenstance," DOI `10.1002/j.1556-6676.1999.tb02431.x`. Direct support for constructed opportunity and prepared luck.
- Granovetter (1973), "The Strength of Weak Ties," DOI `10.1086/225469`. Weak ties expand nonredundant opportunity search space.
- Merton (1968), "The Matthew Effect in Science," DOI `10.1126/science.159.3810.56`. Cumulative advantage converts early chance into durable opportunity access.

### Resilience engineering, normal accidents, Murphy dynamics
- Woods and Hollnagel (2017), *Resilience Engineering*, DOI `10.1201/9781315605685`. Adaptive capacity literature; interface slack as proxy.
- Woods (2015), "Four Concepts for Resilience...", DOI `10.1016/j.ress.2015.03.018`. Maps to rebound, robustness, graceful extensibility, sustained adaptability.
- Woods (2018), "The Theory of Graceful Extensibility," DOI `10.1007/s10669-018-9708-3`. Slack matters near boundaries of competence.
- Perrow (1984/2011), *Normal Accidents*, DOI `10.2307/j.ctt7srgf`. Tight coupling and interactive complexity collapse alternatives.
- Reason (1990), *Human Error*, DOI `10.1017/CBO9781139062367`; Reason (1997/2016), *Managing the Risks of Organizational Accidents*, DOI `10.4324/9781315543543`. Defenses and latent conditions preserve/block branches.
- Vaughan (1996/1997), *The Challenger Launch Decision*, DOI `10.7208/chicago/9780226346960.001.0001`. Normalization of deviance consumes slack while operations appear normal.

### Complex systems slack, robust-yet-fragile designs, queues, cascades
- Carlson and Doyle (2000), "Highly Optimized Tolerance," DOI `10.1103/PhysRevLett.84.2529`; arXiv `cond-mat/9812127`. Robust to expected perturbations, fragile to rare ones.
- Doyle et al. (2005), "The 'robust yet fragile' nature of the Internet," DOI `10.1073/pnas.0501426102`. Performance optimization can hide interface fragility.
- Motter and Lai (2002), "Cascade-based attacks on complex networks," DOI `10.1103/PhysRevE.66.065102`; arXiv `cond-mat/0301086`. Overload cascades when local slack is exhausted.
- Lai, Motter, Nishikawa (2004), "Attacks and Cascades in Complex Networks," DOI `10.1007/978-3-540-44485-5_14`. Topology/load redistribution/cascade linkage.
- Watts (2002), "A Simple Model of Global Cascades on Random Networks," DOI `10.1073/pnas.082090499`. Local thresholds can generate global cascades.
- Kingman (1962), "Some Inequalities for the Queue GI/G/1," DOI `10.2307/2333966`. Queueing slack is quantitative: waiting grows sharply as utilization approaches 1.

Queueing approximation:
```tex
W_q proportional to [rho/(1-rho)] [(c_a^2+c_s^2)/2] E[S].
```

### Algorithmic information, computational irreducibility, risk/irreversibility
- Wolfram (1984), "Universality and Complexity in Cellular Automata," DOI `10.1016/0167-2789(84)90245-8`; Wolfram (1985), "Undecidability and Intractability in Theoretical Physics," DOI `10.1103/PhysRevLett.54.735`. Some trajectories resist compressed prediction.
- Kolmogorov (1968 English translation), "Three approaches to the quantitative definition of information," DOI `10.1080/00207166808803030`; Solomonoff (1964) parts I/II, DOIs `10.1016/S0019-9958(64)90223-2` and `10.1016/S0019-9958(64)90131-7`. Surprise, compressibility, and search priors.
- Langton (1990), "Computation at the edge of chaos," DOI `10.1016/0167-2789(90)90064-V`. Adaptability between rigidity and chaos.
- Kahneman and Tversky (1979), "Prospect Theory," DOI `10.2307/1914185`; Dixit and Pindyck (1994), *Investment under Uncertainty*, DOI `10.1515/9781400830176`. Risk, loss asymmetry, and irreversible commitment increase the value of options/slack.

## Recursion round 3 -- draft-ready synthesis

This paper joins three lines of work that are usually kept separate. First, search and decision theory model agents as choosing among feasible branches under uncertainty, from heuristic state-space search to Markov decision processes. Second, ecological psychology and organization studies treat opportunities as affordances: possibilities for action created jointly by an actor, environment, and interface. Third, resilience engineering and complex-systems safety show that high-performing systems can lose adaptive capacity when optimized to the edge of their envelopes. We connect these literatures by defining luck operationally as a change in the effective branch-factor budget at an interface. Preparation and effort are branch-expanding, branch-preserving, or branch-disambiguating operations; Murphy events are perturbations that reveal low slack by collapsing feasible branches; and random luck is exogenous branch/payoff improvement not attributable to prior branch preparation.

**Incremental novelty.**
- Gives a measurable target: `B_tau(s)`, the number/weight of viable next actions above a threshold under constraints.
- Extends MDP/search models by emphasizing interface slack: time, resources, translation tolerance, and reversibility that make nominal actions executable.
- Extends resilience engineering from accident avoidance to opportunity conversion and prepared luck.
- Separates irreducible unpredictability from controllable exposure: agents may not predict `e`, but they can alter `A(s)`, observation channels, reversibility, and load margins.

**Testable implications.**
1. Prepared-luck hypothesis: actors with larger pre-event `B_tau` or better observability convert the same exogenous event into higher realized opportunity.
2. Murphy/load hypothesis: negative shocks have superlinear effects as utilization `rho -> 1` or coupling increases.
3. Robust-yet-fragile hypothesis: optimization increases success under expected perturbations while reducing branch diversity under off-distribution perturbations.
4. Irreversibility hypothesis: marginal value of slack is higher where actions close future branches or losses are asymmetric.

**Caveats.**
- Branch-factor budget is not subjective choice count; hidden cost, timing, information, and coordination constraints make many nominal choices infeasible.
- More branches are not always better; relevant branches are effective, viable, value-weighted branches net of search cost.
- Slack can be wasteful under stable conditions; the target is option value and failure-mode accounting, not blanket slack maximization.
- Computational irreducibility prevents complete prediction; this is operational risk/opportunity accounting, not deterministic fate theory.
- Semantic Scholar API was attempted but returned rate-limit `429`; CrossRef DOI resolution and arXiv API were successful verification sources in this pass.

## BibTeX-ready source list

Generated file: `/home/mikeb/research_luck_slack/verified_sources.bib`.

arXiv-verified adjuncts:
- Carlson and Doyle HOT preprint: `https://arxiv.org/abs/cond-mat/9812127`
- Motter and Lai cascade-based attacks preprint: `https://arxiv.org/abs/cond-mat/0301086`
