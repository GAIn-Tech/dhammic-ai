# Empirical Methodology for Testing a Branching-Event Framework

## Contribution and falsifiable objective

We contribute a reproducible empirical design for testing whether heterogeneous public events exhibit measurable, temporally localized increases in downstream option-generation, which we operationalize as a **branch-factor index** (BFI). The design is intentionally incremental: it can support (i) descriptive replication, (ii) correlational tests, (iii) causal-negative-control falsification, and (iv) controlled toy-system experiments where ground truth is known. The framework is considered empirically useful only if it produces pre-registered, out-of-sample improvements over null models and survives negative controls, multiple-comparison correction, and independent replication.

Let \(e_i=(t_i,c_i,s_i,m_i)\) denote an event with timestamp \(t_i\), source class \(c_i\), severity \(s_i\), and metadata \(m_i\). Let \(Y_{k,t}\) be a measurable outcome stream for domain \(k\). The central hypothesis is that selected event classes increase near-future branching activity:

\[
H_1: \mathbb{E}[B_{t+h}\mid E_t=1, X_t] > \mathbb{E}[B_{t+h}\mid E_t=0, X_t]
\]

for pre-specified horizons \(h\in H\), where \(B_t\) is the branch-factor index and \(X_t\) denotes calendar, market, platform, and exposure controls. The primary null is no temporal association beyond base rates, seasonality, source reporting intensity, and autocorrelation.

## Data-collection radars

We define each radar as a timestamped, append-only collector with raw payload retention, canonicalized fields, source-health logging, and immutable extraction code. Radars should collect both positive events and background denominators, because event-count spikes are uninterpretable without source availability and reporting volume.

- **AI status radar:** public status pages and incident feeds for major AI providers; model release notes; API latency/error notices; benchmark or leaderboard update timestamps; moderation/policy notices. Fields: provider, service, incident type, start/end, severity, affected region, status-page revision history.
- **Outage radar:** cloud, DNS, telecom, payment, and major SaaS incidents. Fields: provider, layer, region, dependency, customer impact, duration, detection source.
- **Lottery-payout radar:** draw times, jackpot size, rollover count, ticket-sales estimates where available, winner count, payout class distribution, jurisdiction, publicity intensity proxies.
- **Social-trend radar:** trend emergence and decay from public trend APIs, news indices, Wikipedia pageviews, open social datasets, and search-interest proxies. Fields: topic string, platform, rank, volume proxy, geography, language, first-observed time.
- **Geomagnetic radar:** NOAA/space-weather Kp, Ap, solar flare, coronal mass ejection, proton flux, and geomagnetic storm alerts. Fields: measurement time, forecast/nowcast distinction, magnitude, alert level.
- **Finance radar:** equity index, volatility, rates, FX, commodities, crypto market data, macro announcement calendars, trading halts. Fields: instrument, return, realized volatility, volume, announcement surprise, market session.
- **Auth/CDN radar:** identity-provider, CDN, certificate, domain-registry, and routing incidents. Fields: provider, authentication path affected, cache/DNS/TLS layer, affected endpoints, mitigation time.
- **Crypto-exploit radar:** exploit disclosures, on-chain abnormal flows, bridge/DeFi incidents, oracle failures, governance attacks. Fields: chain, protocol, exploit class, estimated loss, transaction hashes, disclosure lag.

Each radar logs \(A_{r,t}\), a source-availability indicator, and \(V_{r,t}\), the volume of parsed source documents. These enter all models as controls to reduce artifacts from scraping failures or media attention.

## Event coding schema

Events are coded in a versioned schema with human-auditable labels and machine-readable uncertainty:

\[
e_i = \{id,t_{start},t_{detect},t_{end},c,sc,s,g,d,u,q,src\}\
\]

where \(c\) is domain class, \(sc\) subclass, \(s\in[0,1]\) normalized severity, \(g\) geography, \(d\) duration, \(u\) uncertainty interval for timing/severity, \(q\) source-quality score, and \(src\) raw-source references. Multi-label events are allowed but must be represented as a parent event plus typed links rather than duplicated independent observations.

Recommended coding variables:

- event class and subclass;
- detection lag \(t_{detect}-t_{start}\);
- duration and recovery state;
- ex ante predictability: scheduled, forecastable, unscheduled;
- severity: ordinal raw label plus normalized score;
- exposure denominator: users, transaction volume, market cap, population, API calls, or pageviews;
- confidence score and coder disagreement;
- duplicate/related-event links;
- media/reporting intensity proxy.

Inter-coder reliability is reported using Krippendorff's \(\alpha\) or Cohen's \(\kappa\) for categorical fields and intra-class correlation for severity. Disagreements are resolved under a frozen adjudication protocol; both pre- and post-adjudication labels are released.

## Branch-factor index estimation

The BFI measures the effective number of plausible next states generated by an event or time window. We estimate it using multiple operational definitions and require robustness across at least two.

### Entropy-based BFI

For a window \([t,t+\Delta]\), define outcome categories \(j=1,\ldots,J\) and empirical probabilities \(p_{j,t}\). The entropy branch factor is

\[
B^{H}_t = \exp\left(-\sum_{j=1}^{J} p_{j,t}\log p_{j,t}\right).
\]

This is the effective number of equally likely outcome categories. Categories must be pre-registered per domain, e.g., incident follow-up types, trend descendants, market-regime bins, or exploit copycat classes.

### Dispersion-based BFI

For continuous outcomes \(Y_{k,t}\), define standardized residuals after removing expected dynamics:

\[
\epsilon_{k,t}=Y_{k,t}-\widehat{\mathbb{E}}(Y_{k,t}\mid X_t,Y_{k,t-1:t-p}).
\]

Then

\[
B^{D}_t = 1 + \frac{1}{K}\sum_{k=1}^{K} w_k \left|\frac{\epsilon_{k,t}}{\widehat{\sigma}_{k,t}}\right|,
\]

where \(w_k\) are pre-specified reliability weights. This index captures unexpected cross-domain dispersion.

### Cascade-based BFI

Represent linked follow-up events as a temporal graph \(G=(V,E)\). For event \(i\), the local offspring estimate is

\[
B^{C}_i(\Delta)=\sum_{j\neq i}\mathbf{1}(0<t_j-t_i\leq \Delta)P(i\rightarrow j),
\]

where \(P(i\rightarrow j)\) is estimated from temporal proximity, semantic similarity, shared entities, and domain-specific linkage rules. A conservative binary variant uses only human-confirmed links.

Primary analyses should report a composite z-scored index:

\[
B_t = \sum_{m\in\{H,D,C\}} \omega_m z(B^m_t), \quad \sum_m \omega_m=1,
\]

with \(\omega_m\) fixed before analysis. Sensitivity analyses vary \(\Delta\), category definitions, and weights.

## Correlation and causal controls

All observational tests include controls for:

- hour-of-day, day-of-week, month, holidays, and market sessions;
- scheduled announcements and known draws/releases;
- source availability \(A_{r,t}\) and source volume \(V_{r,t}\);
- reporting intensity and media attention;
- lagged outcomes and event counts;
- domain-specific exposure denominators;
- global shocks, e.g., macro announcements or major geopolitical events;
- multiple testing across radars, horizons, and outcomes.

A baseline event-study specification is:

\[
B_{t}=\alpha + \sum_{h=-H_0}^{H_1}\beta_h E_{t-h}+\gamma^\top X_t+\phi(L)B_t+\eta_{d(t)}+\varepsilon_t,
\]

where \(\eta_{d(t)}\) are calendar fixed effects and \(\phi(L)\) captures autoregression. Evidence for the framework requires post-event coefficients \(\beta_h\) that are positive at pre-registered horizons while pre-event coefficients are null or substantively smaller.

For quasi-causal inference, use difference-in-differences where unaffected geographies, providers, protocols, or instruments provide controls:

\[
Y_{g,t}=\alpha_g+\lambda_t+\delta(Treated_g\times Post_t)+\theta^\top X_{g,t}+\varepsilon_{g,t}.
\]

The identifying assumption is parallel trends; it is probed with pre-trend tests and placebo event times.

## Null models and falsification tests

We use several nulls because a single null can be too weak.

1. **Calendar-preserving permutation:** shuffle event labels within the same hour-of-week/month strata.
2. **Source-volume-matched permutation:** permute events among windows with similar \(V_{r,t}\) and \(A_{r,t}\).
3. **Phase-randomized time series:** preserve autocorrelation and spectrum while destroying alignment.
4. **Block bootstrap:** resample contiguous blocks to preserve local dependence.
5. **Synthetic control null:** construct weighted controls from non-exposed domains or geographies.
6. **Negative-control events:** use events that should not plausibly affect the outcome, e.g., unrelated jurisdictions or low-severity incidents.
7. **Negative-control outcomes:** use outcomes known before the event or physically unrelated to the event class.

A result is considered falsified if comparable effects appear for negative controls, if pre-event effects dominate post-event effects, if results vanish under source-volume controls, or if out-of-sample predictions fail to exceed null baselines.

## Time-series tests: cross-correlation, Granger, and Hawkes processes

### Cross-correlation

For event stream \(E_t\) and BFI \(B_t\), compute prewhitened cross-correlation:

\[
\rho(\ell)=\operatorname{corr}(\tilde{E}_{t},\tilde{B}_{t+\ell}),
\]

where tildes denote residuals after calendar, exposure, and autoregressive adjustment. Confidence intervals are obtained by block bootstrap and permutation.

### Granger prediction

Estimate nested autoregressive models:

\[
B_t = a_0 + \sum_{p=1}^{P}a_pB_{t-p}+\Gamma^\top X_t+u_t,
\]

\[
B_t = b_0 + \sum_{p=1}^{P}b_pB_{t-p}+\sum_{q=1}^{Q}c_qE_{t-q}+\Gamma^\top X_t+v_t.
\]

The event stream Granger-predicts BFI if the second model improves out-of-sample loss and the joint test \(c_1=\cdots=c_Q=0\) is rejected after correction. We report effect sizes, not only p-values.

### Hawkes/self-exciting processes

For event types \(k\), model conditional intensity:

\[
\lambda_k(t)=\mu_k(t)+\sum_{j}\sum_{t_i^j<t}\alpha_{jk}\exp[-\beta_{jk}(t-t_i^j)],
\]

where \(\mu_k(t)\) includes exogenous seasonality and source volume. A self-exciting interpretation requires \(\alpha_{jk}>0\), improved held-out likelihood, stable residual diagnostics, and no comparable excitation in negative controls. Branching ratio

\[
n_{jk}=\int_0^\infty \alpha_{jk}e^{-\beta_{jk}u}du=\alpha_{jk}/\beta_{jk}
\]

provides a direct cascade metric; stationarity requires the spectral radius of \(N=[n_{jk}]\) to be below 1.

## Permutation and multiple-testing plan

The primary test statistic is the maximum standardized post-event effect across pre-registered horizons:

\[
T=\max_{h\in H}\frac{\widehat{\beta}_h}{\widehat{SE}(\widehat{\beta}_h)}.
\]

Empirical p-values are computed as

\[
p=\frac{1+\sum_{r=1}^{R}\mathbf{1}(T_r^{null}\geq T_{obs})}{R+1}.
\]

Family-wise error is controlled with max-T permutation where feasible; otherwise Benjamini-Hochberg false-discovery-rate control is applied within pre-defined families: radar class, horizon, and outcome family.

## Preregistration

Before observing outcome-aligned results, register:

- radar list, source URLs/APIs, collection frequency, and exclusion criteria;
- event schema, coding manual, severity normalization, and duplicate rules;
- primary BFI definition, windows \(\Delta\), horizons \(H\), and model formulas;
- primary and secondary hypotheses;
- null models, permutation strata, block lengths, and correction method;
- negative-control events and outcomes;
- minimum detectable effect and power simulation assumptions;
- missing-data handling and source-outage handling;
- replication package structure and acceptance criteria;
- rules for labeling exploratory analyses.

## Replication protocol

A replication team should be able to reproduce the study from raw-source manifests without private knowledge.

1. Freeze a time range and source manifest.
2. Run collectors in append-only mode with checksums for raw payloads.
3. Generate canonical event tables using versioned parser containers.
4. Double-code a stratified sample and compute reliability.
5. Produce BFI streams from pre-registered scripts.
6. Run primary models and null tests with a single command.
7. Compare output hashes, coefficient signs, confidence intervals, and permutation p-values.
8. Conduct blinded replication on a later time period or held-out geography/provider.
9. Release deviations as a machine-readable audit log.

Minimum replication criteria: same coefficient direction for primary horizons, overlapping confidence intervals or compatible posterior intervals, no failed negative controls, and no undocumented data exclusions.

## Controlled toy-system experiments

Observational evidence is insufficient; we add toy systems where the true branching process is known.

### LLM prompt branching

Seed an LLM with standardized prompts and perturbations. Define branches as semantically distinct continuations clustered with frozen embeddings and human validation. Experimental conditions vary temperature, prompt ambiguity, context length, tool availability, and injected incident-like cues. Ground-truth expectations: higher temperature and ambiguity should increase entropy BFI; deterministic decoding should reduce it. Falsification: BFI fails to track known decoding parameters.

### CAPTCHA mock experiment

Create mock human-or-bot classification tasks with controlled ambiguity and feedback. Branches are decision paths: pass, fail, retry, escalate, abandon, or request alternative. Randomize interface friction, error messages, and queue delay. Measure whether small perturbations induce cascades of retries or abandonment. Use no real CAPTCHA bypassing or adversarial deployment; this is a laboratory UI simulation.

### Queue simulations

Simulate service queues with Poisson arrivals, bursty arrivals, priority classes, retries, and cascading failures. Known parameters define the true reproduction number of downstream tasks. Estimate Hawkes branching ratios and compare to ground truth. This tests estimator bias under censoring, delayed observation, and source-volume changes.

### Lottery-slip human-choice study

Recruit participants to choose lottery numbers under randomized information conditions: jackpot size, recent winner publicity, rollover count, social-salience primes, and quick-pick availability. Outcomes are number entropy, clustering on salient dates, duplicate-rate predictions, and stated reasoning. This tests whether public payout events alter human choice distributions without requiring claims about actual lottery causation. Ethics: disclose that the task is simulated or non-purchasing unless legally approved; avoid encouraging gambling.

## Scientific process steps

1. State falsifiable hypotheses and negative controls before analysis.
2. Build radars with raw-data retention and source-health logs.
3. Freeze schemas and coding manuals.
4. Conduct pilot coding only on a training period.
5. Run power simulations using realistic autocorrelation and reporting lags.
6. Preregister primary models, windows, and exclusion rules.
7. Analyze a locked test period.
8. Run null, placebo, and sensitivity tests.
9. Replicate on a later period or independent source set.
10. Publish raw manifests, derived data, code, containers, and audit logs.

## Reproducibility checklist

- [ ] Raw-source manifest with retrieval timestamps and checksums.
- [ ] Append-only raw payload archive or legally shareable pointers.
- [ ] Versioned event schema and coding manual.
- [ ] Coder IDs, reliability metrics, and adjudication logs.
- [ ] Parser, model, and figure scripts in containers.
- [ ] Fixed random seeds and documented stochastic components.
- [ ] Pre-registered hypotheses, horizons, nulls, and corrections.
- [ ] Source-availability and source-volume controls.
- [ ] Negative-control events and outcomes.
- [ ] Sensitivity analyses for windows, lags, severity thresholds, and duplicate rules.
- [ ] Out-of-sample validation or later-period replication.
- [ ] Machine-readable computational environment file.
- [ ] One-command reproduction script.
- [ ] Clear separation of confirmatory and exploratory results.

## Caveats and boundary conditions

This methodology cannot by itself prove metaphysical or global causal claims. It can only test whether a defined index of branching activity changes after specified event classes beyond plausible null explanations. Public data are affected by reporting bias, platform outages, API policy changes, and media attention. Event times may be uncertain, especially for exploits and social trends. Hawkes excitation is not causal without strong assumptions; Granger prediction is predictive rather than mechanistic. Lottery and human-choice studies must avoid implying improved gambling outcomes. CAPTCHA experiments must remain benign mock systems. The framework should be rejected or narrowed when it fails preregistered tests, when effects are explained by source-volume artifacts, or when independent replication does not reproduce the primary pattern.
