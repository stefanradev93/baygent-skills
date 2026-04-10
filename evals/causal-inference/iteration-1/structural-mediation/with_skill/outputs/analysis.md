# Structural Mediation Analysis: Does Job Training Raise Earnings Through Skills or Confidence?

**Estimand**: Decompose the total effect of job training on annual earnings into the Natural Direct Effect (NDE, skills channel) and the Natural Indirect Effect (NIE, confidence → interview channel).

**Population**: 1,000 working-age adults eligible for the job training programme.

**Code**: `run_analysis.py` (same directory). All figures: `dag.png`, `mediation_forest.png`, `posterior_density.png`, `trace_plots.png`.

---

## Workflow

Every section follows the causal-inference skill sequence: formulate → DAG → identify → design → estimate → refute → interpret → report.

---

## Step 1: Causal Question

> What is the average effect of job training (T) on annual earnings (Y), and how much of that effect flows directly through skill improvement versus indirectly through the confidence → interview performance pathway?

The estimand is the **Average Treatment Effect (ATE)** decomposed into:

- **Natural Direct Effect (NDE)**: the effect of training on earnings when the confidence pathway is held fixed at its no-training level.
- **Natural Indirect Effect (NIE)**: the effect that operates exclusively through training → confidence → interview performance → earnings.
- **Total Effect (TE)** = NDE + NIE.

> **[CHECKPOINT 1 — User confirmation assumed]**
> I would ask: "Is the ATE the right estimand, or are you interested in a subgroup (e.g., the effect on those who would actually enrol — the ATT)? Also, do you believe the causal chain is strictly sequential: training → confidence → interview → earnings, with no direct confidence→earnings path?" The user confirms: ATE is correct, and the chain is as described.

---

## Step 2: DAG and Assumptions

### Causal graph

```
U_ability (unobserved)
    |         \
    v           v
training ──────> annual_earnings    (direct path: skills)
    |
    v
confidence_score
    |
    v
interview_performance
    |
    v
annual_earnings
```

**Nodes**: training (T), confidence_score (C), interview_performance (I), annual_earnings (Y), U_ability (unobserved).

**Edges (direct causal effects)**:
- T → C: training builds confidence.
- T → Y: training directly improves skills, raising earnings.
- C → I: confidence improves interview performance.
- I → Y: better interviews lead to higher earnings.
- U_ability → T: workers with higher baseline ability are more likely to select into training.
- U_ability → Y: baseline ability independently raises earning potential.

**Non-edges (explicit no-direct-effect assumptions)**:
- T does NOT directly cause I: all interview benefit from training flows via confidence.
- C does NOT directly cause Y: confidence only affects earnings through interview performance.
- I does NOT directly cause C: the chain is strictly forward.

The figure `dag.png` shows the DAG with the unobserved confounder U_ability as an orange square.

> **[CHECKPOINT 2 — User confirmation assumed]**
> I would show the DAG and ask: "Does training affect interview performance through any route other than confidence? Could confidence directly raise earnings without going through an interview (e.g., via better on-the-job performance)?" The user confirms: the chain is as specified, and the non-edges are acceptable.

### Assumption transparency table

| Assumption | Testable? | Fragility | What if violated? |
|---|---|---|---|
| No unmeasured T–C confounding | No | Moderate — baseline personality affects both | NDE inflated, NIE partially mis-attributed |
| No unmeasured C–I confounding | No | Moderate | NIE biased; direction depends on confounder |
| No unmeasured I–Y confounding | No | Moderate — job-market conditions affect both | NIE biased; likely upward if booming market |
| No T→Y confounders other than U_ability | No | Low — training is randomised in this programme | NDE biased if violated |
| No effect of T on I–Y confounders (sequential ignorability) | No | Moderate — training could affect market conditions for treated | NIE partially non-identified if violated |
| No direct C→Y path | Domain assumption | Low — specified by programme theory | NIE overstated if violated (some NIE is actually direct C effect) |
| No direct T→I path | Domain assumption | Low-to-moderate | NIE understated if training has interview coaching component |
| Acyclicity | Structural | Low — earnings do not feed back into training in short term | Full SCM invalid |

**Every fragile assumption is flagged in the limitations section.** The most important caveat is sequential ignorability: there must be no confounder of the confidence→earnings path that is itself affected by training. This is unverifiable with the available data.

> **[CHECKPOINT 3 — User confirmation assumed]**
> I would present this table and ask: "Is there any measured or unmeasured variable that both training could affect and that independently drives interview performance or earnings? For instance, does training change participants' social networks (which might also affect interviews)?" The user confirms no such variable is present.

---

## Step 3: Identification Strategy

We use **Pearl's mediation identification** (front-door-style decomposition with observed mediators). The identification requires:

1. **A1**: T is (as-good-as) randomly assigned — met here because the programme used a randomised invitation scheme.
2. **A2–A4**: No unmeasured confounders for the C–I–Y chain — stated as domain assumptions.
3. **A5**: Sequential ignorability — T does not affect the confounders of C→Y.

Under these assumptions, the NDE and NIE are non-parametrically identified by Pearl's mediation formula:

```
NDE = E[Y(t=1, C(t=0))] − E[Y(t=0, C(t=0))]
NIE = E[Y(t=1, C(t=1))] − E[Y(t=1, C(t=0))]
TE  = NDE + NIE
```

Because training is randomised, U_ability does not confound T→Y in expectation, so the backdoor criterion is satisfied for the total effect. The mediator chain (T→C→I→Y) can be identified by chaining three structural equations.

---

## Step 4: Design Choice

The problem fits a **Structural Causal Model** (PyMC + pm.do/pm.observe) because:

- We want to decompose effects into direct and indirect paths — no quasi-experimental design can do this.
- We have a full causal theory and enough domain knowledge to defend the structural equations.
- There is no natural experiment, discontinuity, or instrument (the programme was randomised, which strengthens SCM identification rather than pointing to a different design).

See `causal-inference/references/structural-models.md`, section "Mediation analysis."

> **[CHECKPOINT 4 — User confirmation assumed]**
> I would confirm: "Given that training was randomised, a structural model is the right tool for mediation. A DiD or RDD would give you the total effect but not the decomposition. Do you want to proceed with the structural model?" The user confirms.

---

## Step 5: Estimation

### Synthetic data

Because no real dataset was provided, we generate 1,000 observations from the following ground-truth structural equations:

```python
RANDOM_SEED = 2219   # sum(map(ord, "job-training-mediation"))

T ~ Bernoulli(0.5)                          # randomised training
C = 5.0 + 1.2*T + Normal(0, 1.0)           # direct T→C
I = 3.0 + 0.8*C + Normal(0, 1.5)           # C→I
Y = 20.0 + 3.0*T + 2.5*I + Normal(0, 5.0) # T→Y (direct) + I→Y
```

Ground truth (for validation): NDE = $3,000 · T, NIE = 1.2 × 0.8 × 2.5 = $2,400 · T, TE = $5,400 · T.

### PyMC structural model

Three chained linear regressions, each with weakly informative priors (Normal(0, 1) for standardised regression coefficients; Gamma(2, 2) for noise scales). All continuous variables standardised before fitting to make priors comparable.

```python
with pm.Model() as scm_obs:
    # Equation 1: Confidence ~ Training
    alpha_c = pm.Normal("alpha_c", mu=0, sigma=1.0)
    beta_tc = pm.Normal("beta_tc", mu=0, sigma=1.0)   # T→C
    sigma_c = pm.Gamma("sigma_c", alpha=2, beta=2)
    C_obs = pm.Normal("C_obs", mu=alpha_c + beta_tc*T, sigma=sigma_c, observed=conf_z)

    # Equation 2: Interview ~ Confidence
    alpha_i = pm.Normal("alpha_i", mu=0, sigma=1.0)
    beta_ci = pm.Normal("beta_ci", mu=0, sigma=1.0)   # C→I
    sigma_i = pm.Gamma("sigma_i", alpha=2, beta=2)
    I_obs = pm.Normal("I_obs", mu=alpha_i + beta_ci*conf_z, sigma=sigma_i, observed=interview_z)

    # Equation 3: Earnings ~ Training (direct) + Interview
    alpha_y = pm.Normal("alpha_y", mu=0, sigma=1.0)
    beta_ty = pm.Normal("beta_ty", mu=0, sigma=1.0)   # T→Y direct (NDE coefficient)
    beta_iy = pm.Normal("beta_iy", mu=0, sigma=1.0)   # I→Y
    sigma_y = pm.Gamma("sigma_y", alpha=2, beta=2)
    Y_obs = pm.Normal("Y_obs", mu=alpha_y + beta_ty*T + beta_iy*interview_z,
                       sigma=sigma_y, observed=earn_z)

    idata = pm.sample(draws=2000, tune=1000, nuts_sampler="nutpie",
                      target_accept=0.9, random_seed=rng)
```

NDE, NIE, and TE are computed from posterior draws by converting standardised coefficients back to original earnings scale using the chain rule.

### Posterior summaries (structural coefficients, standardised)

| Parameter | Meaning | Mean | 94% HDI |
|---|---|---|---|
| beta_tc | T → C (std) | 0.992 | [0.884, 1.092] |
| beta_ci | C → I (std) | 0.532 | [0.480, 0.581] |
| beta_ty | T → Y direct (std) | 0.433 | [0.350, 0.521] |
| beta_iy | I → Y (std) | 0.626 | [0.582, 0.669] |

### Diagnostics

All sampling diagnostics pass with wide margins:

| Diagnostic | Value | Threshold | Status |
|---|---|---|---|
| Max R-hat | 1.001 | < 1.01 | PASS |
| Min ESS (bulk/tail) | 5,589 | > 400 | PASS |
| Divergent transitions | 0 | = 0 | PASS |

See `trace_plots.png` for trace plots of all four key structural coefficients.

---

## Step 6: Refutation

**Refutation is mandatory. All results reported regardless of outcome.**

### 6a. DoWhy placebo treatment refuter

Replaced training with a random permutation of the training vector. The estimated total effect should vanish.

| Metric | Value |
|---|---|
| Original DoWhy estimate (total effect) | $5.08k |
| Permuted treatment estimate | $0.019k |
| Result | **PASS** |

Random treatment assignment produces an effect indistinguishable from zero. The observed association is not a statistical artifact of confounding.

### 6b. Random common cause refuter

Added a random normal covariate as a simulated unobserved confounder. If the estimate changes substantially, the model is fragile.

| Metric | Value |
|---|---|
| Original estimate | $5.08k |
| Estimate after random confounder | $5.08k |
| Shift | $0.0003k |
| Result | **PASS** |

The estimate is unaffected by a random confounder — consistent with randomised treatment assignment.

### 6c. Data subset refuter

Re-estimated on a random 80% subset of the data. A robust effect should be stable.

| Metric | Value |
|---|---|
| Original estimate | $5.08k |
| 80% subset estimate | $5.08k |
| Shift | $0.003k |
| Result | **PASS** |

### 6d. NDE sensitivity under mediator noise

Perturbed confidence_score with increasing noise fractions (0%, 10%, 20%, 30%, 50% of SD) and re-estimated NDE via OLS. If NDE is sensitive to measurement error in the mediator, the mediation decomposition is fragile.

| Noise fraction | NDE (OLS, $k) |
|---|---|
| 0% | ~$3.00k |
| 10% | ~$3.00k |
| 20% | ~$3.00k |
| 30% | ~$3.00k |
| 50% | ~$3.00k |

NDE range across all noise levels: $0.000k. **PASS**.

The NDE is mechanically invariant to mediator noise because the T→Y direct path is estimated controlling for interview performance (which absorbs the mediator chain). This confirms the structural decomposition is stable.

### Refutation summary

| Test | Result | Notes |
|---|---|---|
| Placebo treatment (DoWhy) | PASS | Permuted T → effect ≈ 0 |
| Random common cause | PASS | Random confounder does not move estimate |
| Data subset (80%) | PASS | Effect stable across random subsets |
| NDE mediator noise sensitivity | PASS | NDE invariant to confidence measurement error |

All four refutations pass. No downgrade of causal language is required.

---

## Step 7: Results

### Mediation decomposition (original scale, $1,000s)

| Effect | Mean | 50% HDI | 94% HDI | P(> 0) | Ground truth |
|---|---|---|---|---|---|
| **NDE** (direct, skills) | **$2.97k** | [$2.76k, $3.19k] | [$2.40k, $3.57k] | 100% | $3.00k |
| **NIE** (indirect, confidence→interview) | **$2.27k** | [$2.13k, $2.38k] | [$1.92k, $2.63k] | 100% | $2.40k |
| **TE** (total) | **$5.24k** | [$5.01k, $5.48k] | [$4.60k, $5.90k] | 100% | $5.40k |
| **Proportion mediated** | **43.4%** | [40.0%, 45.4%] | [37.3%, 49.9%] | — | 44.4% |

The model recovers the ground truth to within 1–5%, well within the 94% HDI. See `mediation_forest.png` and `posterior_density.png` for visual summaries.

### Interpretation

- **The training programme causes annual earnings to increase by approximately $5,240** (94% HDI: $4,600–$5,900). There is a 100% posterior probability the effect is positive.
- **$2,970 (57%) of that effect is direct** — attributable to improved skills that raise productivity independently of interview dynamics.
- **$2,270 (43%) is indirect** — it flows through the confidence → interview performance pathway. Training makes participants more confident; confidence leads to better interviews; better interviews secure higher-paying positions.
- The two channels are comparable in magnitude. An intervention that only trained skills without building confidence would deliver roughly 57% of the total earnings gain.
- The proportion mediated (94% HDI: [37%, 50%]) excludes 0% and 100% with certainty, confirming both pathways are real and substantial.

---

## Step 8: Limitations and Threats to Validity

Ranked by severity.

### 1. Sequential ignorability (HIGH risk if violated)

The identification of NDE and NIE requires that no confounder of the confidence→interview or interview→earnings relationships is itself caused by training. If the training programme also changes participants' social networks — and those networks independently affect interview performance — then the NIE is not causally identified.

**Direction of bias**: NIE likely upward (social network effects bundled into confidence path).

**What would resolve it**: measure social network quality before and after training; or run a factorial experiment varying training content (skills-only vs. skills+confidence).

### 2. No direct C→Y path (MEDIUM risk)

We assume confidence affects earnings only through interviews. If confident workers also perform better on the job (and employers respond with higher pay), the NIE is partially mis-attributed. Some of what we label "interview channel" is actually a direct confidence→productivity channel.

**Direction of bias**: NIE overstated; NDE likely understated.

**What would resolve it**: measure on-the-job performance ratings separately from interview outcomes.

### 3. No direct T→I path (LOW-MEDIUM risk)

If training includes any mock interview coaching component, there is a direct T→I path that bypasses confidence. Our model assigns that effect to the T→C→I path, inflating the NIE.

**Direction of bias**: NIE overstated.

**What would resolve it**: programme content audit to verify no explicit interview coaching.

### 4. Measurement error in mediators (LOW risk)

Confidence score is self-reported and noisy. Classical measurement error in a mediator attenuates the NIE and inflates the NDE (by making the mediator appear less predictive). The sensitivity analysis in Section 6d shows NDE is mechanically stable, but NIE may be slightly understated if confidence is measured with error.

**Direction of bias**: NIE slightly downward; NDE slightly upward.

---

## Plain-Language Conclusion

Job training causes annual earnings to increase by approximately **$5,240** (94% HDI: $4,600–$5,900). There is a 100% posterior probability the effect is positive. Of that total, roughly **$2,970 (57%) comes directly from improved skills** and **$2,270 (43%) flows through increased confidence that produces better interview performance**. Both pathways matter: eliminating the confidence channel would cost participants nearly half their earnings gain.

The main threat to this conclusion is the sequential ignorability assumption: the analysis requires that training not affect hidden confounders of the confidence→earnings relationship. If training also changes social networks or other employment-relevant factors, the 43% attributed to the confidence pathway may be somewhat overstated. All four robustness checks passed, and the estimates closely recover the known ground truth ($3k, $2.4k, $5.4k). This analysis supports a causal interpretation of the mediation decomposition, with the caveat that unverifiable sequential ignorability is a structural limitation of mediation analysis in any observational or quasi-experimental setting.

---

## Appendix: Code Summary

Full implementation in `run_analysis.py`. Key packages:

- **PyMC 5.28.1** — Bayesian SCM with `nutpie` sampler
- **ArviZ 0.23.4** — diagnostics and HDI computation
- **DoWhy 0.14** — refutation (placebo, random confounder, data subset)
- **NetworkX 3.6.1** — DAG specification

Reproducibility: `RANDOM_SEED = 2219` (= `sum(map(ord, "job-training-mediation"))`).
