# Workplace Safety Program: Prior Sensitivity Analysis

## Executive summary

We model the effect of a workplace safety program on monthly injury counts across 30 factories, using a hierarchical Poisson regression with log-link. Because the analyst has an informative prior from a previous study (rate ratio ~ 0.75), a colleague challenged whether the conclusions are prior-driven. We address this by fitting the model under three prior scenarios and running power-scaling sensitivity analysis (`psense_summary`) on each.

## Data description

- **Source**: Synthetic data generated to match the scenario (30 factories, 24 months pre/post)
- **Sample size**: 1,440 factory-months (30 factories x 48 months)
- **Key variables**:
  - `injuries`: monthly injury count (outcome)
  - `post_program`: binary indicator (0 = before, 1 = after program)
  - `employees`: number of employees per factory (exposure/offset)
  - `factory`: factory identifier (grouping variable)
- **True data-generating values**: rate ratio = 0.75 (log RR = -0.29), factory SD ~ 0.40

## Model specification

### Generative story

Each factory has a baseline injury rate per employee per month, drawn from a shared population distribution (hierarchical structure). The safety program, once introduced, shifts the factory's injury rate by a multiplicative factor. Monthly injury counts are Poisson-distributed given the expected count = rate x employees.

### Mathematical notation

```
injuries_it ~ Poisson(mu_it)
log(mu_it) = alpha + delta_j[i] + beta * post_it + log(employees_it)

alpha ~ Normal(-5, 1)                    # population mean log-rate
sigma_factory ~ Gamma(2, 5)              # between-factory SD
delta_j_raw ~ Normal(0, 1)               # non-centered offset
delta_j = delta_j_raw * sigma_factory    # factory random intercept

beta ~ [varies by scenario]              # program effect (log rate ratio)
```

### Prior choices

| Parameter | Prior | Justification |
|-----------|-------|---------------|
| `mu_factory` | Normal(-5, 1) | log(0.005) ~ -5.3; weakly informative around plausible injury rates per employee-month |
| `sigma_factory` | Gamma(2, 5) | Mode ~ 0.2, mean 0.4; avoids zero, allows moderate heterogeneity across factories |
| `factory_offset_raw` | Normal(0, 1) | Non-centered parameterization to avoid funnel geometry |
| `beta_program` (informative) | Normal(-0.29, 0.10) | Previous study found ~25% reduction; tight prior encoding that knowledge |
| `beta_program` (weakly informative) | Normal(0, 0.50) | Centered at no effect, wide enough to accommodate large effects in either direction |
| `beta_program` (skeptical) | Normal(0, 0.15) | Mildly skeptical -- centered at no effect, narrower than weakly informative |

### Non-centered parameterization

We use a non-centered parameterization for the factory random intercepts to avoid the funnel geometry that plagues centered hierarchical models when the group-level SD is small.

## Workflow steps

### Step 1: Prior predictive check

Before fitting, we sample from the prior predictive distribution to verify priors produce plausible injury counts. Key checks:
- Are predicted injury counts in a reasonable range (0 to ~50 per factory-month)?
- Do we see implausible values (hundreds of injuries at a small factory)?
- Is the prior predictive distribution too narrow (over-constraining) or too wide?

### Step 2: Inference

We sample using `nutpie` (default chains, default tuning). Three separate fits -- one per prior scenario.

### Step 3: Convergence diagnostics

We use `arviz_stats.diagnose(idata)` as the primary check, covering:
- R-hat <= 1.01
- ESS bulk/tail >= 100 * n_chains
- Zero divergences
- No tree depth saturation
- E-BFMI >= 0.3

Visual inspection via rank plots (`az.plot_trace(kind="rank_vlines")`).

### Step 4: Posterior predictive check

We verify the fitted model can reproduce the observed injury counts using `az.plot_ppc()`.

### Step 5: Prior sensitivity analysis

This is the heart of the analysis. For each scenario, we run `psense_summary(idata)` and check the CJS (Cumulative Jensen-Shannon) divergence for both prior and likelihood perturbations.

#### Interpretation guide

| Pattern | Prior CJS | Likelihood CJS | Meaning |
|---------|-----------|----------------|---------|
| Low sensitivity | < 0.05 | < 0.05 | Posterior robust to prior/likelihood changes |
| Prior-data conflict | > 0.05 | > 0.05 | Prior and data disagree |
| Strong prior / weak likelihood | > 0.05 | < 0.05 | Prior dominates |
| Likelihood-driven | < 0.05 | > 0.05 | Data dominates (usually fine) |

#### Expected outcomes

- **Informative prior (N(-0.29, 0.10))**: With 1,440 observations, the data should be strong enough that even this tight prior does not dominate. However, `psense_summary` may flag it as sensitive because the prior is genuinely informative. If prior CJS > 0.05 but likelihood CJS < 0.05, the prior is pulling the posterior -- but since it agrees with the data (the true effect is indeed ~0.75), this is not a problem as long as we document it.

- **Weakly informative prior (N(0, 0.50))**: Should show low sensitivity on both prior and likelihood sides. The data has plenty of signal to overwhelm this vague prior.

- **Skeptical prior (N(0, 0.15))**: This is the most interesting case. It is centered at "no effect" and moderately tight. If the data is strong enough, the posterior will overcome the skeptical prior and still show a clear effect. If `psense` flags prior sensitivity here, it means the skeptical prior is actively pulling the posterior toward zero -- which is informative for your colleague's concern.

### Step 6: Cross-scenario comparison

We overlay the posterior distributions of `beta_program` (and the derived rate ratio) across all three scenarios. If the posteriors substantially overlap and all point to a similar rate ratio, the conclusion is robust regardless of prior choice. If they diverge materially, the prior matters and this should be reported transparently.

## How to respond to your colleague

The comparison table and sensitivity diagnostics provide a direct answer:

1. **If posteriors agree across scenarios**: "The data is strong enough that the prior choice does not materially affect the conclusion. Even under a skeptical prior centered at no effect, the posterior supports a rate ratio near 0.75."

2. **If the informative prior pulls results away from the weakly informative**: "The informative prior does influence the posterior, but the direction of the effect is consistent across all priors. The magnitude shifts by [X]. For maximum transparency, we report results under both the informative and weakly informative priors."

3. **If psense flags the informative prior**: "This is expected -- an informative prior *should* flag as sensitive when it genuinely encodes knowledge. The key question is whether it conflicts with the data (both CJS > 0.05) or simply contributes information (prior CJS > 0.05, likelihood CJS < 0.05). In the latter case, the prior is doing its job."

## Limitations

- Synthetic data: the true DGP is known and matches our model. Real data may have overdispersion (warranting NegBinomial), time trends, seasonal patterns, or non-random program rollout.
- We do not model temporal autocorrelation within factories.
- The program is assumed to have an immediate, constant effect -- no ramp-up or fade-out.
- Factory sizes are fixed over time, which may not hold in practice.

## Files produced

| File | Description |
|------|-------------|
| `model.py` | Complete runnable script |
| `idata_informative.nc` | InferenceData for informative prior scenario |
| `idata_weakly_informative.nc` | InferenceData for weakly informative prior scenario |
| `idata_skeptical.nc` | InferenceData for skeptical prior scenario |
| `trace_*.png` | Rank plots per scenario |
| `ppc_*.png` | Posterior predictive check plots per scenario |
| `prior_sensitivity_comparison.png` | Side-by-side posterior comparison across scenarios |
