# Workplace Safety Program: Prior Sensitivity Analysis

## Executive summary

We model the effect of a workplace safety program on monthly injury counts across 30 factories using a hierarchical Poisson regression. The key question is whether conclusions about the program's effectiveness are robust to prior choice, or whether an informative prior (based on a previous study suggesting a ~25% reduction) is driving the results. We fit three models with different priors on the treatment effect and compare posteriors.

## Data description

- **Source**: Synthetic data matching the stated scenario
- **Sample size**: N = 1,440 observations (30 factories x 48 months)
- **Structure**: 24 months pre-intervention, 24 months post-intervention
- **Key variables**: monthly injury counts, factory ID, treatment indicator, number of employees (exposure)
- **Notable features**: Hierarchical structure (observations nested in factories), count outcome with exposure offset

## Model specification

### Generative story

Each factory has a baseline injury rate per employee per month that varies around a grand mean. When the safety program is introduced (at month 25), it multiplies the injury rate by a treatment factor (the rate ratio). Monthly injuries are Poisson-distributed with rate = baseline_rate x n_employees x treatment_effect.

### Mathematical notation

```
injuries_fm ~ Poisson(lambda_fm)
log(lambda_fm) = log(n_employees_f) + alpha_f + beta * treated_fm
alpha_f = mu_alpha + sigma_alpha * z_f        (non-centered parameterization)
z_f ~ Normal(0, 1)
mu_alpha ~ Normal(log(0.02), 0.5)
sigma_alpha ~ Gamma(2, 5)
```

The treatment effect prior varies across models (see below).

### Prior choices

| Parameter | Prior | Justification |
|-----------|-------|---------------|
| mu_alpha | Normal(log(0.02), 0.5) | Grand mean ~2 injuries per 100 employees/month; sigma=0.5 allows ~1-4% rates |
| sigma_alpha | Gamma(2, 5) | Mean 0.4 on log-scale; avoids near-zero (no hierarchy needed if zero) |
| alpha_offset (z_f) | Normal(0, 1) | Non-centered parameterization to avoid funnel geometry |
| beta (informative) | Normal(log(0.75), 0.1) | Previous study: ~25% reduction. Tight: 94% HDI on RR ~ [0.61, 0.92] |
| beta (weakly informative) | Normal(0, 0.5) | Centered at no effect; allows RR ~ [0.37, 2.72] at 94% HDI |
| beta (skeptical) | Normal(0, 0.15) | Centered at no effect, moderately tight: RR ~ [0.74, 1.35] at 94% HDI |

### Non-centered parameterization

We use non-centered factory intercepts (alpha = mu + sigma * z, z ~ Normal(0,1)) to avoid the funnel geometry that commonly causes divergences in hierarchical models.

## Results (expected)

### Convergence diagnostics

All three models should pass convergence checks:
- R-hat <= 1.01 for all parameters
- ESS bulk and tail >= 100 x number of chains
- Zero divergences (non-centered parameterization + Gamma prior on sigma avoids funnels)

We verify with `arviz_stats.diagnose(idata)` and visual rank plots.

### Prior predictive check

The informative model's prior predictive check verifies:
1. Total monthly injury counts fall in a plausible range (not negative, not astronomically large)
2. The prior on rate ratio is concentrated around 0.75 as intended, with most mass below 1.0

If prior predictions span implausible ranges, we would tighten priors before proceeding.

### Posterior predictive check and calibration

We run `az.plot_ppc()` and `plot_ppc_pit()` (PIT calibration) on the informative model. For a well-specified Poisson model with appropriate rates:
- PPC should show the model can reproduce the observed distribution of injury counts
- PIT should be approximately uniform (no systematic miscalibration)

If the PIT shows issues (e.g., overdispersion), we would consider switching to NegativeBinomial.

### Prior sensitivity analysis (power-scaling)

`psense_summary(idata)` computes the Cumulative Jensen-Shannon (CJS) divergence for prior and likelihood perturbations. The interpretation framework:

| Pattern | Prior CJS | Likelihood CJS | Meaning |
|---------|-----------|----------------|---------|
| Low sensitivity | < 0.05 | < 0.05 | Robust -- the ideal outcome |
| Prior-data conflict | > 0.05 | > 0.05 | Prior and data disagree |
| Strong prior / weak likelihood | > 0.05 | < 0.05 | Prior dominates posterior |
| Likelihood-driven | < 0.05 | > 0.05 | Data dominates -- usually fine |

**Expected outcome for beta (informative prior)**: With 30 factories x 24 post-treatment months = 720 post-treatment observations, the likelihood should be strong. Whether beta flags as prior-sensitive depends on whether the prior (RR ~ 0.75) aligns with the data.

- If the data agrees with the prior (true RR near 0.75): low prior CJS -- prior and data reinforce each other.
- If there were prior-data conflict: both CJS values would be elevated, warranting investigation.

`plot_psense_dist()` shows the posterior at three power-scaling levels (alpha = 0.8, 1.0, 1.25) to visualize the direction and magnitude of any shift.

### Cross-prior comparison

The strongest evidence for robustness comes from comparing posteriors across all three prior specifications:

**If conclusions are robust (expected with this dataset)**:
- All three posteriors for the rate ratio should substantially overlap
- The 94% HDIs should all exclude 1.0 (all agree the program reduces injuries)
- The posterior means should be similar, with the informative prior pulling slightly toward 0.75 and the skeptical prior pulling slightly toward 1.0
- P(program reduces injuries) should be high (> 0.95) for all three models

**If the informative prior were driving results** (not expected here, but the pattern to watch for):
- The informative model would show a rate ratio near 0.75 with a narrow HDI
- The weakly informative model would show a rate ratio near 1.0 or with an HDI crossing 1.0
- The skeptical model would be even more equivocal
- This would mean the conclusion depends on prior choice -- a real problem

## Addressing the colleague's challenge

The colleague's concern -- "your prior might be driving the results" -- maps to the sensitivity framework as follows:

1. **Power-scaling analysis** (psense_summary): If beta shows Prior CJS > 0.05 with Likelihood CJS < 0.05, the prior is indeed dominating. If both are low, the conclusion is robust.

2. **Cross-prior comparison**: If all three models (informative, weak, skeptical) agree on the direction and approximate magnitude of the effect, the conclusion does not depend on the prior. This is the most compelling evidence for a skeptical colleague.

3. **The right framing**: A sensitivity flag on an informative prior is not automatically a problem. If the prior is well-justified (previous study with similar intervention), it *should* matter when data is limited. The question is whether the data alone (weakly informative model) supports the same conclusion.

With 720 post-treatment observations across 30 factories, we expect the data to be informative enough that all three priors converge to similar posteriors -- the informative prior adds precision but does not change the qualitative conclusion.

## Limitations

- **Synthetic data**: Results here validate the methodology, not the actual program effect. With real data, the sensitivity comparison becomes truly informative.
- **Poisson assumption**: Real injury counts may exhibit overdispersion. If PIT calibration flags issues, switch to NegativeBinomial.
- **No time trends**: The model assumes a step change at intervention. Gradual implementation or secular trends would require a more complex specification (e.g., interrupted time series with slope change).
- **No confounders**: We assume no other changes coincided with the program. In practice, this is rarely guaranteed -- causal identification depends on the study design, not just the statistical model.
- **Factory-level treatment**: All factories were treated simultaneously. Without a staggered rollout or control group, identification relies on the before-after contrast, which is vulnerable to period effects.
