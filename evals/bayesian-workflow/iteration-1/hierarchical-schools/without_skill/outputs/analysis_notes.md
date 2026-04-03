# Hierarchical Model for School Test Scores: Analysis Notes

## Problem Statement

We have test scores for 200 students across 15 schools, with unequal sample sizes (some schools have as few as 5 students). We want to:

1. **Decompose variance**: How much of the total variation in scores is between schools vs. within schools?
2. **Estimate school means**: Get school-level estimates that properly account for differing sample sizes.

## Why a Hierarchical (Multilevel) Model?

There are three naive approaches to this problem, and all have serious drawbacks:

- **Complete pooling** (ignore schools, estimate a single grand mean): Throws away real differences between schools. Useless for understanding school effects.
- **No pooling** (estimate each school mean independently): The observed mean for a school with n=5 students has enormous uncertainty, yet it gets treated with the same confidence as a school with n=25. Small-school estimates are noisy and unreliable.
- **Hierarchical / partial pooling**: School means are modeled as draws from a shared population distribution. This allows small schools to "borrow strength" from the ensemble. School estimates are **shrunk** toward the grand mean, with small schools shrunk more than large schools. This is optimal in terms of out-of-sample prediction.

## Model Specification

```
Likelihood:
    y_ij ~ Normal(mu_j, sigma_within)

School-level (non-centered):
    school_offset_j ~ Normal(0, 1)
    mu_j = mu_global + sigma_between * school_offset_j

Hyperpriors:
    mu_global ~ Normal(50, 25)
    sigma_between ~ HalfNormal(20)
    sigma_within ~ HalfNormal(20)

Derived:
    ICC = sigma_between^2 / (sigma_between^2 + sigma_within^2)
```

Where:
- `y_ij` is the test score of student i in school j
- `mu_j` is the mean score for school j
- `sigma_within` captures student-to-student variation within a school
- `sigma_between` captures school-to-school variation in means
- `mu_global` is the overall average across all schools

## Prior Choices and Justification

### `mu_global ~ Normal(50, 25)`

- **Rationale**: Test scores are often on a 0-100 scale. Centering at 50 with SD=25 makes the prior weakly informative -- it covers the full plausible range of average scores (roughly 0 to 100 at 2 SDs) without being so vague as to cause sampling problems.
- **Prior predictive check**: This generates plausible overall means. It is not strongly informative -- the data will dominate with 200 observations.

### `sigma_between ~ HalfNormal(20)`

- **Rationale**: This is the between-school standard deviation. A HalfNormal(20) prior puts most mass below about 35 points, which is generous -- it allows for large school-to-school differences while making extreme values (SD > 50) very unlikely. The half-normal is a good default for variance components because it is proper, has zero mass at zero (so it can detect the case where there is no school effect), and has a gentle tail.
- **Alternative considered**: HalfCauchy or Exponential priors. HalfCauchy has heavier tails which can cause sampling issues when the true value is small. Exponential concentrates more mass near zero. HalfNormal is a good middle ground.

### `sigma_within ~ HalfNormal(20)`

- **Rationale**: Same logic as sigma_between. Student-level variation within schools is typically larger than between-school variation, and this prior comfortably covers the plausible range.

### Why Non-Centered Parameterization?

The standard ("centered") parameterization `mu_j ~ Normal(mu_global, sigma_between)` creates a problematic **funnel geometry** in the posterior when `sigma_between` is small. When the group-level SD is near zero, the school means must all be near the grand mean, creating a narrow funnel that HMC struggles to traverse.

The non-centered parameterization:
```
school_offset_j ~ Normal(0, 1)
mu_j = mu_global + sigma_between * school_offset_j
```
decouples the school offsets from the variance parameter, eliminating the funnel. This is especially important with only 15 schools -- a relatively small number of groups.

## Diagnostics Interpretation

The analysis produces several diagnostic outputs. Here is what to look for:

### R-hat (Gelman-Rubin Statistic)
- **Target**: All R-hat values should be < 1.01
- **What it measures**: Convergence of chains. Compares within-chain variance to between-chain variance. Values near 1.0 indicate chains have mixed well and converged to the same distribution.
- **If problematic**: Run longer chains, increase tuning, or reparameterize.

### Effective Sample Size (ESS)
- **Target**: ESS > 400 for all parameters (at minimum; higher is better)
- **What it measures**: The number of effectively independent samples, accounting for autocorrelation in the MCMC chain. An ESS of 400 gives a Monte Carlo standard error of about 5% of the posterior SD.
- **If problematic**: Run more draws or reparameterize.

### Divergences
- **Target**: Zero divergences
- **What they indicate**: The sampler encountered regions of the posterior where the Hamiltonian dynamics break down, usually due to high curvature. This causes biased estimates. The non-centered parameterization and `target_accept=0.95` should prevent this.
- **If problematic**: Increase `target_accept` (up to 0.99), reparameterize, or simplify the model.

### Trace Plots
- **What to look for**: "Fuzzy caterpillars" -- chains should look like white noise with no trends, jumps, or stuck regions. All 4 chains should overlap and cover the same range.

### Rank Plots
- More informative than trace plots. For well-mixing chains, the rank histograms should look roughly uniform across chains. Non-uniform shapes indicate poor mixing.

### Energy Plot
- Checks for pathological behavior in HMC. The marginal energy and energy transition distributions should overlap well. A large gap indicates the sampler is struggling.

### Prior Predictive Check
- Verifies that the priors generate data in a plausible range. Scores should generally fall in the 0-100 range, though some spread beyond is acceptable for weakly informative priors.

### Posterior Predictive Check
- The distribution of simulated data from the posterior should resemble the observed data. Systematic discrepancies point to model misspecification (e.g., heavy tails, skewness, or multimodality in the real data that the Normal likelihood cannot capture).

## Results Summary

### Variance Decomposition

The key output is the **Intraclass Correlation Coefficient (ICC)**:

```
ICC = sigma_between^2 / (sigma_between^2 + sigma_within^2)
```

- **ICC near 0**: Almost all variation is within schools. Schools are not meaningfully different.
- **ICC near 1**: Almost all variation is between schools. Students within the same school are very similar.
- **Typical values in education**: ICC values of 0.10 to 0.25 are common in educational research -- meaning 10-25% of score variation is between schools.

With the synthetic data (true sigma_between=10, true sigma_within=12):
- True ICC = 100 / (100 + 144) = 0.410
- This is deliberately higher than typical to make the school effect clearly visible in the analysis.

### Shrinkage Effect

The shrinkage plot is the most important visualization. It shows:

1. **Observed means** (red dots): The raw sample means for each school.
2. **Posterior means** (blue dots): The model's estimates after partial pooling.
3. **Arrows**: Show the direction and magnitude of shrinkage.

Key patterns:
- **Small schools (n=5) are shrunk the most** toward the grand mean. Their observed means are noisy, so the model "trusts" the ensemble more.
- **Large schools are shrunk less** -- their larger samples provide more reliable estimates.
- **Schools with extreme observed means are shrunk the most** in absolute terms.
- The **true means** (green diamonds) demonstrate that the partially-pooled posterior estimates are typically closer to the truth than the raw observed means, especially for small schools.

### School-Level Estimates

The forest plot shows the posterior distributions for each school's mean, with 94% Highest Density Intervals (HDI). Schools with fewer students will have wider intervals, reflecting greater uncertainty.

## Limitations and Extensions

1. **Normal likelihood**: We assume scores are normally distributed within schools. If scores are bounded (0-100), have floor/ceiling effects, or are heavily skewed, a different likelihood (e.g., Beta, truncated Normal, or ordinal) may be more appropriate.

2. **No covariates**: This model has no student-level or school-level predictors. Adding predictors (e.g., socioeconomic status, school resources) would help explain the between-school variance and improve predictions.

3. **Homogeneous within-school variance**: We assume sigma_within is the same for all schools. A more flexible model could allow each school to have its own within-school variance.

4. **Only 15 schools**: With a small number of groups, estimates of sigma_between are uncertain. The prior on sigma_between matters more than it would with 50+ schools.

5. **Cross-sectional only**: This is a single time point. Longitudinal data would allow modeling growth trajectories.
