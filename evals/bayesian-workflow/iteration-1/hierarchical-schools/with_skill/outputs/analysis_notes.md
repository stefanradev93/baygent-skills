# Hierarchical Model for Student Test Scores Across Schools

## Executive Summary

This analysis uses a Bayesian hierarchical (multilevel) model to decompose variation in student test scores into between-school and within-school components, and to produce school-level estimates that intelligently handle unequal sample sizes through partial pooling. Schools with few students (as few as 5) have their estimates shrunk toward the global mean, protecting against overfitting; schools with many students retain estimates closer to their observed averages.

## Data Description

- **Source**: Synthetic data generated for demonstration purposes.
- **Sample size**: N = 200 students across 15 schools.
- **Key variable**: Test scores (continuous, roughly in the 0--100 range).
- **Grouping structure**: Students nested within schools.
- **Notable features**: Highly imbalanced group sizes -- some schools have only 5 students, while others have 20+. This imbalance is precisely the scenario where hierarchical models shine compared to separate per-school estimates (no pooling) or ignoring school structure entirely (complete pooling).

The synthetic data was generated with:
- True global mean: 65
- True between-school SD: 8
- True within-school SD: 12

These "true" values let us verify that the model recovers the known parameters, serving as a sanity check.

## Model Specification

### Generative Story

We imagine the following data-generating process:

1. There is a population of schools, each with a "true" average test score. These school means are not identical -- they vary around a global average.
2. Each school's true mean is drawn from a Normal distribution centered on the global mean, with spread governed by the between-school standard deviation.
3. Each student's observed score within a school is drawn from a Normal distribution centered on that school's true mean, with spread governed by the within-school standard deviation.

This is a **varying-intercepts** model -- each school gets its own intercept (mean), but these intercepts are related through a shared population distribution.

### Mathematical Notation

```
y_{ij} ~ Normal(mu_j, sigma_within)         [student i in school j]
mu_j   ~ Normal(mu_global, sigma_between)   [school j's true mean]

mu_global     ~ Normal(50, 20)               [population mean]
sigma_between ~ Gamma(2, 0.5)                [between-school SD]
sigma_within  ~ Gamma(2, 0.2)                [within-school SD]
```

Equivalently, using the non-centered parameterization (which is what is actually implemented):

```
mu_raw_j ~ Normal(0, 1)
mu_j = mu_global + mu_raw_j * sigma_between
```

### Prior Choices

| Parameter       | Prior            | Justification |
|-----------------|------------------|---------------|
| `mu_global`     | Normal(50, 20)   | Test scores typically span 0--100. Centering at 50 with SD=20 places 95% prior mass in roughly [10, 90]. This is weakly informative -- it rules out nonsensical values (scores of -500 or +1000) while letting the data dominate. |
| `sigma_between` | Gamma(2, 0.5)    | Mean of 4, mode of 2. Allows values up to ~15--20 but places low mass near zero. Avoiding zero is important: if the between-school SD collapses to zero, the model degenerates into complete pooling, losing the hierarchical structure. We want the data, not the prior, to determine whether schools truly differ. |
| `sigma_within`  | Gamma(2, 0.2)    | Mean of 10, mode of 5. Student-level variation is expected to be larger than school-level variation. This prior allows a wide range of within-school spreads. |
| `mu_raw`        | Normal(0, 1)     | Standard normal offsets for the non-centered parameterization. These are not "priors" in the substantive sense -- they are part of the reparameterization that eliminates funnel geometry. |

**Why these priors and not others?**

- We avoid flat/diffuse priors (e.g., Normal(0, 1000)) because they place excessive mass on implausible values and can destabilize sampling.
- We avoid Exponential priors on the scale parameters because, while valid, they place maximum density at zero, which can encourage the model to collapse hierarchical variation.
- The Gamma priors with `alpha=2` are the recommended default from the skill guide for scale parameters -- they avoid near-zero values while remaining weakly informative.

### Centered vs. Non-Centered Parameterization

We use the **non-centered** parameterization, following the skill's rule of thumb: "Start with non-centered. Switch to centered only if non-centered shows poor ESS AND groups have substantial data (50+ observations each)."

Since some of our schools have only 5 students, the centered parameterization would likely produce a "funnel" geometry in the joint posterior of (sigma_between, mu_school), causing divergent transitions. The non-centered parameterization eliminates this by sampling standardized offsets (mu_raw) and reconstructing the school means deterministically.

## Prior Predictive Check

The code generates prior predictive samples and plots them. What to look for:

- **Prior predictive scores** should span a plausible range (roughly 0--100, perhaps slightly beyond). If prior predictions produce scores of -200 or +500, the priors are too diffuse.
- **Prior predictive school means** should similarly be in a reasonable range.

If more than 10% of prior predictive samples are clearly implausible, the priors should be tightened before proceeding to inference.

## Convergence Diagnostics

After sampling, the code runs the full diagnostic checklist mandated by the workflow:

1. **R-hat (split R-hat)**: Must be <= 1.01 for all parameters. Values above 1.01 indicate the chains have not mixed and results are unreliable.

2. **ESS (bulk and tail)**: Must be >= 100 * number of chains (i.e., >= 400 with 4 chains). Low ESS means the effective number of independent samples is too small for reliable summaries.

3. **Divergences**: Ideally zero. Even a handful (10+) can bias results. If present, increase `target_accept` (up to 0.99) or reparameterize.

4. **Trace/rank plots**: Rank plots should appear uniform across chains with no spikes or gaps. The code generates these for both global parameters and school-level means.

5. **Energy plot**: The marginal energy and energy transition distributions should overlap. A large gap indicates poor exploration.

**If any diagnostic fails**: Do NOT interpret the results. Fix the issue first. The most common fixes for hierarchical models are:
- Switch parameterization (centered <-> non-centered)
- Increase `target_accept` to 0.95 or 0.99
- Strengthen priors on scale parameters to avoid the near-zero region

## Model Criticism

### Posterior Predictive Check (PPC)

The posterior predictive check is the most important model criticism tool. It simulates new datasets from the fitted model and compares them to the observed data. The code uses `az.plot_ppc()` to overlay 100 posterior predictive samples on the observed score distribution.

**What to look for**:
- The posterior predictive distribution should envelop the observed data.
- Check shape (unimodal? symmetric?), spread, and tails.
- If the posterior predictive consistently misses a feature of the data (e.g., bimodality, heavy tails), the model is misspecified and needs revision.

### LOO-CV (Leave-One-Out Cross-Validation)

LOO-CV via PSIS estimates out-of-sample predictive accuracy. Key outputs:
- **elpd_loo**: Expected log pointwise predictive density. Higher (less negative) is better.
- **p_loo**: Effective number of parameters. If much larger than the actual count, the model may be misspecified.
- **Pareto k**: Per-observation diagnostic. Values > 0.7 flag observations where the LOO estimate is unreliable. These are often outliers or poorly-fit students worth investigating.

## Results Interpretation

### Variance Partition Coefficient (VPC / ICC)

The central quantity of interest: what fraction of total variation is between schools?

```
VPC = sigma_between^2 / (sigma_between^2 + sigma_within^2)
```

The code computes a full posterior distribution for VPC, not just a point estimate. This is reported with a 94% HDI.

**Interpreting VPC**:
- VPC near 0 means almost all variation is within schools (schools are similar; differences between students within a school dominate).
- VPC near 1 means almost all variation is between schools (students within a school are similar to each other; the school you attend matters enormously).
- For the synthetic data (true sigma_between=8, true sigma_within=12): true VPC = 64 / (64 + 144) = 0.31. About 31% of variation is between schools.

### School-Level Estimates and Shrinkage

The forest plot shows each school's posterior mean with its 94% HDI. The shrinkage plot is the hallmark visualization of partial pooling:

- **Arrows** connect each school's observed sample mean (no-pooling estimate) to its posterior mean (partial-pooling estimate).
- **Direction**: Arrows always point toward the global mean.
- **Magnitude**: Schools with fewer students show longer arrows (more shrinkage), while schools with many students show shorter arrows (less shrinkage).
- **Dot size**: Proportional to sample size, making the relationship between sample size and shrinkage visually obvious.

This shrinkage is not a bug -- it is a feature. For a school with only 5 students, the sample mean is very noisy. The hierarchical model recognizes this and pulls the estimate toward the global mean, producing a more reliable estimate. This is the Bayesian analog of James-Stein estimation and is guaranteed to improve out-of-sample prediction on average.

### Group-Level SD Posterior

The code plots the posterior distributions of sigma_between and sigma_within. An important check: if sigma_between's posterior piles up near zero, the data does not support school-level variation, and the model is effectively doing complete pooling. This would suggest that school identity does not matter for test scores.

## Limitations

1. **Synthetic data**: This analysis uses generated data. Real-world data may have features not captured here (non-normality, outliers, covariates that explain between-school differences).

2. **Normal likelihood assumed**: Test scores may not be perfectly Normal. If scores are bounded (0--100), a truncated or Beta likelihood might be more appropriate. The posterior predictive check is the key diagnostic here -- if the Normal model produces predictions outside the plausible range, consider alternatives.

3. **No covariates**: This model includes only varying intercepts. If school-level predictors (e.g., funding, class size) or student-level predictors (e.g., socioeconomic status) are available, they could explain some of the between-school variation and should be included as varying slopes.

4. **Single level of nesting**: If classrooms within schools matter, a three-level model (student within classroom within school) might be warranted, though the skill guide cautions against going overboard with too many hierarchy levels.

5. **Prior sensitivity**: For controversial conclusions, it would be worth re-running the model under alternative priors (e.g., Exponential(1) for scale parameters, or tighter/wider Normal for the global mean) to verify that the results are robust.

## File Outputs

The code produces the following files in the `outputs/` directory:

| File | Description |
|------|-------------|
| `prior_predictive_check.png` | Prior predictive distributions for scores and school means |
| `trace_rank_plots_global.png` | Rank plots for global parameters (mu_global, sigma_between, sigma_within) |
| `trace_rank_plots_schools.png` | Rank plots for school-level means |
| `energy_plot.png` | Energy diagnostic plot |
| `posterior_predictive_check.png` | PPC overlay of simulated vs. observed scores |
| `pareto_k_plot.png` | LOO-CV Pareto k diagnostic values |
| `school_forest_plot.png` | Forest plot of school-level mean estimates with 94% HDI |
| `shrinkage_plot.png` | Shrinkage from observed means to posterior means |
| `vpc_posterior.png` | Posterior distribution of the Variance Partition Coefficient |
| `variance_components_posterior.png` | Posteriors for sigma_between and sigma_within |
| `model_graph.png` | Graphical model diagram (requires graphviz) |
| `school_model_idata.nc` | Saved InferenceData for further analysis |

## Software Requirements

- Python 3.10+
- PyMC 5+ (latest recommended)
- ArviZ 0.17+ (or ArviZ 1.0+ for advanced calibration plots)
- nutpie (for faster NUTS sampling; falls back to PyMC's default sampler if unavailable)
- NumPy, pandas, matplotlib
- graphviz (optional, for model graph rendering)
