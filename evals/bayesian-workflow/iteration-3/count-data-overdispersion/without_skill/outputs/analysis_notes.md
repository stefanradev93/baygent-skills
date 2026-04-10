# Bayesian Analysis of Customer Support Ticket Counts

## 1. Problem Statement

We are modeling the daily count of customer support tickets for a SaaS company over 180 days. The key features of this data are:

- **Count data**: tickets per day are non-negative integers.
- **Overdispersion**: the variance appears substantially larger than the mean (most days see 5-15 tickets, but spikes reach 40+). This is the central statistical challenge.
- **Predictors**: day of the week (categorical, 7 levels) and whether a product release occurred that day (binary).

## 2. Likelihood Choice: Why Negative Binomial over Poisson

### The Poisson assumption and why it fails here

The Poisson distribution is the default starting point for count data. However, it has a strict property: the variance equals the mean (equidispersion). In our data the variance-to-mean ratio is well above 1 — some days spike to 40+ tickets while the typical day sees around 8-12. This is **overdispersion**.

Fitting a Poisson model to overdispersed data leads to:
- Posterior intervals that are too narrow (false precision).
- Poor posterior predictive checks — the model cannot reproduce the heavy right tail.
- Misleading inferences about predictor effects.

### The Negative Binomial as the natural extension

The Negative Binomial distribution adds an overdispersion parameter (commonly called `alpha` or `n`) that decouples the variance from the mean. It is parameterized as:

- `mu`: the conditional mean (same role as in Poisson).
- `alpha`: the overdispersion parameter. The variance is `mu + mu^2 / alpha`. As `alpha -> infinity`, the Negative Binomial converges to the Poisson.

This makes it the minimal extension of Poisson that handles overdispersion. It is widely used in practice for count data with extra-Poisson variation (e.g., insurance claims, disease counts, web traffic).

### Why not zero-inflated or hurdle models?

The problem description does not mention excess zeros — most days have 5-15 tickets, suggesting zeros are rare. A zero-inflated model would add unnecessary complexity. If the data did show excess zeros, we would revisit this choice.

## 3. Model Structure

### Linear predictor (log link)

We use a log link function, which is standard for count GLMs:

```
log(mu_i) = intercept + beta_dow * X_dow[i] + beta_release * X_release[i]
```

- **Intercept**: the log expected ticket count on a Monday (reference day) with no product release.
- **Day-of-week effects** (`beta_dow`): 6 coefficients (Tuesday through Sunday), with Monday as the reference level. These capture weekly seasonality.
- **Release effect** (`beta_release`): a single coefficient capturing the multiplicative impact of a product release on ticket volume.

The log link ensures `mu` is always positive (a requirement for count distributions) and gives the coefficients a multiplicative interpretation: `exp(beta)` is the factor by which the mean count changes per unit change in the predictor.

## 4. Prior Choices

### Intercept: Normal(2.0, 1.0)

- `exp(2.0) ~ 7.4` tickets, which is a reasonable central estimate given the reported range of 5-15 typical tickets.
- A standard deviation of 1.0 on the log scale means the prior covers roughly `exp(0) = 1` to `exp(4) = 55` tickets at +/- 2 sigma. This is weakly informative: it concentrates mass in a plausible range but does not dominate the data.

### Day-of-week effects: Normal(0.0, 0.5) each

- Centered at 0 (no effect), which encodes the mild prior belief that any given day is not dramatically different from Monday.
- A standard deviation of 0.5 on the log scale means the prior allows individual days to differ by a factor of roughly `exp(1) ~ 2.7x` at +/- 2 sigma. This is generous enough for plausible day-of-week variation.

### Release effect: Normal(0.0, 1.0)

- Centered at 0 (no effect), but with a wider standard deviation than the day-of-week effects because we expect releases could have a large impact.
- At +/- 2 sigma, this allows multiplicative effects from `exp(-2) ~ 0.14x` to `exp(2) ~ 7.4x`. Given domain knowledge that releases can cause substantial spikes, this range is appropriate.

### Overdispersion parameter: HalfNormal(10.0)

- `alpha > 0` by definition, so we use a half-normal prior.
- `sigma=10` is weakly informative: it places most prior mass on `alpha` values between 0 and ~20, which covers everything from extreme overdispersion (alpha ~ 1) to mild overdispersion (alpha ~ 20). The data will dominate this prior.

## 5. Workflow

The analysis follows the standard Bayesian workflow:

1. **Data generation and EDA**: Synthetic data is generated from a known Negative Binomial DGP. Histograms and bar charts provide initial visual understanding.

2. **Prior predictive check**: Before fitting, we sample from the prior predictive distribution to verify that our priors generate data in a plausible range. If prior predictions were consistently producing, say, thousands of tickets per day, we would tighten our priors.

3. **Model fitting**: Both Poisson and Negative Binomial models are fit using MCMC (NUTS sampler) with 4 chains, 1000 tuning steps, and 2000 draws each. The `target_accept=0.9` setting helps with the geometry of count models.

4. **Convergence diagnostics**: We check R-hat (< 1.01) and effective sample size (ESS > 400 for both bulk and tail) to ensure the chains have converged and mixed well. Trace plots provide visual confirmation.

5. **Posterior predictive checks**: We compare simulated data from the fitted model against the observed data. The Poisson model should fail to capture the overdispersion (too narrow), while the Negative Binomial should match the observed distribution well.

6. **Model comparison**: LOO-CV (Leave-One-Out Cross-Validation via Pareto-Smoothed Importance Sampling) provides a principled comparison. We expect the Negative Binomial to have a substantially better (higher) ELPD.

7. **Interpretation**: Posterior summaries are converted to the response scale (exponentiated) for interpretability.

## 6. Diagnostics Interpretation

### What to look for

- **R-hat**: Values should all be below 1.01. Values above 1.05 indicate serious convergence problems. All chains should be exploring the same region of parameter space.

- **ESS (Effective Sample Size)**: Both `ess_bulk` and `ess_tail` should exceed 400 (ideally much more). Low ESS means the posterior estimates are noisy despite having many nominal samples.

- **Trace plots**: The "caterpillar" appearance (well-mixed, stationary chains) is desired. Trends, multimodality, or sticky chains would be red flags.

- **Divergences**: PyMC reports the number of divergent transitions. Any divergences suggest the sampler is struggling with the posterior geometry and results may be biased. If present, consider reparameterization or increasing `target_accept`.

### Posterior predictive checks

- The Poisson model should produce posterior predictive distributions that are too concentrated around the mean, failing to reproduce the heavy tail of observed data.
- The Negative Binomial model should produce distributions that visually match the observed histogram, including the right tail.

## 7. Expected Results

Given the data-generating process:

- **Intercept**: Should recover approximately 2.1 (log scale), corresponding to ~8 baseline tickets on a Monday with no release.

- **Release effect**: Should recover approximately 1.0 (log scale), meaning releases roughly triple (`exp(1) ~ 2.7x`) the expected ticket count. This is the primary driver of spikes.

- **Day-of-week effects**: Tuesday through Friday should show small positive or near-zero effects. Saturday and Sunday should show negative effects (fewer tickets on weekends), with Sunday being the lowest.

- **Alpha (overdispersion)**: Should recover approximately 3.0. This moderate value indicates substantial overdispersion — the variance is much larger than the mean.

- **Model comparison**: The Negative Binomial should decisively outperform the Poisson model in LOO-CV, reflecting its ability to capture the overdispersion in the data.

## 8. Key Takeaways for the SaaS Company

1. **Product releases are the primary driver of ticket spikes.** The multiplicative effect is expected to be around 2.5-3x, meaning a release day generates roughly triple the normal ticket volume.

2. **Weekend ticket volumes are lower.** This is consistent with fewer users being active on weekends.

3. **Even after accounting for releases and day-of-week, there is substantial residual overdispersion.** This means that some day-to-day variability in ticket counts is not explained by these two predictors alone. Additional predictors (e.g., marketing campaigns, outages, feature adoption metrics) could be explored in future iterations.

4. **The Poisson model gives misleadingly precise estimates.** Using a Negative Binomial is essential for honest uncertainty quantification in this setting.
