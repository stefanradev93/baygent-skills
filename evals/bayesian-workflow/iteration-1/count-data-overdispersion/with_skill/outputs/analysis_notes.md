# Bayesian Analysis of Customer Support Ticket Counts

## Executive Summary

Customer support ticket counts per day are modeled as a function of day-of-week and whether a product release occurred. The data exhibits overdispersion (variance substantially exceeding the mean), which makes a Poisson model inadequate. A Negative Binomial model is used to capture this extra-Poisson variation. Model comparison via LOO-CV is expected to strongly favor the Negative Binomial over the Poisson. Product releases are expected to roughly double the daily ticket rate, and day-of-week effects show a moderate weekly pattern.

## Data Description

- **Source**: Synthetic data generated to mimic the described scenario (180 days of ticket counts).
- **Sample size**: N = 180 days.
- **Key variables**:
  - `tickets`: Count of customer support tickets per day (outcome). Most days 5--15, some days 40+.
  - `day_of_week`: Integer 0--6 (Monday--Sunday). Categorical predictor.
  - `product_release`: Binary indicator (0/1). About 10% of days have a release.
- **Notable features**: The variance-to-mean ratio exceeds 1 (likely around 2--4), indicating overdispersion. This is the central modeling challenge.

## Approach: Why Negative Binomial?

### The overdispersion problem

A Poisson distribution constrains Var(Y) = E(Y). When the observed variance exceeds the observed mean -- as described here (most days 5--15 tickets but spikes to 40+) -- the Poisson assumption is violated. Fitting a Poisson model to overdispersed data produces:

1. **Artificially narrow credible intervals** -- the model is overconfident because it underestimates the true variability.
2. **Poor posterior predictive checks** -- the model cannot reproduce the observed spread or the tail of high-count days.
3. **Unreliable LOO-CV** -- many observations will have high Pareto k values because the model assigns them very low probability.

### The Negative Binomial solution

The Negative Binomial (NegBin) adds an overdispersion parameter `alpha` such that:

    Var(Y) = mu + mu^2 / alpha

- When `alpha` is large (> ~20), the NegBin collapses to the Poisson.
- When `alpha` is small (1--5), the variance can be substantially larger than the mean.
- This extra flexibility allows the model to accommodate the observed spread without distorting the regression coefficients.

The NegBin is the natural first step beyond Poisson for count data with overdispersion. It is more appropriate than alternatives like a zero-inflated model (the problem description does not mention excess zeros) or a quasi-Poisson approach (which is frequentist and does not provide a proper generative model).

### Model structure

Both models use the same regression structure with a log link:

    log(mu_i) = intercept + beta_dow[dow_i] + beta_release * release_i

- **Poisson model**: tickets_i ~ Poisson(mu_i)
- **NegBin model**: tickets_i ~ NegativeBinomial(mu_i, alpha)

This allows direct comparison of predictive performance via LOO-CV.

## Prior Choices and Rationale

All priors are weakly informative, designed to rule out implausible values while letting the data dominate.

### Intercept: Normal(mu=2.0, sigma=1.0)

- **Scale**: This is on the log scale (log link function). exp(2.0) = 7.4 tickets/day as the prior central estimate.
- **Range**: 95% prior mass spans exp(0) to exp(4) = 1 to 55 tickets/day, which comfortably covers the described range of 5--40.
- **Justification**: Centered near the plausible baseline rate. A sigma of 1 on the log scale is moderately informative but does not unduly constrain the intercept.

### Day-of-week effects: ZeroSumNormal(sigma=0.5)

- **Parameterization**: ZeroSumNormal ensures the 7 day-of-week effects sum to zero. This provides identifiability (no confounding with the intercept) and is more principled than reference encoding when no single day is a natural baseline.
- **Scale**: sigma=0.5 on the log scale means each day can shift the rate by roughly exp(0.5) = 1.65x up or down from the overall mean. This is reasonable -- we do not expect day-of-week variation to exceed a factor of ~3x in either direction for support tickets.
- **Alternative considered**: A hierarchical prior across days could be used, but with only 7 categories that recur ~26 times each, partial pooling is not critical.

### Product release effect: Normal(mu=0, sigma=1.0)

- **Centering at zero**: We do not assume the direction of the effect a priori (though we expect it to be positive). The data will determine this.
- **Scale**: sigma=1 on the log scale allows up to about exp(2) = 7.4x multiplier at 2 SD. This is generous but not absurd -- a major release could plausibly triple or quadruple the ticket rate.
- **No positivity constraint**: We deliberately avoid a positivity constraint because (a) some releases might reduce tickets (bug fixes), and (b) letting the prior be symmetric and weakly informative keeps us honest. The posterior will strongly concentrate on positive values if the data supports it.

### Overdispersion alpha: Gamma(alpha=2, beta=0.5)

- **Mean**: 2/0.5 = 4, implying moderate overdispersion as a prior expectation. Var = mu + mu^2/4 at the prior mean.
- **Avoiding near-zero**: The Gamma(2, 0.5) shape avoids placing mass near zero, which would imply extreme overdispersion (essentially a geometric-like distribution). This follows the skill's guidance to use Gamma priors that avoid zero for scale parameters.
- **Upper tail**: The prior allows alpha up to ~15--20 with non-negligible probability, so if the data is close to Poisson, the model can learn that.
- **Alternative considered**: An Exponential or HalfNormal prior. Exponential places too much mass near zero. HalfNormal could work but the Gamma shape is more appropriate for this parameter.

## Diagnostics Interpretation Guide

When running the code, here is what to check at each stage:

### Prior predictive checks

- The prior predictive distribution for ticket counts should span roughly 0 to 50--100 tickets/day.
- If a large fraction of prior predictive draws exceed 500 tickets/day, the priors are too diffuse -- tighten them.
- If the prior predictive distribution is very narrow (e.g., 5--10 only), the priors are too informative.

### Convergence diagnostics

- **R-hat <= 1.01** for all parameters: Chains have converged.
- **ESS bulk and tail >= 400** (100 per chain * 4 chains): Sufficient effective samples.
- **Zero divergences**: No posterior geometry problems. If divergences appear, increase `target_accept` to 0.95--0.99.
- **Rank plots**: Should look uniform (flat histograms) for all parameters. Any spikes or gaps indicate mixing problems.
- **Energy plots**: Marginal energy and energy transition distributions should overlap substantially.

### Posterior predictive checks

- **Overall PPC**: The NegBin model's posterior predictive distribution should closely match the observed histogram, including the right tail (high-count days). The Poisson model is expected to fail here -- its predicted distribution will be too narrow.
- **Variance PPC**: This is the critical targeted check. The observed variance should fall within the posterior predictive distribution of variances for the NegBin model. For the Poisson model, the observed variance is expected to be far in the right tail (or entirely outside) the predicted variance distribution.

### LOO-CV and Pareto k

- The NegBin model should have a substantially higher (less negative) ELPD than the Poisson model.
- **Poisson model**: Expect many observations with high Pareto k (> 0.7), especially the high-count days. This signals that the Poisson model assigns very low probability to these observations -- they are influential and poorly fit.
- **NegBin model**: Expect most or all Pareto k values below 0.5 if the model is well-specified. Any observations with k > 0.7 warrant investigation.

### Residual analysis

- Residuals vs. fitted values: No systematic fan shape or trend should be present in the NegBin model.
- Residuals vs. day of week: No systematic pattern should remain after accounting for day-of-week effects.
- Residuals vs. product release: No systematic pattern should remain.

## Model Comparison Interpretation

The LOO-CV comparison table reports:

- **elpd_loo**: Expected log pointwise predictive density. Higher (less negative) is better.
- **elpd_diff**: Difference from the best model. Should be positive for the NegBin if it is preferred.
- **dse**: Standard error of the difference.
- **Interpretation**: If elpd_diff > 4 * dse, there is strong evidence for the better model. For overdispersed data, the NegBin advantage should be decisive.

Note on likelihood families: Strictly, LOO-CV comparison is most interpretable when models share the same likelihood. Here we compare Poisson vs. NegBin, which have different likelihoods. However, LOO-CV is still valid for comparing predictive accuracy -- it answers "which model predicts unseen observations better?" regardless of the likelihood family.

## Expected Results Summary

Based on the synthetic data generation process (and what we would expect from real data matching this description):

1. **Product release effect**: The posterior for `beta_release` should concentrate around 0.7--0.9 on the log scale, corresponding to a multiplicative effect of roughly 2.0--2.5x more tickets on release days. The 94% HDI should exclude zero, providing strong evidence of a positive effect.

2. **Day-of-week effects**: Moderate variation, with weekdays (especially Thu/Fri) showing slightly higher rates and weekends (especially Sunday) showing lower rates. Effects on the log scale should be small (within +/- 0.3).

3. **Overdispersion parameter alpha**: Should be estimated around 2--5, confirming that the data is substantially overdispersed (far from the Poisson limit).

4. **Baseline rate**: Around 6--9 tickets/day on a typical non-release weekday.

5. **Model comparison**: The Negative Binomial should decisively outperform the Poisson, with a large ELPD difference (likely > 4x the standard error of the difference).

## Limitations

1. **Temporal structure ignored**: The model treats days as exchangeable (given predictors). If ticket counts are autocorrelated (e.g., a bug discovered on Monday generates tickets through Wednesday), a time-series component or autoregressive term would be needed. This would be a natural next iteration.

2. **No interaction effects**: We do not model interactions between day-of-week and product release (e.g., releases on Fridays might generate different ticket patterns than releases on Mondays). This could be added if there is domain motivation.

3. **Release effect assumed instantaneous**: The model assumes the release effect is confined to the release day. In reality, tickets might remain elevated for 2--3 days after a release. A lagged effect or a decaying release indicator could capture this.

4. **No seasonality**: Over 180 days (~6 months), there could be broader trends (growing user base, holiday effects). These are not modeled.

5. **Synthetic data**: The analysis uses synthetic data generated from a known NegBin process. Real data would require validating that the NegBin is appropriate (not just assumed) and might reveal additional structure.

6. **Prior sensitivity**: For a real analysis with consequential decisions, it would be worth checking that key conclusions are robust to alternative prior choices (e.g., wider or narrower sigma on beta_release, different alpha prior).

## Technical Notes

- **Sampler**: The code uses `nuts_sampler="nutpie"` for faster sampling via the nutpie package. If nutpie is not installed, remove this argument to use PyMC's default NUTS sampler.
- **HDI**: All credible intervals use 94% HDI (Highest Density Interval), following the convention that 94% avoids the false precision of 95% and acknowledges the arbitrary nature of the threshold.
- **ZeroSumNormal**: This parameterization for day-of-week effects ensures that the effects sum to zero, providing clean identifiability with the intercept. It is preferred over reference encoding when no single category is a natural baseline.
- **Log link**: The log link ensures the predicted rate `mu` is always positive, which is a physical requirement for count data.
- **Inference data saved**: Both models' InferenceData objects are saved as NetCDF files for potential later analysis, comparison, or reporting.
