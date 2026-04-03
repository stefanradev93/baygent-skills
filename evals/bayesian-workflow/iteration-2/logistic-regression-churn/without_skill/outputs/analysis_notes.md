# Bayesian Logistic Regression for Customer Churn: Analysis Notes

## 1. Approach Overview

This analysis uses a Bayesian logistic regression to model customer churn as a function of four predictors: age, tenure (months), monthly spend, and recent support tickets. The Bayesian approach was chosen over frequentist logistic regression for three key reasons:

1. **Uncertainty quantification**: Instead of point estimates of churn probability, we get a full posterior distribution of P(churn) for each customer. This lets us distinguish between "we're 80% confident this customer will churn" and "we have no idea but the mean is 50%."

2. **Regularization through priors**: Weakly informative priors act as gentle regularization, preventing extreme coefficient estimates that can arise from separation or multicollinearity -- common issues in logistic regression.

3. **Coherent probabilistic framework**: All inferences (coefficients, predictions, model comparisons) flow from a single probabilistic model. No need to bolt on separate confidence intervals, bootstrap procedures, or ad-hoc regularization.

The model is fit using PyMC's NUTS sampler (No-U-Turn Sampler), a state-of-the-art Hamiltonian Monte Carlo method that efficiently explores the posterior distribution.

## 2. Data Generation

Since no real data was provided, synthetic data is generated with realistic properties:

- **5000 observations**, split 80/20 into train/test sets
- **Age**: Normal(45, 12), clipped to [18, 85]
- **Tenure**: Exponential(scale=24), clipped to [1, 120] -- right-skewed, as expected
- **Monthly spend**: LogNormal(4.0, 0.5), clipped to [10, 500] -- right-skewed
- **Support tickets**: Poisson(1.5) -- count data, most customers have 0-3 tickets

The true data-generating process uses these relationships (on standardized scale):
- Intercept = -0.8 (baseline churn ~31% at mean feature values)
- Age: -0.15 (older customers slightly less likely to churn)
- Tenure: -0.6 (longer tenure strongly reduces churn -- makes business sense)
- Monthly spend: -0.2 (higher spenders slightly less likely to churn)
- Support tickets: +0.7 (more tickets strongly increases churn -- frustrated customers leave)

## 3. Feature Preprocessing

All features are **standardized** (zero mean, unit variance) before modeling. This is essential for Bayesian logistic regression because:

- It makes priors scale-agnostic: a Normal(0, 1) prior on a coefficient means "a 1-SD change in the feature shifts the log-odds by at most ~2-3" regardless of whether the feature is age (range ~60) or tickets (range ~8).
- It dramatically improves NUTS sampler efficiency by creating a more isotropic posterior geometry.
- It makes coefficients directly comparable in magnitude.

The scaler is fit on the training set only and applied to the test set to prevent data leakage.

## 4. Prior Choices

### Intercept: Normal(0, 1.5)

On the logit scale, 0 corresponds to a 50% churn rate. A standard deviation of 1.5 means:
- At +/- 1 SD: logit in [-1.5, 1.5] -> P(churn) in [18%, 82%]
- At +/- 2 SD: logit in [-3.0, 3.0] -> P(churn) in [5%, 95%]

This is diffuse enough to cover any plausible baseline churn rate while still being weakly informative (it downweights extreme baseline rates like 0.1% or 99.9%).

### Coefficients (betas): Normal(0, 1)

On standardized features, each beta represents the change in log-odds per 1-SD change in the feature. A Normal(0, 1) prior means:
- We center at zero (no prior preference for positive or negative effects)
- 95% of prior mass is within [-2, 2] on the log-odds scale
- A beta of 2 corresponds to an odds ratio of e^2 = 7.4 per 1-SD change, which is already a very large effect

This follows the recommendations of Gelman et al. (2008) for weakly informative priors in logistic regression. With n=4000 training observations, the likelihood will dominate the prior for all parameters, so the exact prior choice matters very little. The priors serve primarily as regularization against numerically extreme estimates.

### Why not flat/uninformative priors?

Flat priors on logistic regression coefficients are actually *informative* in a problematic way: they imply that effects of size 100 on the log-odds scale are just as plausible as effects of size 0.1. This creates a U-shaped prior predictive distribution on P(churn), concentrating mass at 0 and 1. Our weakly informative priors avoid this pathology.

## 5. Model Structure

The model is a standard Bayesian logistic regression:

```
y_i ~ Bernoulli(p_i)
logit(p_i) = alpha + X_i * beta
alpha ~ Normal(0, 1.5)
beta_j ~ Normal(0, 1)  for j = 1, ..., 4
```

No interactions or nonlinear terms are included. This is a deliberate choice for a first model: start simple, check diagnostics and fit, then add complexity only if warranted. A logistic regression is a reasonable first-pass model for binary classification with continuous predictors.

## 6. MCMC Sampling Configuration

- **Sampler**: NUTS (No-U-Turn Sampler)
- **Chains**: 4 (standard for convergence diagnostics)
- **Tuning steps**: 1000 per chain (adaptation period, discarded)
- **Posterior draws**: 2000 per chain (8000 total)
- **Target acceptance**: 0.9 (slightly higher than default 0.8 for better exploration)

The total of 8000 posterior draws is more than sufficient for stable summary statistics. With 4 independent chains, we can reliably compute R-hat and rank-based diagnostics.

## 7. Diagnostics to Check

### R-hat (Potential Scale Reduction Factor)

- **What it measures**: Agreement between chains. If all chains are sampling from the same distribution, R-hat should be close to 1.
- **Threshold**: R-hat < 1.01 for all parameters indicates convergence.
- **Expected outcome**: For this simple model with weakly informative priors and abundant data, R-hat should be very close to 1.0 for all parameters.

### Effective Sample Size (ESS)

- **Bulk ESS**: Measures the effective number of independent samples for estimating the central tendency (mean, median). Should be > 400.
- **Tail ESS**: Measures effective samples for estimating tail quantiles (important for HDIs). Should be > 400.
- **Expected outcome**: With 8000 total draws and a well-behaved posterior, ESS should be in the thousands.

### Divergences

- **What they indicate**: Geometric pathologies in the posterior that prevent NUTS from exploring correctly. Even a single divergence can indicate biased estimates.
- **Expected outcome**: Zero divergences for this simple model. If divergences occur, options include increasing `target_accept`, reparameterizing the model, or investigating the posterior geometry.

### Trace Plots

- Visual check that chains are "mixing well" -- they should look like stationary white noise ("hairy caterpillars"), not show trends, multimodality, or sticky regions.

### Rank Plots

- A more rigorous alternative to trace plots. If chains are mixing well, the rank histogram for each chain should be approximately uniform.

## 8. Results Interpretation

### Coefficients

The posterior means of the coefficients should be close to the true data-generating values (-0.15, -0.6, -0.2, +0.7 on standardized scale). The 94% HDI should contain the true values.

Key interpretive quantities:
- **Posterior mean/median**: Best estimate of the effect
- **94% HDI**: Range containing 94% of the posterior mass. If this excludes zero, we have strong evidence the feature matters. The 94% interval is preferred over 95% because it has nicer numerical properties for MCMC estimation.
- **P(beta > 0)**: Posterior probability that the effect is positive. Values near 0 or 1 indicate strong directional evidence.
- **Odds ratios**: exp(beta) gives the multiplicative change in odds of churn per 1-SD change in the feature. More intuitive than log-odds for stakeholder communication.

### Predictions

For each test observation, we get a full posterior distribution of P(churn), not just a point estimate. This means:
- The **mean** of the posterior predictive gives our best guess at P(churn)
- The **standard deviation** quantifies how uncertain we are about this probability
- The **94% HDI** gives a credible interval for P(churn)

High-uncertainty predictions (wide HDI) are cases where we should be cautious about acting on the prediction. Low-uncertainty predictions with high P(churn) are the customers most worth intervening on.

## 9. Model Comparison

The script compares the full model against an intercept-only (null) model using LOO-CV (Leave-One-Out Cross-Validation via Pareto-smoothed importance sampling). This is the Bayesian analog of comparing against a null model in frequentist statistics.

The full model should show a substantially better ELPD (Expected Log Predictive Density) than the null model, confirming that the features are informative for predicting churn. The ELPD difference and its standard error give a principled comparison that accounts for model complexity.

## 10. Limitations and Extensions

### Current limitations
- **Linear effects only**: The model assumes each feature has a linear effect on the log-odds. Nonlinear effects (e.g., U-shaped relationship between age and churn) would be missed.
- **No interactions**: The model doesn't capture interactions (e.g., "high tickets + low tenure = especially high churn").
- **No hierarchical structure**: If customers belong to segments or regions, a hierarchical model could share information across groups.

### Possible extensions
1. **Spline terms**: Replace linear effects with spline basis functions for continuous features to capture nonlinear relationships.
2. **Interaction terms**: Add product terms for plausible interactions (e.g., tenure * support_tickets).
3. **Hierarchical model**: If customer segments exist, use varying intercepts/slopes by segment.
4. **Horseshoe prior**: If many features are added, use a horseshoe prior for automatic variable selection.
5. **Time-varying effects**: If data spans multiple periods, model time-varying churn dynamics.
6. **Posterior predictive calibration**: Check whether predicted probabilities match observed churn rates across deciles (calibration plot).

## 11. Output Files

- `churn_model.py`: Complete, runnable Python script
- `analysis_notes.md`: This file
- `trace_plots.png`: MCMC trace plots for visual convergence assessment
- `posterior_distributions.png`: Posterior distributions with 94% HDIs
- `forest_plot.png`: Coefficient comparison forest plot
- `rank_plots.png`: Rank plots for convergence assessment
- `prior_predictive.png`: Prior predictive check
- `predictions.png`: Predicted probability distributions and uncertainty visualization
- `uncertainty_intervals.png`: Individual prediction intervals
- `model_comparison.png`: LOO-CV model comparison
- `churn_model_idata.nc`: Saved inference data (NetCDF format) for downstream use
