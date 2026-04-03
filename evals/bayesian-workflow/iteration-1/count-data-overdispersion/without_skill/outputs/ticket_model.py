"""
Bayesian analysis of customer support ticket counts with overdispersion.

Models daily support ticket counts for a SaaS company using a Negative Binomial
likelihood to handle overdispersion (variance > mean). Predictors include
day_of_week and whether a product release occurred that day.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Generate synthetic data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n_days = 180

day_of_week = np.tile(np.arange(7), n_days // 7 + 1)[:n_days]  # 0=Mon ... 6=Sun
is_release = rng.binomial(1, 0.08, size=n_days)  # ~8% of days have a release

# True data-generating process (Negative Binomial with known parameters)
# log(mu) = intercept + dow_effects + release_effect
true_intercept = 2.1  # baseline ~ exp(2.1) ≈ 8 tickets
true_dow_effects = np.array([0.0, 0.05, 0.1, 0.1, 0.05, -0.3, -0.5])  # lower on weekends
true_release_effect = 1.0  # releases roughly triple the ticket count

log_mu = true_intercept + true_dow_effects[day_of_week] + true_release_effect * is_release
mu = np.exp(log_mu)

# Negative Binomial parameterized by mu and alpha (smaller alpha = more overdispersion)
true_alpha = 3.0
tickets = rng.negative_binomial(
    n=true_alpha,
    p=true_alpha / (true_alpha + mu),
    size=n_days,
)

df = pd.DataFrame({
    "day_index": np.arange(n_days),
    "day_of_week": day_of_week,
    "is_release": is_release,
    "tickets": tickets,
})

print("=== Data summary ===")
print(f"Mean tickets/day:     {df['tickets'].mean():.1f}")
print(f"Variance tickets/day: {df['tickets'].var():.1f}")
print(f"Variance / Mean:      {df['tickets'].var() / df['tickets'].mean():.2f}")
print(f"Max tickets in a day: {df['tickets'].max()}")
print(f"Days with 40+ tickets:{(df['tickets'] >= 40).sum()}")
print()

# ---------------------------------------------------------------------------
# 2. Exploratory plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df["tickets"], bins=30, edgecolor="white")
axes[0].set_xlabel("Tickets per day")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution of daily ticket counts")

dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
df.groupby("day_of_week")["tickets"].mean().plot.bar(ax=axes[1])
axes[1].set_xticklabels(dow_labels, rotation=45)
axes[1].set_ylabel("Mean tickets")
axes[1].set_title("Mean tickets by day of week")

df.groupby("is_release")["tickets"].mean().plot.bar(ax=axes[2])
axes[2].set_xticklabels(["No release", "Release"], rotation=0)
axes[2].set_ylabel("Mean tickets")
axes[2].set_title("Mean tickets: release vs. no release")

fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
    "without_skill/outputs/eda_plots.png",
    dpi=150,
)
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Model specification
# ---------------------------------------------------------------------------
# We compare two models:
#   Model 1: Poisson (assumes variance == mean)
#   Model 2: Negative Binomial (allows variance > mean)

# --- Shared design matrix ---
# Encode day_of_week as dummy variables (reference = Monday, index 0)
dow_dummies = pd.get_dummies(df["day_of_week"], prefix="dow", drop_first=True).values
release = df["is_release"].values
y = df["tickets"].values

# --- Model 1: Poisson ---
with pm.Model() as poisson_model:
    # Priors
    intercept = pm.Normal("intercept", mu=2.0, sigma=1.0)
    beta_dow = pm.Normal("beta_dow", mu=0.0, sigma=0.5, shape=6)
    beta_release = pm.Normal("beta_release", mu=0.0, sigma=1.0)

    # Linear predictor
    log_mu_p = intercept + pm.math.dot(dow_dummies, beta_dow) + beta_release * release

    # Likelihood
    obs = pm.Poisson("tickets", mu=pm.math.exp(log_mu_p), observed=y)

# --- Model 2: Negative Binomial ---
with pm.Model() as negbin_model:
    # Priors
    intercept = pm.Normal("intercept", mu=2.0, sigma=1.0)
    beta_dow = pm.Normal("beta_dow", mu=0.0, sigma=0.5, shape=6)
    beta_release = pm.Normal("beta_release", mu=0.0, sigma=1.0)

    # Overdispersion parameter: alpha > 0
    # Smaller alpha = more overdispersion; as alpha -> inf, NegBin -> Poisson
    # HalfNormal(10) is weakly informative — allows substantial overdispersion
    # but keeps the prior mass away from extreme values
    alpha = pm.HalfNormal("alpha", sigma=10.0)

    # Linear predictor
    log_mu_nb = intercept + pm.math.dot(dow_dummies, beta_dow) + beta_release * release
    mu_nb = pm.math.exp(log_mu_nb)

    # Likelihood: NegativeBinomial parameterized by (mu, alpha)
    obs = pm.NegativeBinomial("tickets", mu=mu_nb, alpha=alpha, observed=y)

# ---------------------------------------------------------------------------
# 4. Prior predictive checks
# ---------------------------------------------------------------------------
with negbin_model:
    prior_pred = pm.sample_prior_predictive(samples=500, random_seed=42)

fig, ax = plt.subplots(figsize=(8, 4))
prior_samples = prior_pred.prior_predictive["tickets"].values.flatten()
# Clip extreme values for visualization
clipped = prior_samples[prior_samples < 200]
ax.hist(clipped, bins=80, density=True, alpha=0.6, label="Prior predictive")
ax.axvline(df["tickets"].mean(), color="red", ls="--", label="Observed mean")
ax.set_xlabel("Tickets per day")
ax.set_ylabel("Density")
ax.set_title("Prior predictive distribution (Negative Binomial model)")
ax.legend()
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
    "without_skill/outputs/prior_predictive.png",
    dpi=150,
)
plt.close(fig)

print("Prior predictive check: median predicted tickets =", np.median(prior_samples))
print("Prior predictive check: 95% interval =", np.percentile(prior_samples, [2.5, 97.5]))
print()

# ---------------------------------------------------------------------------
# 5. Fit both models
# ---------------------------------------------------------------------------
with poisson_model:
    trace_poisson = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        random_seed=42,
        target_accept=0.9,
    )

with negbin_model:
    trace_negbin = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        random_seed=42,
        target_accept=0.9,
    )

# ---------------------------------------------------------------------------
# 6. Diagnostics
# ---------------------------------------------------------------------------
print("=== Negative Binomial model diagnostics ===")
summary_nb = az.summary(trace_negbin, var_names=["intercept", "beta_dow", "beta_release", "alpha"])
print(summary_nb)
print()

# Check convergence: R-hat should be < 1.01, ESS should be > 400
rhat_ok = (summary_nb["r_hat"] < 1.01).all()
ess_ok = (summary_nb["ess_bulk"] > 400).all() and (summary_nb["ess_tail"] > 400).all()
print(f"All R-hat < 1.01: {rhat_ok}")
print(f"All ESS > 400:    {ess_ok}")
print()

# Trace plots
axes_trace = az.plot_trace(
    trace_negbin,
    var_names=["intercept", "beta_dow", "beta_release", "alpha"],
    figsize=(12, 10),
)
plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
    "without_skill/outputs/trace_plots.png",
    dpi=150,
)
plt.close()

# ---------------------------------------------------------------------------
# 7. Posterior predictive checks
# ---------------------------------------------------------------------------
with negbin_model:
    ppc_negbin = pm.sample_posterior_predictive(trace_negbin, random_seed=42)

with poisson_model:
    ppc_poisson = pm.sample_posterior_predictive(trace_poisson, random_seed=42)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Poisson PPC
az.plot_ppc(
    az.InferenceData(
        posterior_predictive=ppc_poisson.posterior_predictive,
        observed_data=ppc_poisson.observed_data,
    ),
    ax=axes[0],
    num_pp_samples=100,
)
axes[0].set_title("Posterior Predictive Check — Poisson")
axes[0].set_xlim(0, 60)

# Negative Binomial PPC
az.plot_ppc(
    az.InferenceData(
        posterior_predictive=ppc_negbin.posterior_predictive,
        observed_data=ppc_negbin.observed_data,
    ),
    ax=axes[1],
    num_pp_samples=100,
)
axes[1].set_title("Posterior Predictive Check — Negative Binomial")
axes[1].set_xlim(0, 60)

fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
    "without_skill/outputs/ppc_comparison.png",
    dpi=150,
)
plt.close(fig)

# ---------------------------------------------------------------------------
# 8. Model comparison via LOO-CV
# ---------------------------------------------------------------------------
poisson_loo = az.loo(
    az.InferenceData(
        posterior=trace_poisson.posterior,
        posterior_predictive=ppc_poisson.posterior_predictive,
        observed_data=ppc_poisson.observed_data,
    ),
)
negbin_loo = az.loo(
    az.InferenceData(
        posterior=trace_negbin.posterior,
        posterior_predictive=ppc_negbin.posterior_predictive,
        observed_data=ppc_negbin.observed_data,
    ),
)

print("=== Model comparison (LOO-CV) ===")
comparison = az.compare(
    {"Poisson": trace_poisson, "NegBin": trace_negbin},
)
print(comparison)
print()

# ---------------------------------------------------------------------------
# 9. Interpret results
# ---------------------------------------------------------------------------
print("=== Posterior summaries (Negative Binomial) ===")
print(az.summary(trace_negbin, var_names=["intercept", "beta_dow", "beta_release", "alpha"]))
print()

# Compute multiplicative effects on the response scale
post = trace_negbin.posterior
release_effect_samples = np.exp(post["beta_release"].values.flatten())
print(f"Release multiplicative effect on tickets:")
print(f"  Median: {np.median(release_effect_samples):.2f}x")
print(f"  94% HDI: {az.hdi(release_effect_samples, hdi_prob=0.94)}")
print()

# Day-of-week effects relative to Monday
dow_effects = np.exp(post["beta_dow"].values.reshape(-1, 6))
dow_labels_short = ["Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print("Day-of-week multiplicative effects (relative to Monday):")
for i, label in enumerate(dow_labels_short):
    median_eff = np.median(dow_effects[:, i])
    hdi = az.hdi(dow_effects[:, i], hdi_prob=0.94)
    print(f"  {label}: {median_eff:.2f}x  (94% HDI: [{hdi[0]:.2f}, {hdi[1]:.2f}])")
print()

# Overdispersion parameter
alpha_samples = post["alpha"].values.flatten()
print(f"Overdispersion (alpha):")
print(f"  Median: {np.median(alpha_samples):.2f}")
print(f"  94% HDI: {az.hdi(alpha_samples, hdi_prob=0.94)}")
print(f"  (Smaller alpha = more overdispersion; alpha -> inf means Poisson)")
print()

# ---------------------------------------------------------------------------
# 10. Forest plot of coefficients
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
az.plot_forest(
    trace_negbin,
    var_names=["beta_release", "beta_dow"],
    combined=True,
    hdi_prob=0.94,
    ax=ax,
)
ax.axvline(0, color="grey", ls="--", alpha=0.5)
ax.set_title("94% HDI for regression coefficients (log scale)")
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
    "without_skill/outputs/forest_plot.png",
    dpi=150,
)
plt.close(fig)

print("All outputs saved. Analysis complete.")
