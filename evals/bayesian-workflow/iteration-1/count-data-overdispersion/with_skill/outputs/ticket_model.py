"""
Bayesian analysis of customer support ticket counts per day.

Problem: Count data (tickets/day) with overdispersion (variance > mean)
         and potential spikes driven by day_of_week and product_release.

Workflow follows the Bayesian modeling skill:
1. Formulate generative story
2. Specify priors with justification
3. Implement in PyMC (with coords and dims)
4. Prior predictive checks
5. Inference
6. Convergence diagnostics
7. Model criticism (PPC, LOO-CV, calibration, residuals)
8. Model comparison (Poisson vs. NegativeBinomial)
9. Report results
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Seed for reproducibility
rng = np.random.default_rng(42)

# ============================================================
# 0. Generate synthetic data
# ============================================================
n_days = 180

# Day of week: 0=Monday ... 6=Sunday
day_of_week = np.tile(np.arange(7), n_days // 7 + 1)[:n_days]

# Product releases: ~10% of days, slightly more likely on Tuesdays/Wednesdays
release_prob = np.where(np.isin(day_of_week, [1, 2]), 0.15, 0.07)
product_release = rng.binomial(1, release_prob, size=n_days)

# True generative process: NegativeBinomial with log-link
# log(mu) = intercept + beta_dow[day_of_week] + beta_release * product_release
true_intercept = 2.0  # exp(2.0) ~ 7.4 baseline tickets
true_beta_dow = np.array([-0.1, 0.05, 0.0, 0.1, 0.15, -0.2, -0.3])  # Mon-Sun
# Zero-sum center the DOW effects for identifiability
true_beta_dow = true_beta_dow - true_beta_dow.mean()
true_beta_release = 0.8  # exp(0.8) ~ 2.2x multiplier on release days
true_alpha = 3.0  # NegBin overdispersion (lower = more overdispersion)

log_mu = true_intercept + true_beta_dow[day_of_week] + true_beta_release * product_release
mu = np.exp(log_mu)

# NegativeBinomial parameterized as (mu, alpha):
# Var = mu + mu^2/alpha. When alpha is small, variance >> mean.
tickets = rng.negative_binomial(
    n=true_alpha,
    p=true_alpha / (true_alpha + mu),
    size=n_days,
)

# Assemble into a DataFrame
df = pd.DataFrame({
    "day_index": np.arange(n_days),
    "day_of_week": day_of_week,
    "product_release": product_release,
    "tickets": tickets,
})

# Quick exploratory summary
print("=== Data Summary ===")
print(f"Mean tickets/day: {df['tickets'].mean():.2f}")
print(f"Variance tickets/day: {df['tickets'].var():.2f}")
print(f"Variance/Mean ratio: {df['tickets'].var() / df['tickets'].mean():.2f}")
print(f"Max tickets/day: {df['tickets'].max()}")
print(f"Days with product release: {df['product_release'].sum()}")
print()
print("Tickets by day of week:")
print(df.groupby("day_of_week")["tickets"].agg(["mean", "std", "count"]))
print()
print("Tickets by product release:")
print(df.groupby("product_release")["tickets"].agg(["mean", "std", "count"]))

# ============================================================
# 1. Formulate the generative story
# ============================================================
# Each day, the customer support team receives a random number of tickets.
# The baseline rate varies by day of week (weekdays may differ from weekends)
# and spikes when a product release occurs. The count data shows
# overdispersion (variance >> mean), so a Poisson likelihood is
# insufficient. We model this with a Negative Binomial likelihood,
# which adds an overdispersion parameter.
#
# Generative story:
#   log(mu_i) = intercept + beta_dow[dow_i] + beta_release * release_i
#   tickets_i ~ NegativeBinomial(mu=mu_i, alpha=alpha)
#
# where alpha controls overdispersion: Var = mu + mu^2 / alpha.
# As alpha -> inf, NegBin -> Poisson. Small alpha = heavy overdispersion.

# ============================================================
# 2. Define coordinates and model structure
# ============================================================
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

coords = {
    "day_of_week": dow_names,
    "obs_id": np.arange(n_days),
}

# ============================================================
# 3a. Model 1: Poisson (baseline, ignoring overdispersion)
# ============================================================
with pm.Model(coords=coords) as poisson_model:
    # --- Data containers ---
    dow_idx = pm.Data("dow_idx", day_of_week, dims="obs_id")
    release = pm.Data("release", product_release.astype(float), dims="obs_id")
    observed_tickets = pm.Data("observed_tickets", tickets, dims="obs_id")

    # --- Priors ---
    # Intercept: on log scale. exp(2) ~ 7.4, exp(3) ~ 20.
    # Normal(2, 1) puts 95% mass on exp([0, 4]) = [1, 55] tickets/day.
    # This comfortably covers the observed range of ~5-40.
    intercept = pm.Normal("intercept", mu=2.0, sigma=1.0)

    # Day-of-week effects: ZeroSumNormal ensures identifiability
    # (effects sum to zero, no confounding with intercept).
    # sigma=0.5 on log scale means day-of-week can shift rates by
    # about exp(0.5) ~ 1.65x up or down -- reasonable for weekly patterns.
    beta_dow = pm.ZeroSumNormal("beta_dow", sigma=0.5, dims="day_of_week")

    # Product release effect: expect positive effect (more tickets after release).
    # Normal(0, 1) on log scale allows up to ~exp(2)=7.4x multiplier
    # at 2 SD, which is generous but not absurd for a major release.
    beta_release = pm.Normal("beta_release", mu=0.0, sigma=1.0)

    # --- Linear predictor (log link) ---
    log_mu = intercept + beta_dow[dow_idx] + beta_release * release
    mu = pm.math.exp(log_mu)

    # --- Likelihood ---
    pm.Poisson("tickets", mu=mu, observed=observed_tickets, dims="obs_id")

    # --- Prior predictive check ---
    prior_pred_poisson = pm.sample_prior_predictive(random_seed=42)

# Visualize prior predictive for Poisson model
fig, ax = plt.subplots(figsize=(10, 5))
az.plot_ppc(prior_pred_poisson, group="prior", num_pp_samples=100, ax=ax)
ax.set_title("Poisson Model: Prior Predictive Check")
ax.set_xlabel("Tickets per day")
plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
    "with_skill/outputs/prior_predictive_poisson.png",
    dpi=150,
)
plt.close()

# ============================================================
# 3b. Model 2: Negative Binomial (accounts for overdispersion)
# ============================================================
with pm.Model(coords=coords) as negbin_model:
    # --- Data containers ---
    dow_idx = pm.Data("dow_idx", day_of_week, dims="obs_id")
    release = pm.Data("release", product_release.astype(float), dims="obs_id")
    observed_tickets = pm.Data("observed_tickets", tickets, dims="obs_id")

    # --- Priors ---
    # Same regression priors as Poisson model for fair comparison.

    # Intercept: log-scale baseline rate.
    # Normal(2, 1) => 95% of prior mass on [1, 55] tickets/day baseline.
    intercept = pm.Normal("intercept", mu=2.0, sigma=1.0)

    # Day-of-week effects: ZeroSumNormal for identifiability.
    # sigma=0.5 allows moderate day-to-day variation on log scale.
    beta_dow = pm.ZeroSumNormal("beta_dow", sigma=0.5, dims="day_of_week")

    # Product release effect: weakly informative, centered at zero.
    # Allows data to determine direction and magnitude.
    beta_release = pm.Normal("beta_release", mu=0.0, sigma=1.0)

    # Overdispersion parameter alpha:
    # Var = mu + mu^2 / alpha. Smaller alpha = more overdispersion.
    # Gamma(2, 0.5) has mean=4, avoids near-zero values (which would
    # imply extreme overdispersion), while allowing a wide range.
    # At alpha > 20, NegBin is essentially Poisson, so this prior
    # comfortably covers both overdispersed and Poisson-like regimes.
    alpha = pm.Gamma("alpha", alpha=2, beta=0.5)

    # --- Linear predictor (log link) ---
    log_mu = intercept + beta_dow[dow_idx] + beta_release * release
    mu = pm.math.exp(log_mu)

    # --- Likelihood ---
    # PyMC's NegativeBinomial is parameterized as (mu, alpha)
    # where Var = mu + mu^2 / alpha.
    pm.NegativeBinomial(
        "tickets", mu=mu, alpha=alpha, observed=observed_tickets, dims="obs_id"
    )

    # --- Prior predictive check ---
    prior_pred_negbin = pm.sample_prior_predictive(random_seed=42)

# Visualize prior predictive for NegBin model
fig, ax = plt.subplots(figsize=(10, 5))
az.plot_ppc(prior_pred_negbin, group="prior", num_pp_samples=100, ax=ax)
ax.set_title("Negative Binomial Model: Prior Predictive Check")
ax.set_xlabel("Tickets per day")
plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
    "with_skill/outputs/prior_predictive_negbin.png",
    dpi=150,
)
plt.close()

# ============================================================
# 4. Inference
# ============================================================
# --- Poisson model ---
with poisson_model:
    idata_poisson = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        nuts_sampler="nutpie",
        random_seed=42,
        idata_kwargs={"log_likelihood": True},
    )
    idata_poisson.extend(prior_pred_poisson)
    idata_poisson.extend(pm.sample_posterior_predictive(idata_poisson, random_seed=42))

# --- Negative Binomial model ---
with negbin_model:
    idata_negbin = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        nuts_sampler="nutpie",
        random_seed=42,
        idata_kwargs={"log_likelihood": True},
    )
    idata_negbin.extend(prior_pred_negbin)
    idata_negbin.extend(pm.sample_posterior_predictive(idata_negbin, random_seed=42))

# ============================================================
# 5. Convergence Diagnostics
# ============================================================
print("\n" + "=" * 60)
print("CONVERGENCE DIAGNOSTICS: POISSON MODEL")
print("=" * 60)

summary_poisson = az.summary(idata_poisson, round_to=3)
print(summary_poisson)

num_chains = 4

rhat_ok_p = (summary_poisson["r_hat"] <= 1.01).all()
ess_bulk_ok_p = (summary_poisson["ess_bulk"] >= 100 * num_chains).all()
ess_tail_ok_p = (summary_poisson["ess_tail"] >= 100 * num_chains).all()
n_div_p = idata_poisson.sample_stats["diverging"].sum().item()

print(f"\nR-hat OK (all <= 1.01): {rhat_ok_p}")
print(f"ESS bulk OK (all >= {100 * num_chains}): {ess_bulk_ok_p}")
print(f"ESS tail OK (all >= {100 * num_chains}): {ess_tail_ok_p}")
print(f"Divergences: {n_div_p}")

print("\n" + "=" * 60)
print("CONVERGENCE DIAGNOSTICS: NEGATIVE BINOMIAL MODEL")
print("=" * 60)

summary_negbin = az.summary(idata_negbin, round_to=3)
print(summary_negbin)

rhat_ok_nb = (summary_negbin["r_hat"] <= 1.01).all()
ess_bulk_ok_nb = (summary_negbin["ess_bulk"] >= 100 * num_chains).all()
ess_tail_ok_nb = (summary_negbin["ess_tail"] >= 100 * num_chains).all()
n_div_nb = idata_negbin.sample_stats["diverging"].sum().item()

print(f"\nR-hat OK (all <= 1.01): {rhat_ok_nb}")
print(f"ESS bulk OK (all >= {100 * num_chains}): {ess_bulk_ok_nb}")
print(f"ESS tail OK (all >= {100 * num_chains}): {ess_tail_ok_nb}")
print(f"Divergences: {n_div_nb}")

# Trace / rank plots
for name, idata in [("Poisson", idata_poisson), ("NegBin", idata_negbin)]:
    axes = az.plot_trace(idata, kind="rank_vlines")
    plt.suptitle(f"{name} Model: Rank Plots", y=1.02)
    plt.tight_layout()
    plt.savefig(
        f"/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
        f"bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
        f"with_skill/outputs/trace_rank_{name.lower()}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

# Energy diagnostic
for name, idata in [("Poisson", idata_poisson), ("NegBin", idata_negbin)]:
    az.plot_energy(idata)
    plt.title(f"{name} Model: Energy Diagnostic")
    plt.tight_layout()
    plt.savefig(
        f"/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
        f"bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
        f"with_skill/outputs/energy_{name.lower()}.png",
        dpi=150,
    )
    plt.close()

# ============================================================
# 6. Model Criticism
# ============================================================

# --- 6a. Posterior Predictive Checks ---
for name, idata in [("Poisson", idata_poisson), ("NegBin", idata_negbin)]:
    fig, ax = plt.subplots(figsize=(10, 5))
    az.plot_ppc(idata, num_pp_samples=100, ax=ax)
    ax.set_title(f"{name} Model: Posterior Predictive Check")
    ax.set_xlabel("Tickets per day")
    plt.tight_layout()
    plt.savefig(
        f"/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
        f"bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
        f"with_skill/outputs/ppc_{name.lower()}.png",
        dpi=150,
    )
    plt.close()

# --- 6b. Targeted PPC: variance and max ---
# The key features we need the model to capture: overall spread and tail behavior.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (name, idata) in enumerate(
    [("Poisson", idata_poisson), ("NegBin", idata_negbin)]
):
    pp_data = idata.posterior_predictive["tickets"]
    # Variance across observations for each posterior draw
    pp_var = pp_data.var(dim="obs_id")
    observed_var = df["tickets"].var()

    axes[idx].hist(
        pp_var.values.flatten(), bins=50, alpha=0.7, density=True, label="Posterior pred."
    )
    axes[idx].axvline(
        observed_var, color="red", linewidth=2, linestyle="--", label="Observed"
    )
    axes[idx].set_title(f"{name}: Posterior Predictive Variance")
    axes[idx].set_xlabel("Variance of tickets/day")
    axes[idx].legend()

plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
    "with_skill/outputs/ppc_variance_comparison.png",
    dpi=150,
)
plt.close()

# --- 6c. LOO-CV ---
print("\n" + "=" * 60)
print("LOO-CV DIAGNOSTICS")
print("=" * 60)

loo_poisson = az.loo(idata_poisson, pointwise=True)
loo_negbin = az.loo(idata_negbin, pointwise=True)

print("\nPoisson LOO:")
print(loo_poisson)
print(f"\nHigh Pareto k observations (>0.7): "
      f"{np.where(loo_poisson.pareto_k.values > 0.7)[0]}")

print("\nNegative Binomial LOO:")
print(loo_negbin)
print(f"\nHigh Pareto k observations (>0.7): "
      f"{np.where(loo_negbin.pareto_k.values > 0.7)[0]}")

# Pareto k diagnostic plots
for name, loo_result in [("Poisson", loo_poisson), ("NegBin", loo_negbin)]:
    az.plot_khat(loo_result)
    plt.title(f"{name} Model: Pareto k Diagnostic")
    plt.tight_layout()
    plt.savefig(
        f"/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
        f"bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
        f"with_skill/outputs/pareto_k_{name.lower()}.png",
        dpi=150,
    )
    plt.close()

# --- 6d. Residual Analysis (NegBin model) ---
pp_mean_nb = idata_negbin.posterior_predictive["tickets"].mean(dim=["chain", "draw"])
residuals_nb = df["tickets"].values - pp_mean_nb.values

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Residuals vs fitted
axes[0].scatter(pp_mean_nb, residuals_nb, alpha=0.5, s=20)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Fitted values (posterior mean)")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs. Fitted")

# Residuals vs day of week
axes[1].scatter(df["day_of_week"], residuals_nb, alpha=0.5, s=20)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("Day of week")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residuals vs. Day of Week")
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(dow_names, rotation=45)

# Residuals vs product release
axes[2].scatter(
    df["product_release"] + rng.normal(0, 0.05, n_days),  # jitter
    residuals_nb,
    alpha=0.5,
    s=20,
)
axes[2].axhline(0, color="red", linestyle="--")
axes[2].set_xlabel("Product Release")
axes[2].set_ylabel("Residuals")
axes[2].set_title("Residuals vs. Product Release")

plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
    "with_skill/outputs/residuals_negbin.png",
    dpi=150,
)
plt.close()

# ============================================================
# 7. Model Comparison
# ============================================================
print("\n" + "=" * 60)
print("MODEL COMPARISON: POISSON vs. NEGATIVE BINOMIAL")
print("=" * 60)

# NOTE: model comparison with LOO requires same likelihood family
# for strict interpretation. Here we compare predictive performance,
# which is valid even across different likelihoods.
comparison = az.compare(
    {"Poisson": idata_poisson, "NegBin": idata_negbin},
    ic="loo",
)
print(comparison)

# Comparison plot
az.plot_compare(comparison)
plt.title("Model Comparison: LOO-CV")
plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
    "with_skill/outputs/model_comparison.png",
    dpi=150,
)
plt.close()

# ============================================================
# 8. Report Results (Negative Binomial Model)
# ============================================================
print("\n" + "=" * 60)
print("RESULTS: NEGATIVE BINOMIAL MODEL")
print("=" * 60)

# Parameter summary with 94% HDI (the Bayesian workflow default)
summary_final = az.summary(idata_negbin, hdi_prob=0.94, round_to=3)
print("\nParameter Summary (94% HDI):")
print(summary_final)

# Forest plot
az.plot_forest(
    idata_negbin,
    var_names=["intercept", "beta_dow", "beta_release", "alpha"],
    combined=True,
    hdi_prob=0.94,
)
plt.title("Negative Binomial Model: Parameter Estimates (94% HDI)")
plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
    "with_skill/outputs/forest_plot_negbin.png",
    dpi=150,
)
plt.close()

# Posterior distributions of key effects
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Intercept -> baseline tickets/day
intercept_samples = idata_negbin.posterior["intercept"].values.flatten()
baseline_tickets = np.exp(intercept_samples)
axes[0].hist(baseline_tickets, bins=50, alpha=0.7, density=True)
axes[0].set_xlabel("Baseline tickets/day")
axes[0].set_ylabel("Density")
axes[0].set_title("Posterior: Baseline Daily Tickets\n(exp of intercept)")
hdi_baseline = az.hdi(baseline_tickets, hdi_prob=0.94)
axes[0].axvline(
    np.median(baseline_tickets), color="red", linestyle="-", label="Median"
)
axes[0].axvline(hdi_baseline[0], color="red", linestyle="--", label="94% HDI")
axes[0].axvline(hdi_baseline[1], color="red", linestyle="--")
axes[0].legend()

# Product release multiplier
beta_release_samples = idata_negbin.posterior["beta_release"].values.flatten()
release_multiplier = np.exp(beta_release_samples)
axes[1].hist(release_multiplier, bins=50, alpha=0.7, density=True)
axes[1].set_xlabel("Multiplicative effect on tickets")
axes[1].set_ylabel("Density")
axes[1].set_title("Posterior: Product Release Effect\n(exp of beta_release)")
hdi_multiplier = az.hdi(release_multiplier, hdi_prob=0.94)
axes[1].axvline(
    np.median(release_multiplier), color="red", linestyle="-", label="Median"
)
axes[1].axvline(hdi_multiplier[0], color="red", linestyle="--", label="94% HDI")
axes[1].axvline(hdi_multiplier[1], color="red", linestyle="--")
axes[1].legend()

# Day-of-week effects
beta_dow_samples = idata_negbin.posterior["beta_dow"].values
beta_dow_mean = beta_dow_samples.mean(axis=(0, 1))
beta_dow_hdi = az.hdi(idata_negbin, var_names=["beta_dow"], hdi_prob=0.94)
dow_hdi_vals = beta_dow_hdi["beta_dow"].values

axes[2].barh(dow_names, np.exp(beta_dow_mean), xerr=[
    np.exp(beta_dow_mean) - np.exp(dow_hdi_vals[:, 0]),
    np.exp(dow_hdi_vals[:, 1]) - np.exp(beta_dow_mean),
], capsize=4)
axes[2].axvline(1.0, color="red", linestyle="--", alpha=0.7)
axes[2].set_xlabel("Multiplicative effect on tickets")
axes[2].set_title("Posterior: Day-of-Week Effects\n(exp of beta_dow, 94% HDI)")

plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
    "with_skill/outputs/posterior_effects.png",
    dpi=150,
)
plt.close()

# Probability statements (the Bayesian advantage)
print("\n--- Key Probability Statements ---")
prob_release_positive = (beta_release_samples > 0).mean()
print(
    f"P(product release increases tickets) = {prob_release_positive:.3f}"
)
print(
    f"Median multiplier on release days: {np.median(release_multiplier):.2f}x"
)
print(
    f"94% HDI for release multiplier: [{hdi_multiplier[0]:.2f}, {hdi_multiplier[1]:.2f}]"
)
print(
    f"\nMedian baseline tickets/day: {np.median(baseline_tickets):.1f}"
)
print(
    f"94% HDI for baseline: [{hdi_baseline[0]:.1f}, {hdi_baseline[1]:.1f}]"
)

alpha_samples = idata_negbin.posterior["alpha"].values.flatten()
alpha_hdi = az.hdi(alpha_samples, hdi_prob=0.94)
print(
    f"\nOverdispersion parameter alpha: median={np.median(alpha_samples):.2f}, "
    f"94% HDI=[{alpha_hdi[0]:.2f}, {alpha_hdi[1]:.2f}]"
)
print(
    "  (Smaller alpha = more overdispersion. "
    "If alpha >> 20, data is essentially Poisson.)"
)

# Model graph
print("\n--- Model Graph ---")
pm.model_to_graphviz(negbin_model).render(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
    "with_skill/outputs/negbin_model_graph",
    format="png",
    cleanup=True,
)

# Save inference data for potential later use
idata_negbin.to_netcdf(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
    "with_skill/outputs/idata_negbin.nc"
)
idata_poisson.to_netcdf(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-1/count-data-overdispersion/"
    "with_skill/outputs/idata_poisson.nc"
)

print("\n=== Analysis complete. All outputs saved. ===")
