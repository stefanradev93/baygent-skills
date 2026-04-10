"""
Bayesian model for daily customer support ticket counts.

Problem: A SaaS company observes 5-15 tickets most days, with occasional spikes to 40+.
The variance appears larger than the mean, suggesting overdispersion. We model ticket counts
using a Negative Binomial likelihood (which naturally handles overdispersion) with
day_of_week and product_release as predictors.

Workflow:
  1. Generate synthetic data matching the described pattern
  2. Specify a Negative Binomial regression with day-of-week effects and a release indicator
  3. Run prior predictive checks
  4. Fit via MCMC (nutpie)
  5. Diagnose convergence
  6. Criticize the model (PPC, LOO-CV, calibration)
  7. Compare against a baseline Poisson model to demonstrate overdispersion handling
"""

import warnings

import arviz as az
import arviz_plots as azp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# 0. Reproducible seed (derived from analysis name, per skill rules)
# ---------------------------------------------------------------------------
RANDOM_SEED = sum(map(ord, "ticket-count-overdispersion-v1"))
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Generate synthetic data
# ---------------------------------------------------------------------------
N_DAYS = 180

day_of_week = rng.integers(0, 7, size=N_DAYS)  # 0=Mon ... 6=Sun
# ~10% of days have a product release
product_release = rng.binomial(1, 0.10, size=N_DAYS)

# True generative process (Negative Binomial):
#   log(mu) = intercept
#              + day_of_week effects (Mon busier, weekends quieter)
#              + release_effect * product_release
TRUE_INTERCEPT = np.log(10.0)  # baseline ~10 tickets/day
TRUE_DOW_EFFECTS = np.array([
    0.15,   # Mon: slightly above average
    0.05,   # Tue
    0.0,    # Wed (reference-ish)
    -0.05,  # Thu
    -0.10,  # Fri
    -0.30,  # Sat: fewer tickets
    -0.35,  # Sun: fewest
])
TRUE_RELEASE_EFFECT = 1.1  # releases roughly triple the rate (~exp(1.1) ~ 3x)
TRUE_ALPHA = 3.0  # NB dispersion: smaller = more overdispersion

log_mu = TRUE_INTERCEPT + TRUE_DOW_EFFECTS[day_of_week] + TRUE_RELEASE_EFFECT * product_release
mu = np.exp(log_mu)

# Draw from NB(mu, alpha) -- parameterized so that Var = mu + mu^2/alpha
tickets = rng.negative_binomial(
    n=TRUE_ALPHA,
    p=TRUE_ALPHA / (TRUE_ALPHA + mu),
)

df = pd.DataFrame({
    "day": np.arange(N_DAYS),
    "day_of_week": day_of_week,
    "product_release": product_release,
    "tickets": tickets,
})

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

print("=== Data summary ===")
print(f"N = {N_DAYS} days")
print(f"Mean tickets/day:     {tickets.mean():.1f}")
print(f"Variance tickets/day: {tickets.var():.1f}")
print(f"Variance / Mean:      {tickets.var() / tickets.mean():.2f}  (>>1 = overdispersed)")
print(f"Max tickets:          {tickets.max()}")
print(f"Days with release:    {product_release.sum()}")
print()

# ---------------------------------------------------------------------------
# 2. Model specification -- Negative Binomial regression
# ---------------------------------------------------------------------------
# Coordinates for labeled dimensions
coords = {
    "dow": DOW_NAMES,
    "obs_id": np.arange(N_DAYS),
}

with pm.Model(coords=coords) as nb_model:
    # --- Data containers ---
    dow_idx = pm.Data("dow_idx", day_of_week, dims="obs_id")
    release = pm.Data("release", product_release.astype(float), dims="obs_id")
    observed_tickets = pm.Data("observed_tickets", tickets, dims="obs_id")

    # --- Priors ---
    # Intercept: centered on log(10) ~ 2.3 because the user says most days
    # have 5-15 tickets; sigma=1 covers log(2) to log(50) at +/-2 SD.
    intercept = pm.Normal("intercept", mu=np.log(10), sigma=1.0)

    # Day-of-week effects: ZeroSumNormal so they sum to zero.
    # No natural reference day, so zero-sum avoids over-parameterization.
    # sigma=0.5 allows moderate day-to-day variation (roughly +/-50% on rate scale).
    dow_effect = pm.ZeroSumNormal("dow_effect", sigma=0.5, dims="dow")

    # Release effect: Normal(0.5, 0.5) -- centered slightly positive because
    # releases are expected to increase tickets; sigma=0.5 covers "no effect"
    # through "quadrupling" (exp(0.5+/-1) ~ 1 to 2.7x).
    release_effect = pm.Normal("release_effect", mu=0.5, sigma=0.5)

    # Dispersion parameter alpha: Gamma(2, 0.5) keeps alpha positive and away
    # from zero (mean=4, mostly between 1-10). Smaller alpha = more overdispersion.
    alpha = pm.Gamma("alpha", alpha=2, beta=0.5)

    # --- Linear predictor ---
    log_mu = intercept + dow_effect[dow_idx] + release_effect * release

    # --- Likelihood ---
    # NegativeBinomial parameterized by mu and alpha:
    #   Var(Y) = mu + mu^2 / alpha
    pm.NegativeBinomial(
        "tickets",
        mu=pm.math.exp(log_mu),
        alpha=alpha,
        observed=observed_tickets,
        dims="obs_id",
    )

    # --- Prior predictive check ---
    print("Sampling prior predictive...")
    prior_pred = pm.sample_prior_predictive(random_seed=rng)

# ---------------------------------------------------------------------------
# 3. Prior predictive check -- verify priors produce plausible ticket counts
# ---------------------------------------------------------------------------
print("\n=== Prior predictive check ===")
prior_tickets = prior_pred.prior_predictive["tickets"].values.flatten()
print(f"Prior predictive mean:   {prior_tickets.mean():.1f}")
print(f"Prior predictive median: {np.median(prior_tickets):.1f}")
print(f"Prior predictive 2.5%:   {np.percentile(prior_tickets, 2.5):.0f}")
print(f"Prior predictive 97.5%:  {np.percentile(prior_tickets, 97.5):.0f}")
print("Priors look plausible if most values are in a reasonable range for daily ticket counts.")

fig, ax = plt.subplots(figsize=(8, 4))
az.plot_ppc(prior_pred, group="prior", num_pp_samples=100, ax=ax)
ax.set_title("Prior Predictive Check -- NegBinomial Model")
ax.set_xlabel("Tickets per day")
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-3/count-data-overdispersion/with_skill/outputs/prior_predictive.png",
    dpi=150,
)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Inference
# ---------------------------------------------------------------------------
print("\n=== Sampling posterior (nutpie) ===")
with nb_model:
    idata = pm.sample(nuts_sampler="nutpie", random_seed=rng)
    idata.extend(prior_pred)

# ---------------------------------------------------------------------------
# 5. Posterior predictive
# ---------------------------------------------------------------------------
print("\n=== Sampling posterior predictive ===")
with nb_model:
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=rng))

# --- Save immediately after sampling (skill rule: save before post-processing) ---
SAVE_PATH = (
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-3/count-data-overdispersion/with_skill/outputs/nb_model_output.nc"
)
idata.to_netcdf(SAVE_PATH)
print(f"InferenceData saved to {SAVE_PATH}")

# ---------------------------------------------------------------------------
# 6. Convergence diagnostics
# ---------------------------------------------------------------------------
print("\n=== Convergence diagnostics ===")
summary = az.summary(idata, round_to=3)
print(summary)

num_chains = idata.posterior.sizes["chain"]
rhat_ok = (summary["r_hat"] <= 1.01).all()
ess_bulk_ok = (summary["ess_bulk"] >= 100 * num_chains).all()
ess_tail_ok = (summary["ess_tail"] >= 100 * num_chains).all()
n_div = idata.sample_stats["diverging"].sum().item()

print(f"\nR-hat all <= 1.01:    {rhat_ok}")
print(f"ESS bulk >= {100 * num_chains}:     {ess_bulk_ok} (min: {summary['ess_bulk'].min():.0f})")
print(f"ESS tail >= {100 * num_chains}:     {ess_tail_ok} (min: {summary['ess_tail'].min():.0f})")
print(f"Divergences:          {n_div}")

# Trace / rank plots
fig = az.plot_trace(idata, kind="rank_vlines").flatten()[0].get_figure()
fig.suptitle("Trace / Rank Plots -- NegBinomial Model", y=1.02)
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-3/count-data-overdispersion/with_skill/outputs/trace_rank_plots.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)

# Energy plot
fig, ax = plt.subplots(figsize=(6, 4))
az.plot_energy(idata, ax=ax)
ax.set_title("Energy Diagnostic")
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-3/count-data-overdispersion/with_skill/outputs/energy_plot.png",
    dpi=150,
)
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Model criticism
# ---------------------------------------------------------------------------

# --- 7a. Posterior predictive check ---
print("\n=== Posterior predictive check ===")
fig, ax = plt.subplots(figsize=(8, 4))
az.plot_ppc(idata, num_pp_samples=100, ax=ax)
ax.set_title("Posterior Predictive Check -- NegBinomial Model")
ax.set_xlabel("Tickets per day")
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-3/count-data-overdispersion/with_skill/outputs/ppc_nb_model.png",
    dpi=150,
)
plt.close(fig)

# --- 7b. LOO-CV ---
print("\n=== LOO-CV ===")
with nb_model:
    pm.compute_log_likelihood(idata, model=nb_model)

loo_nb = az.loo(idata, pointwise=True)
print(loo_nb)

# Pareto k diagnostic
pareto_k = loo_nb.pareto_k.values
bad_obs = np.where(pareto_k > 0.7)[0]
print(f"\nObservations with Pareto k > 0.7: {bad_obs if len(bad_obs) > 0 else 'None'}")

fig, ax = plt.subplots(figsize=(8, 4))
az.plot_khat(loo_nb, ax=ax)
ax.set_title("Pareto k Diagnostic -- NegBinomial Model")
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-3/count-data-overdispersion/with_skill/outputs/pareto_k_nb.png",
    dpi=150,
)
plt.close(fig)

# --- 7c. Calibration (PIT) ---
print("\n=== Calibration (PPC-PIT) ===")
try:
    fig, ax = plt.subplots(figsize=(6, 5))
    azp.plot_ppc_pit(idata, ax=ax)
    ax.set_title("PPC-PIT Calibration -- NegBinomial Model")
    fig.tight_layout()
    fig.savefig(
        "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
        "iteration-3/count-data-overdispersion/with_skill/outputs/pit_calibration_nb.png",
        dpi=150,
    )
    plt.close(fig)
    print("PIT calibration plot saved.")
except Exception as e:
    print(f"PIT calibration plot encountered an issue: {e}")
    print("Falling back to LOO-PIT via az.plot_loo_pit...")
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        az.plot_loo_pit(idata, y="tickets", ax=ax)
        ax.set_title("LOO-PIT Calibration -- NegBinomial Model")
        fig.tight_layout()
        fig.savefig(
            "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
            "iteration-3/count-data-overdispersion/with_skill/outputs/pit_calibration_nb.png",
            dpi=150,
        )
        plt.close(fig)
        print("LOO-PIT calibration plot saved (fallback).")
    except Exception as e2:
        print(f"LOO-PIT also failed: {e2}. Skipping calibration plot.")

# --- 7d. Residual analysis ---
print("\n=== Residual analysis ===")
pp_mean = idata.posterior_predictive["tickets"].mean(dim=["chain", "draw"]).values
residuals = tickets - pp_mean

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Residuals vs. fitted
axes[0].scatter(pp_mean, residuals, alpha=0.5, s=20)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Fitted values (posterior predictive mean)")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs. Fitted")

# Residuals by day of week
for d in range(7):
    mask = day_of_week == d
    axes[1].scatter(
        np.full(mask.sum(), d) + rng.normal(0, 0.1, size=mask.sum()),
        residuals[mask],
        alpha=0.5,
        s=20,
    )
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("Day of week")
axes[1].set_ylabel("Residuals")
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(DOW_NAMES, rotation=45)
axes[1].set_title("Residuals by Day of Week")

fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-3/count-data-overdispersion/with_skill/outputs/residuals_nb.png",
    dpi=150,
)
plt.close(fig)

# ---------------------------------------------------------------------------
# 8. Model comparison: NegBinomial vs. Poisson (demonstrate overdispersion)
# ---------------------------------------------------------------------------
print("\n=== Fitting baseline Poisson model for comparison ===")

with pm.Model(coords=coords) as poisson_model:
    # --- Data containers ---
    dow_idx_p = pm.Data("dow_idx", day_of_week, dims="obs_id")
    release_p = pm.Data("release", product_release.astype(float), dims="obs_id")
    observed_tickets_p = pm.Data("observed_tickets", tickets, dims="obs_id")

    # Same priors for comparable coefficients
    intercept_p = pm.Normal("intercept", mu=np.log(10), sigma=1.0)
    dow_effect_p = pm.ZeroSumNormal("dow_effect", sigma=0.5, dims="dow")
    release_effect_p = pm.Normal("release_effect", mu=0.5, sigma=0.5)

    log_mu_p = intercept_p + dow_effect_p[dow_idx_p] + release_effect_p * release_p

    # Poisson likelihood -- Var(Y) = mu, no overdispersion
    pm.Poisson(
        "tickets",
        mu=pm.math.exp(log_mu_p),
        observed=observed_tickets_p,
        dims="obs_id",
    )

    idata_poisson = pm.sample(nuts_sampler="nutpie", random_seed=rng)
    idata_poisson.extend(pm.sample_posterior_predictive(idata_poisson, random_seed=rng))

# Save Poisson results
SAVE_PATH_POISSON = (
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-3/count-data-overdispersion/with_skill/outputs/poisson_model_output.nc"
)
idata_poisson.to_netcdf(SAVE_PATH_POISSON)

# Compute log-likelihood for Poisson model
with poisson_model:
    pm.compute_log_likelihood(idata_poisson, model=poisson_model)

# --- Poisson PPC ---
fig, ax = plt.subplots(figsize=(8, 4))
az.plot_ppc(idata_poisson, num_pp_samples=100, ax=ax)
ax.set_title("Posterior Predictive Check -- Poisson Model (no overdispersion)")
ax.set_xlabel("Tickets per day")
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-3/count-data-overdispersion/with_skill/outputs/ppc_poisson_model.png",
    dpi=150,
)
plt.close(fig)

# --- LOO comparison ---
print("\n=== Model comparison (LOO-CV) ===")
loo_poisson = az.loo(idata_poisson, pointwise=True)
print("Poisson LOO:")
print(loo_poisson)

comparison = az.compare(
    {"NegBinomial": idata, "Poisson": idata_poisson},
)
print("\n--- LOO Comparison Table ---")
print(comparison)

fig = az.plot_compare(comparison)
if hasattr(fig, "get_figure"):
    fig = fig.get_figure()
elif hasattr(fig, "figure"):
    fig = fig.figure
else:
    fig = plt.gcf()
fig.suptitle("Model Comparison: NegBinomial vs. Poisson", y=1.02)
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-3/count-data-overdispersion/with_skill/outputs/model_comparison.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)

# ---------------------------------------------------------------------------
# 9. Summary of key results
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RESULTS SUMMARY -- Negative Binomial Model")
print("=" * 60)

# Parameter estimates with 94% HDI
print("\n--- Parameter estimates (94% HDI) ---")
summary_94 = az.summary(idata, hdi_prob=0.94, round_to=3)
print(summary_94)

# Interpret release effect on the rate scale
release_samples = idata.posterior["release_effect"].values.flatten()
release_multiplier = np.exp(release_samples)
print(f"\nRelease effect (log scale): mean = {release_samples.mean():.3f}, "
      f"94% HDI = [{np.percentile(release_samples, 3):.3f}, "
      f"{np.percentile(release_samples, 97):.3f}]")
print(f"Release multiplier (rate scale): mean = {release_multiplier.mean():.2f}x, "
      f"94% HDI = [{np.percentile(release_multiplier, 3):.2f}x, "
      f"{np.percentile(release_multiplier, 97):.2f}x]")

# Day-of-week effects
print("\n--- Day-of-week effects (log-rate scale) ---")
dow_samples = idata.posterior["dow_effect"].mean(dim=["chain", "draw"]).values
dow_hdi = az.hdi(idata, var_names=["dow_effect"], hdi_prob=0.94)
for i, day_name in enumerate(DOW_NAMES):
    hdi_low = dow_hdi["dow_effect"].values[i, 0]
    hdi_high = dow_hdi["dow_effect"].values[i, 1]
    print(f"  {day_name}: {dow_samples[i]:+.3f}  94% HDI [{hdi_low:+.3f}, {hdi_high:+.3f}]")

# Overdispersion
alpha_samples = idata.posterior["alpha"].values.flatten()
print(f"\nalpha (dispersion): mean = {alpha_samples.mean():.2f}, "
      f"94% HDI = [{np.percentile(alpha_samples, 3):.2f}, "
      f"{np.percentile(alpha_samples, 97):.2f}]")
print(f"(Smaller alpha = more overdispersion. "
      f"At the posterior mean rate of ~{np.exp(idata.posterior['intercept'].values.flatten().mean()):.0f} tickets, "
      f"Var/Mean ~ 1 + mu/alpha ~ {1 + np.exp(idata.posterior['intercept'].values.flatten().mean()) / alpha_samples.mean():.1f})")

# Forest plot
fig = az.plot_forest(
    idata,
    var_names=["intercept", "release_effect", "dow_effect", "alpha"],
    combined=True,
    hdi_prob=0.94,
).flatten()[0].get_figure()
fig.suptitle("Parameter Estimates (94% HDI)", y=1.02)
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-3/count-data-overdispersion/with_skill/outputs/forest_plot.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)

# Model graph
try:
    graph = pm.model_to_graphviz(nb_model)
    graph.render(
        "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
        "iteration-3/count-data-overdispersion/with_skill/outputs/model_graph",
        format="png",
        cleanup=True,
    )
    print("\nModel graph saved.")
except Exception as e:
    print(f"\nModel graph rendering skipped: {e}")

print("\n=== Analysis complete. All outputs saved. ===")
