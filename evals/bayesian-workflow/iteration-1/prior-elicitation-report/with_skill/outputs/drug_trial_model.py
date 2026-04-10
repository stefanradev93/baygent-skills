"""
Bayesian Analysis: Effect of a New Drug on Blood Pressure
=========================================================

This script implements a complete Bayesian workflow for analyzing
the effect of a new drug on blood pressure, using pre/post measurements
from a randomized controlled trial with 80 treatment and 80 control patients.

Workflow steps (following the bayesian-workflow skill):
  1. Formulate the generative story
  2. Specify and justify priors
  3. Implement in PyMC
  4. Run prior predictive checks
  5. Inference (MCMC sampling)
  6. Diagnose convergence
  7. Criticize the model (PPC, LOO-CV, calibration)
  8. Report results

Requirements:
    pip install pymc nutpie arviz numpy pandas matplotlib preliz
"""

import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

warnings.filterwarnings("ignore", category=FutureWarning)

# Reproducibility
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ============================================================================
# STEP 0: Generate synthetic data
# ============================================================================
# Realistic scenario: a drug that lowers systolic blood pressure.
# Baseline SBP ~ Normal(135, 12) for a mildly hypertensive cohort.
# True drug effect: -8 mmHg (treatment lowers BP by 8 on average).
# Control group shows a small regression-to-the-mean / placebo effect of -2 mmHg.
# Individual variation in response: SD ~ 6 mmHg.

N_TREATMENT = 80
N_CONTROL = 80
N_TOTAL = N_TREATMENT + N_CONTROL

TRUE_DRUG_EFFECT = -8.0      # mmHg (the treatment lowers BP)
TRUE_PLACEBO_EFFECT = -2.0   # mmHg (small placebo / regression to mean)
TRUE_BASELINE_MEAN = 135.0   # mmHg (mildly hypertensive population)
TRUE_BASELINE_SD = 12.0      # mmHg (between-patient variability in baseline)
TRUE_RESPONSE_SD = 6.0       # mmHg (within-patient variability in change)

# Generate baseline (pre-treatment) blood pressures
bp_pre_treatment = rng.normal(TRUE_BASELINE_MEAN, TRUE_BASELINE_SD, N_TREATMENT)
bp_pre_control = rng.normal(TRUE_BASELINE_MEAN, TRUE_BASELINE_SD, N_CONTROL)

# Generate post-treatment blood pressures
# Change = group effect + individual noise
bp_post_treatment = bp_pre_treatment + rng.normal(
    TRUE_DRUG_EFFECT, TRUE_RESPONSE_SD, N_TREATMENT
)
bp_post_control = bp_pre_control + rng.normal(
    TRUE_PLACEBO_EFFECT, TRUE_RESPONSE_SD, N_CONTROL
)

# Compute the change (post - pre) for each patient
change_treatment = bp_post_treatment - bp_pre_treatment
change_control = bp_post_control - bp_pre_control

# Assemble into a DataFrame
df = pd.DataFrame(
    {
        "patient_id": np.arange(N_TOTAL),
        "group": ["treatment"] * N_TREATMENT + ["control"] * N_CONTROL,
        "bp_pre": np.concatenate([bp_pre_treatment, bp_pre_control]),
        "bp_post": np.concatenate([bp_post_treatment, bp_post_control]),
        "bp_change": np.concatenate([change_treatment, change_control]),
    }
)

# Create a numeric indicator: treatment = 1, control = 0
df["is_treatment"] = (df["group"] == "treatment").astype(int)

print("=== Data Summary ===")
print(f"Treatment group (n={N_TREATMENT}):")
print(f"  Mean BP change: {change_treatment.mean():.1f} mmHg")
print(f"  SD BP change:   {change_treatment.std():.1f} mmHg")
print(f"Control group (n={N_CONTROL}):")
print(f"  Mean BP change: {change_control.mean():.1f} mmHg")
print(f"  SD BP change:   {change_control.std():.1f} mmHg")
print()

# ============================================================================
# STEP 1: Formulate the generative story
# ============================================================================
# The data-generating process we assume:
#
#   Each patient i has a pre-treatment blood pressure measurement.
#   After the intervention period, they have a post-treatment measurement.
#   The change in BP (post - pre) for patient i is:
#
#       bp_change_i ~ Normal(mu_i, sigma)
#
#   where:
#       mu_i = alpha + beta * is_treatment_i
#
#   - alpha is the baseline change in the control group
#     (captures placebo effect + regression to the mean)
#   - beta is the ADDITIONAL effect of the drug beyond the control change
#   - sigma is the residual standard deviation (patient-to-patient variability)
#
#   So the drug's total effect on BP change is alpha + beta for the treatment group,
#   and the causal effect attributable to the drug (vs. control) is beta.

# ============================================================================
# STEP 2: Specify and justify priors
# ============================================================================
# Prior reasoning (documented as required by the skill):
#
# alpha (control group mean change):
#   Prior: Normal(0, 5)
#   Justification: The control group change is expected to be small (regression
#   to the mean, placebo). Centering at 0 with SD=5 allows changes from roughly
#   -10 to +10 mmHg, covering typical placebo effects without favoring any
#   direction strongly.
#
# beta (drug effect, additional change beyond control):
#   Prior: Normal(-7.5, 6.25)
#   Justification: Domain knowledge says the drug effect (the TOTAL change,
#   treatment - control) could range from -20 to +5 mmHg. Using the
#   rule of thumb from priors.md: mu = (plausible_max + plausible_min) / 2
#   = (+5 + -20) / 2 = -7.5 and sigma = (plausible_max - plausible_min) / 4
#   = (5 - (-20)) / 4 = 6.25. This gives a weakly informative prior that
#   places most mass on the plausible range while allowing the data to dominate.
#
# sigma (residual SD of BP change):
#   Prior: Gamma(alpha=2, beta=0.2)
#   Justification: We expect patient-to-patient variability in BP change to be
#   on the order of 5-15 mmHg. Gamma(2, 0.2) has a mean of 10 and puts most
#   mass between roughly 1 and 25, covering this range. Using Gamma avoids
#   near-zero values (which would imply no patient variability, unrealistic).

# ============================================================================
# STEP 3: Implement in PyMC
# ============================================================================

# Define coordinates and data (following skill template)
coords = {
    "obs_id": df["patient_id"].values,
    "group_name": ["control", "treatment"],
}

with pm.Model(coords=coords) as bp_model:
    # --- Data containers ---
    bp_change_obs = pm.Data(
        "bp_change_obs", df["bp_change"].values, dims="obs_id"
    )
    treatment_indicator = pm.Data(
        "treatment_indicator", df["is_treatment"].values, dims="obs_id"
    )

    # --- Priors ---
    # Control group mean change (placebo + regression to mean)
    # Weakly informative: allows ~[-10, +10] mmHg
    alpha = pm.Normal(
        "alpha", mu=0, sigma=5
    )  # Control group baseline change

    # Drug effect (additional to control): domain knowledge says -20 to +5 mmHg
    # Prior centered on midpoint with SD = range/4
    beta = pm.Normal(
        "beta", mu=-7.5, sigma=6.25
    )  # Additional drug effect beyond control

    # Residual SD: patient-to-patient variability in BP change
    # Gamma(2, 0.2): mean=10, avoids near-zero, allows 1-25 mmHg range
    sigma = pm.Gamma(
        "sigma", alpha=2, beta=0.2
    )  # Residual standard deviation

    # --- Deterministic: group means for interpretability ---
    mu_control = pm.Deterministic("mu_control", alpha)
    mu_treatment = pm.Deterministic("mu_treatment", alpha + beta)

    # --- Expected value per observation ---
    mu = alpha + beta * treatment_indicator

    # --- Likelihood ---
    pm.Normal(
        "bp_change_likelihood",
        mu=mu,
        sigma=sigma,
        observed=bp_change_obs,
        dims="obs_id",
    )

# Print model structure
print("=== Model Structure ===")
print(bp_model)
print()

# ============================================================================
# STEP 4: Prior predictive checks
# ============================================================================
# MANDATORY per skill: never skip this step.

with bp_model:
    prior_pred = pm.sample_prior_predictive(random_seed=RANDOM_SEED)

# Visualize prior predictive distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Prior predictive for BP change (all observations)
az.plot_ppc(prior_pred, group="prior", num_pp_samples=200, ax=axes[0])
axes[0].set_title("Prior Predictive Check: BP Change Distribution")
axes[0].set_xlabel("Blood Pressure Change (mmHg)")

# Prior distributions for key parameters
prior_beta = prior_pred.prior["beta"].values.flatten()
axes[1].hist(prior_beta, bins=50, density=True, alpha=0.7, color="steelblue")
axes[1].axvline(x=-20, color="red", linestyle="--", label="Plausible min (-20)")
axes[1].axvline(x=5, color="red", linestyle="--", label="Plausible max (+5)")
axes[1].set_title("Prior on Drug Effect (beta)")
axes[1].set_xlabel("Drug Effect (mmHg)")
axes[1].legend()

plt.tight_layout()
plt.savefig("prior_predictive_check.png", dpi=150, bbox_inches="tight")
plt.show()
print("Prior predictive check saved to prior_predictive_check.png")

# Decision rule (from priors.md):
# Check if >10% of prior predictive samples are clearly implausible
prior_bp_change = prior_pred.prior_predictive["bp_change_likelihood"].values.flatten()
pct_extreme = np.mean(np.abs(prior_bp_change) > 50) * 100
print(f"\n% of prior predictive BP changes > |50| mmHg: {pct_extreme:.1f}%")
if pct_extreme > 10:
    print("WARNING: >10% of prior predictions are implausible. Consider tightening priors.")
else:
    print("Prior predictive check PASSED: predictions are within plausible range.")
print()

# ============================================================================
# STEP 5: Inference (MCMC sampling)
# ============================================================================

with bp_model:
    # Use nutpie for faster sampling when available (as recommended by skill)
    idata = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        nuts_sampler="nutpie",
        random_seed=RANDOM_SEED,
    )
    # Extend with prior predictive samples
    idata.extend(prior_pred)

    # Sample posterior predictive (mandatory per skill)
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=RANDOM_SEED))

# ============================================================================
# STEP 6: Diagnose convergence
# ============================================================================
# Following diagnostics.md: R-hat, ESS, divergences, trace plots

print("=== Convergence Diagnostics ===")

# 1. Summary table (R-hat + ESS at a glance)
summary = az.summary(idata, var_names=["alpha", "beta", "sigma"], round_to=3)
print(summary)
print()

# 2. Check R-hat
rhat_ok = (summary["r_hat"] <= 1.01).all()
print(f"R-hat OK (all <= 1.01): {rhat_ok}")

# 3. Check ESS (bulk and tail)
num_chains = len(idata.posterior.chain)
ess_bulk_ok = (summary["ess_bulk"] >= 100 * num_chains).all()
ess_tail_ok = (summary["ess_tail"] >= 100 * num_chains).all()
print(f"ESS bulk OK (all >= {100 * num_chains}): {ess_bulk_ok}")
print(f"ESS tail OK (all >= {100 * num_chains}): {ess_tail_ok}")

# 4. Check divergences
n_div = int(idata.sample_stats["diverging"].sum())
print(f"Divergences: {n_div}")
if n_div > 0:
    print("WARNING: Divergent transitions detected. Results may be biased.")
print()

# 5. Visual diagnostics: trace/rank plots
fig = az.plot_trace(
    idata,
    var_names=["alpha", "beta", "sigma"],
    kind="rank_vlines",
)
plt.suptitle("Trace and Rank Plots", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("trace_rank_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("Trace/rank plots saved to trace_rank_plots.png")

# Energy diagnostic
az.plot_energy(idata)
plt.savefig("energy_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Energy plot saved to energy_plot.png")
print()

# ============================================================================
# STEP 7: Model criticism
# ============================================================================
# Following model-criticism.md

print("=== Model Criticism ===")

# --- 7a. Posterior predictive check ---
fig, ax = plt.subplots(figsize=(10, 5))
az.plot_ppc(idata, num_pp_samples=100, ax=ax)
ax.set_title("Posterior Predictive Check: BP Change")
ax.set_xlabel("Blood Pressure Change (mmHg)")
plt.tight_layout()
plt.savefig("posterior_predictive_check.png", dpi=150, bbox_inches="tight")
plt.show()
print("Posterior predictive check saved to posterior_predictive_check.png")

# --- 7b. LOO-CV ---
loo = az.loo(idata, var_name="bp_change_likelihood", pointwise=True)
print(f"\nLOO-CV Results:")
print(f"  ELPD LOO: {loo.elpd_loo:.1f} (SE: {loo.se:.1f})")
print(f"  p_loo (effective parameters): {loo.p_loo:.1f}")

# Pareto k diagnostic
pareto_k = loo.pareto_k.values
n_bad_k = int(np.sum(pareto_k > 0.7))
n_marginal_k = int(np.sum((pareto_k > 0.5) & (pareto_k <= 0.7)))
print(f"  Pareto k: max = {pareto_k.max():.2f}")
print(f"    Observations with k > 0.7 (unreliable): {n_bad_k}")
print(f"    Observations with 0.5 < k <= 0.7 (marginal): {n_marginal_k}")

if n_bad_k > 0:
    bad_obs = np.where(pareto_k > 0.7)[0]
    print(f"    Problematic observations: {bad_obs}")
    print("    Consider K-fold CV or moment matching for these observations.")
else:
    print("    All Pareto k values OK (< 0.7).")

# Pareto k plot
az.plot_khat(loo)
plt.title("Pareto k Diagnostic")
plt.savefig("pareto_k_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Pareto k plot saved to pareto_k_plot.png")

# --- 7c. Residual analysis ---
pp_mean = (
    idata.posterior_predictive["bp_change_likelihood"]
    .mean(dim=["chain", "draw"])
    .values
)
residuals = df["bp_change"].values - pp_mean

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals vs. fitted
axes[0].scatter(pp_mean, residuals, alpha=0.5, color="steelblue")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Fitted Values (mmHg)")
axes[0].set_ylabel("Residuals (mmHg)")
axes[0].set_title("Residuals vs. Fitted")

# Residuals by group
for i, group in enumerate(["control", "treatment"]):
    mask = df["group"] == group
    axes[1].scatter(
        df.loc[mask, "bp_pre"],
        residuals[mask],
        alpha=0.5,
        label=group.capitalize(),
    )
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("Baseline BP (mmHg)")
axes[1].set_ylabel("Residuals (mmHg)")
axes[1].set_title("Residuals vs. Baseline BP")
axes[1].legend()

plt.tight_layout()
plt.savefig("residual_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("Residual plots saved to residual_plots.png")
print()

# ============================================================================
# STEP 8: Report results
# ============================================================================
# Following reporting.md: full posteriors, HDI, probability language

print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

# Parameter estimates with 94% HDI (as per skill: default 94%, not 95%)
summary_94 = az.summary(
    idata,
    var_names=["alpha", "beta", "sigma", "mu_control", "mu_treatment"],
    hdi_prob=0.94,
    round_to=2,
)
print("\nParameter Estimates (94% HDI):")
print(summary_94)
print()

# Extract posterior samples for key quantities
beta_posterior = idata.posterior["beta"].values.flatten()
alpha_posterior = idata.posterior["alpha"].values.flatten()
mu_treatment_posterior = idata.posterior["mu_treatment"].values.flatten()
mu_control_posterior = idata.posterior["mu_control"].values.flatten()

# Key summary statistics
beta_mean = beta_posterior.mean()
beta_hdi = az.hdi(idata, var_names=["beta"], hdi_prob=0.94)["beta"].values
prob_negative = (beta_posterior < 0).mean()
prob_clinically_meaningful = (beta_posterior < -3).mean()

print("=== Key Findings ===")
print(f"Drug effect (beta):")
print(f"  Mean: {beta_mean:.1f} mmHg")
print(f"  94% HDI: [{beta_hdi[0]:.1f}, {beta_hdi[1]:.1f}] mmHg")
print(f"  Probability that drug lowers BP (beta < 0): {prob_negative:.1%}")
print(
    f"  Probability of clinically meaningful effect (> 3 mmHg reduction): "
    f"{prob_clinically_meaningful:.1%}"
)
print()
print(f"Control group mean change:")
print(f"  Mean: {alpha_posterior.mean():.1f} mmHg")
print(
    f"  94% HDI: [{az.hdi(idata, var_names=['alpha'], hdi_prob=0.94)['alpha'].values[0]:.1f}, "
    f"{az.hdi(idata, var_names=['alpha'], hdi_prob=0.94)['alpha'].values[1]:.1f}] mmHg"
)
print()
print(f"Treatment group mean change:")
print(f"  Mean: {mu_treatment_posterior.mean():.1f} mmHg")
print(
    f"  94% HDI: [{az.hdi(idata, var_names=['mu_treatment'], hdi_prob=0.94)['mu_treatment'].values[0]:.1f}, "
    f"{az.hdi(idata, var_names=['mu_treatment'], hdi_prob=0.94)['mu_treatment'].values[1]:.1f}] mmHg"
)

# --- Visualization: Forest plot ---
fig = az.plot_forest(
    idata,
    var_names=["alpha", "beta"],
    combined=True,
    hdi_prob=0.94,
    figsize=(10, 4),
)
plt.title("Parameter Estimates with 94% HDI")
plt.savefig("forest_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nForest plot saved to forest_plot.png")

# --- Visualization: Posterior of drug effect ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(beta_posterior, bins=60, density=True, alpha=0.7, color="steelblue",
        label="Posterior")
ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="No effect")
ax.axvline(x=-3, color="orange", linestyle="--", linewidth=2,
           label="Clinically meaningful (-3 mmHg)")
ax.axvline(x=beta_mean, color="darkblue", linewidth=2, label=f"Mean = {beta_mean:.1f}")

# Shade the HDI
ax.axvspan(beta_hdi[0], beta_hdi[1], alpha=0.15, color="steelblue",
           label=f"94% HDI [{beta_hdi[0]:.1f}, {beta_hdi[1]:.1f}]")

ax.set_xlabel("Drug Effect on BP Change (mmHg)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Posterior Distribution of Drug Effect", fontsize=14)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("drug_effect_posterior.png", dpi=150, bbox_inches="tight")
plt.show()
print("Drug effect posterior saved to drug_effect_posterior.png")

# --- Visualization: Model graph ---
try:
    graph = pm.model_to_graphviz(bp_model)
    graph.render("model_graph", format="png", cleanup=True)
    print("Model graph saved to model_graph.png")
except Exception as e:
    print(f"Could not generate model graph (graphviz may not be installed): {e}")

# --- Save inference data for later use ---
idata.to_netcdf("bp_drug_trial_inference_data.nc")
print("\nInference data saved to bp_drug_trial_inference_data.nc")

# --- Optional: run the automated diagnostics script ---
# python scripts/diagnose_model.py --idata bp_drug_trial_inference_data.nc
# python scripts/calibration_check.py --idata bp_drug_trial_inference_data.nc --save-plots

print("\n=== Analysis complete ===")
print("All figures and inference data have been saved.")
print("Review the plots and summary to assess model quality before reporting.")
