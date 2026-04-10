"""
Bayesian analysis of a drug trial: effect of a new drug on blood pressure.

Design: Pre/post measurements for 80 treatment patients and 80 controls.
Outcome: Change in blood pressure (post - pre) in mmHg.
Goal: Estimate the treatment effect (difference in BP change between groups).

Full Bayesian workflow:
1. Formulate the generative story
2. Specify priors with justification
3. Implement in PyMC
4. Prior predictive checks
5. Inference
6. Convergence diagnostics
7. Model criticism (PPC, LOO-CV, calibration)
8. Report results
"""

import warnings

import arviz as az
import arviz_plots as azp
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# 0. REPRODUCIBLE SEED
# ============================================================================
# Descriptive seed derived from the analysis name — never use magic numbers like 42
RANDOM_SEED = sum(map(ord, "drug-trial-blood-pressure"))
rng = np.random.default_rng(RANDOM_SEED)

# ============================================================================
# 1. GENERATE SYNTHETIC DATA
# ============================================================================
# We simulate a realistic clinical trial:
# - 80 patients in treatment group, 80 in control group
# - Each patient has a pre-treatment and post-treatment BP measurement
# - The drug is expected to lower BP by about 10 mmHg (based on similar drugs)
# - Control group has small natural variation (regression to the mean, ~-2 mmHg)

N_TREATMENT = 80
N_CONTROL = 80
N_TOTAL = N_TREATMENT + N_CONTROL

# True parameters (for data generation only — the model does not know these)
TRUE_BASELINE_CHANGE = -2.0  # Natural change in control group (mmHg)
TRUE_TREATMENT_EFFECT = -8.0  # Additional effect of drug (mmHg), so total = -10
TRUE_SIGMA = 7.0  # Within-group SD of BP change

# Group indicator: 0 = control, 1 = treatment
group = np.concatenate([np.zeros(N_CONTROL), np.ones(N_TREATMENT)]).astype(int)
group_labels = np.array(["control"] * N_CONTROL + ["treatment"] * N_TREATMENT)

# Simulate BP changes (post - pre)
bp_change = (
    TRUE_BASELINE_CHANGE
    + TRUE_TREATMENT_EFFECT * group
    + rng.normal(0, TRUE_SIGMA, N_TOTAL)
)

# Quick data summary
print("=" * 60)
print("DATA SUMMARY")
print("=" * 60)
print(f"Control group (n={N_CONTROL}):")
print(f"  Mean BP change: {bp_change[group == 0].mean():.1f} mmHg")
print(f"  SD BP change:   {bp_change[group == 0].std():.1f} mmHg")
print(f"\nTreatment group (n={N_TREATMENT}):")
print(f"  Mean BP change: {bp_change[group == 1].mean():.1f} mmHg")
print(f"  SD BP change:   {bp_change[group == 1].std():.1f} mmHg")
print(f"\nRaw difference in means: {bp_change[group == 1].mean() - bp_change[group == 0].mean():.1f} mmHg")

# ============================================================================
# 2. FORMULATE THE GENERATIVE STORY
# ============================================================================
# Each patient's BP change is drawn from a Normal distribution whose mean
# depends on which group they belong to:
#
#   bp_change_i ~ Normal(mu_control + treatment_effect * group_i, sigma)
#
# Where:
#   mu_control       = average BP change in the control group
#   treatment_effect = additional BP change caused by the drug
#   sigma            = within-group standard deviation of BP changes
#   group_i          = 0 for control, 1 for treatment
#
# This is a simple two-group comparison with a shared residual SD.

# ============================================================================
# 3. SPECIFY PRIORS (with justifications)
# ============================================================================
# The user says: "the blood pressure difference could be anywhere from -20 to
# +5 mmHg based on similar drugs."
#
# Prior choices:
#
# mu_control ~ Normal(-2, 5)
#   Justification: BP can naturally drift slightly down (regression to mean).
#   Center on -2 mmHg with SD=5, so 94% of prior mass spans roughly [-12, +8].
#   This is weakly informative — allows a wide range of natural changes.
#
# treatment_effect ~ Normal(-7.5, 6.25)
#   Justification: Based on the user's domain knowledge that similar drugs
#   produce effects from -20 to +5 mmHg. We center the prior at the midpoint
#   of [-20, +5] = -7.5, and set sigma = (5 - (-20)) / 4 = 6.25, so that
#   the prior 94% interval spans roughly [-20, +5]. This faithfully encodes
#   the user's stated range.
#
# sigma ~ Gamma(2, 0.3)
#   Justification: BP measurements typically vary by 5-15 mmHg within groups.
#   Gamma(2, 0.3) has mean ~6.7, mode ~3.3, and allows values roughly from
#   1 to 20. Avoids near-zero (which would be implausible for BP data) and
#   avoids excessively large values. Gamma avoids the near-zero region that
#   causes sampling problems.

# ============================================================================
# 4. BUILD THE MODEL
# ============================================================================
coords = {
    "obs_id": np.arange(N_TOTAL),
    "group_name": ["control", "treatment"],
}

with pm.Model(coords=coords) as bp_model:
    # Data containers
    group_idx = pm.Data("group_idx", group, dims="obs_id")
    bp_observed = pm.Data("bp_observed", bp_change, dims="obs_id")

    # --- Priors ---
    # Average BP change in the control group
    # Weakly informative: allows natural drift, centered near small decrease
    mu_control = pm.Normal("mu_control", mu=-2, sigma=5)

    # Treatment effect (additional change from the drug)
    # Based on user's domain knowledge: effects of similar drugs range -20 to +5
    # Center at midpoint (-7.5), sigma = range/4 = 6.25
    treatment_effect = pm.Normal("treatment_effect", mu=-7.5, sigma=6.25)

    # Group means as a Deterministic for easy reporting
    mu = pm.Deterministic(
        "mu_group",
        pm.math.stack([mu_control, mu_control + treatment_effect]),
        dims="group_name",
    )

    # Within-group variability of BP changes
    # Gamma(2, 0.3): mean ~6.7 mmHg, avoids near-zero, allows 1-20 range
    sigma = pm.Gamma("sigma", alpha=2, beta=0.3)

    # --- Likelihood ---
    bp_obs = pm.Normal(
        "bp_change",
        mu=mu[group_idx],
        sigma=sigma,
        observed=bp_observed,
        dims="obs_id",
    )

# ============================================================================
# 5. PRIOR PREDICTIVE CHECK
# ============================================================================
print("\n" + "=" * 60)
print("PRIOR PREDICTIVE CHECK")
print("=" * 60)

with bp_model:
    prior_pred = pm.sample_prior_predictive(random_seed=rng)

# Visualize prior predictive distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

prior_bp = prior_pred.prior_predictive["bp_change"].values.flatten()
axes[0].hist(prior_bp, bins=80, density=True, alpha=0.6, color="steelblue")
axes[0].set_title("Prior Predictive: BP Change Distribution")
axes[0].set_xlabel("BP Change (mmHg)")
axes[0].set_ylabel("Density")
axes[0].axvline(x=-20, color="red", linestyle="--", label="Plausible range")
axes[0].axvline(x=5, color="red", linestyle="--")
axes[0].legend()

# Check the prior on treatment effect specifically
prior_te = prior_pred.prior["treatment_effect"].values.flatten()
axes[1].hist(prior_te, bins=60, density=True, alpha=0.6, color="darkorange")
axes[1].set_title("Prior: Treatment Effect")
axes[1].set_xlabel("Treatment Effect (mmHg)")
axes[1].set_ylabel("Density")
axes[1].axvline(x=-20, color="red", linestyle="--", label="User's range [-20, +5]")
axes[1].axvline(x=5, color="red", linestyle="--")
axes[1].legend()

plt.tight_layout()
plt.savefig("prior_predictive_check.png", dpi=150, bbox_inches="tight")
plt.show()

# Assess: Are prior predictions plausible?
print(f"Prior predictive BP change: mean={prior_bp.mean():.1f}, "
      f"sd={prior_bp.std():.1f}")
print(f"Prior predictive range (2.5%-97.5%): "
      f"[{np.percentile(prior_bp, 2.5):.1f}, {np.percentile(prior_bp, 97.5):.1f}]")
print(f"Prior treatment effect: mean={prior_te.mean():.1f}, "
      f"sd={prior_te.std():.1f}")
print("Assessment: Prior predictions span a wide but plausible range of BP "
      "changes. No impossible values (e.g., changes > 100 mmHg). Proceeding.")

# ============================================================================
# 6. INFERENCE
# ============================================================================
print("\n" + "=" * 60)
print("SAMPLING (using nutpie for speed)")
print("=" * 60)

with bp_model:
    # Use nutpie for fast sampling. Don't hardcode number of chains —
    # let the sampler choose the optimal default for this platform.
    idata = pm.sample(nuts_sampler="nutpie", random_seed=rng)
    idata.extend(prior_pred)

# Save immediately after sampling — late crashes can destroy valid results
idata.to_netcdf("drug_trial_idata.nc")
print("InferenceData saved to drug_trial_idata.nc")

# ============================================================================
# 7. CONVERGENCE DIAGNOSTICS
# ============================================================================
print("\n" + "=" * 60)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 60)

# 7a. Summary table (R-hat + ESS at a glance)
summary = az.summary(idata, var_names=["mu_control", "treatment_effect", "sigma"], round_to=3)
print("\nParameter summary:")
print(summary)

# 7b. Check R-hat
rhat_ok = (summary["r_hat"] <= 1.01).all()
print(f"\nR-hat all <= 1.01: {rhat_ok}")

# 7c. Check ESS (bulk and tail) — threshold is 100 * number of chains
num_chains = idata.posterior.sizes["chain"]
ess_bulk_ok = (summary["ess_bulk"] >= 100 * num_chains).all()
ess_tail_ok = (summary["ess_tail"] >= 100 * num_chains).all()
print(f"ESS bulk all >= {100 * num_chains}: {ess_bulk_ok} (min: {summary['ess_bulk'].min():.0f})")
print(f"ESS tail all >= {100 * num_chains}: {ess_tail_ok} (min: {summary['ess_tail'].min():.0f})")

# 7d. Check divergences
n_div = idata.sample_stats["diverging"].sum().item()
print(f"Divergences: {n_div}")

if rhat_ok and ess_bulk_ok and ess_tail_ok and n_div == 0:
    print("\nAll convergence diagnostics PASSED. Safe to interpret results.")
else:
    print("\nWARNING: Some diagnostics failed. Investigate before interpreting.")

# 7e. Trace and rank plots
axes_trace = az.plot_trace(
    idata,
    var_names=["mu_control", "treatment_effect", "sigma"],
    kind="rank_vlines",
)
plt.suptitle("Trace and Rank Plots", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig("trace_rank_plots.png", dpi=150, bbox_inches="tight")
plt.show()

# 7f. Energy diagnostics
az.plot_energy(idata)
plt.savefig("energy_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================================
# 8. POSTERIOR PREDICTIVE CHECK
# ============================================================================
print("\n" + "=" * 60)
print("POSTERIOR PREDICTIVE CHECK")
print("=" * 60)

with bp_model:
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=rng))

# Save updated InferenceData with posterior predictive
idata.to_netcdf("drug_trial_idata.nc")

# Overall PPC
az.plot_ppc(idata, num_pp_samples=100)
plt.title("Posterior Predictive Check: BP Change")
plt.savefig("ppc_overall.png", dpi=150, bbox_inches="tight")
plt.show()

# PPC by group
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

pp_samples = idata.posterior_predictive["bp_change"]

# Control group
control_pp = pp_samples.sel(obs_id=np.where(group == 0)[0])
control_means = control_pp.mean(dim="obs_id")
axes[0].hist(
    control_means.values.flatten(), bins=50, density=True, alpha=0.5,
    color="steelblue", label="Posterior predictive"
)
axes[0].axvline(
    bp_change[group == 0].mean(), color="black", linewidth=2,
    label=f"Observed mean: {bp_change[group == 0].mean():.1f}"
)
axes[0].set_title("PPC: Control Group Mean BP Change")
axes[0].set_xlabel("Mean BP Change (mmHg)")
axes[0].legend()

# Treatment group
treat_pp = pp_samples.sel(obs_id=np.where(group == 1)[0])
treat_means = treat_pp.mean(dim="obs_id")
axes[1].hist(
    treat_means.values.flatten(), bins=50, density=True, alpha=0.5,
    color="darkorange", label="Posterior predictive"
)
axes[1].axvline(
    bp_change[group == 1].mean(), color="black", linewidth=2,
    label=f"Observed mean: {bp_change[group == 1].mean():.1f}"
)
axes[1].set_title("PPC: Treatment Group Mean BP Change")
axes[1].set_xlabel("Mean BP Change (mmHg)")
axes[1].legend()

plt.tight_layout()
plt.savefig("ppc_by_group.png", dpi=150, bbox_inches="tight")
plt.show()

# PPC for standard deviation (does the model capture the observed variance?)
fig, ax = plt.subplots(figsize=(6, 4))
pp_sds = pp_samples.std(dim="obs_id")
ax.hist(
    pp_sds.values.flatten(), bins=50, density=True, alpha=0.5,
    color="steelblue", label="Posterior predictive SD"
)
ax.axvline(
    bp_change.std(), color="black", linewidth=2,
    label=f"Observed SD: {bp_change.std():.1f}"
)
ax.set_title("PPC: Standard Deviation of BP Change")
ax.set_xlabel("SD (mmHg)")
ax.legend()
plt.tight_layout()
plt.savefig("ppc_sd.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================================
# 9. LOO-CV (Leave-One-Out Cross-Validation)
# ============================================================================
print("\n" + "=" * 60)
print("LOO-CV (Leave-One-Out Cross-Validation)")
print("=" * 60)

# nutpie silently ignores idata_kwargs={"log_likelihood": True}
# So we compute log-likelihood explicitly after sampling
with bp_model:
    pm.compute_log_likelihood(idata, model=bp_model)

# Save again with log-likelihood
idata.to_netcdf("drug_trial_idata.nc")

loo = az.loo(idata, pointwise=True)
print(loo)

# Pareto k diagnostic
pareto_k = loo.pareto_k.values
n_bad_k = (pareto_k > 0.7).sum()
print(f"\nObservations with Pareto k > 0.7: {n_bad_k}")
print(f"Pareto k range: [{pareto_k.min():.3f}, {pareto_k.max():.3f}]")

if n_bad_k == 0:
    print("All Pareto k values < 0.7: LOO estimates are reliable.")
else:
    bad_obs = np.where(pareto_k > 0.7)[0]
    print(f"WARNING: Problematic observations: {bad_obs}")

# Plot Pareto k values
az.plot_khat(loo)
plt.title("Pareto k Diagnostic")
plt.savefig("pareto_k_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================================
# 10. CALIBRATION CHECK (PIT)
# ============================================================================
print("\n" + "=" * 60)
print("CALIBRATION CHECK (PPC-PIT)")
print("=" * 60)

# PPC-PIT: compares posterior predictive to observed data
# A well-calibrated model produces uniform PIT values
azp.plot_ppc_pit(idata)
plt.savefig("ppc_pit.png", dpi=150, bbox_inches="tight")
plt.show()

# LOO-PIT: leave-one-out calibration (more robust, preferred when LOO is available)
azp.plot_ppc_pit(idata, loo_pit=True)
plt.savefig("loo_pit.png", dpi=150, bbox_inches="tight")
plt.show()

print("PIT interpretation guide:")
print("  - Uniform: well-calibrated")
print("  - U-shaped: underdispersed (intervals too narrow)")
print("  - Inverted-U: overdispersed (intervals too wide)")
print("  - Skewed: systematic bias in location")

# ============================================================================
# 11. RESULTS INTERPRETATION
# ============================================================================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

# Full summary
full_summary = az.summary(
    idata,
    var_names=["mu_control", "treatment_effect", "sigma", "mu_group"],
    round_to=2,
)
print("\nFull parameter summary:")
print(full_summary)

# Extract key posterior quantities using xarray (preferred over numpy)
te_posterior = idata.posterior["treatment_effect"]
te_mean = float(te_posterior.mean())
te_hdi = az.hdi(idata, var_names=["treatment_effect"], hdi_prob=0.94)
te_hdi_low = float(te_hdi["treatment_effect"].sel(hdi="lower"))
te_hdi_high = float(te_hdi["treatment_effect"].sel(hdi="higher"))

# Probability that drug lowers BP (treatment_effect < 0)
prob_negative = float((te_posterior < 0).mean())

# Probability of clinically meaningful effect (> 5 mmHg reduction)
prob_clinically_meaningful = float((te_posterior < -5).mean())

print(f"\n--- Key Finding ---")
print(f"Treatment effect (drug vs control):")
print(f"  Mean: {te_mean:.1f} mmHg")
print(f"  94% HDI: [{te_hdi_low:.1f}, {te_hdi_high:.1f}] mmHg")
print(f"  Probability drug lowers BP: {prob_negative:.1%}")
print(f"  Probability of clinically meaningful effect (>5 mmHg): "
      f"{prob_clinically_meaningful:.1%}")

# Posterior for control group mean
mc_posterior = idata.posterior["mu_control"]
mc_mean = float(mc_posterior.mean())
mc_hdi = az.hdi(idata, var_names=["mu_control"], hdi_prob=0.94)
mc_hdi_low = float(mc_hdi["mu_control"].sel(hdi="lower"))
mc_hdi_high = float(mc_hdi["mu_control"].sel(hdi="higher"))

print(f"\nControl group natural change:")
print(f"  Mean: {mc_mean:.1f} mmHg")
print(f"  94% HDI: [{mc_hdi_low:.1f}, {mc_hdi_high:.1f}] mmHg")

# Sigma
sigma_posterior = idata.posterior["sigma"]
sigma_mean = float(sigma_posterior.mean())
sigma_hdi = az.hdi(idata, var_names=["sigma"], hdi_prob=0.94)
sigma_hdi_low = float(sigma_hdi["sigma"].sel(hdi="lower"))
sigma_hdi_high = float(sigma_hdi["sigma"].sel(hdi="higher"))

print(f"\nWithin-group variability (sigma):")
print(f"  Mean: {sigma_mean:.1f} mmHg")
print(f"  94% HDI: [{sigma_hdi_low:.1f}, {sigma_hdi_high:.1f}] mmHg")

# ============================================================================
# 12. KEY VISUALIZATIONS
# ============================================================================
# Forest plot of key parameters
az.plot_forest(
    idata,
    var_names=["mu_control", "treatment_effect"],
    combined=True,
    hdi_prob=0.94,
)
plt.title("Parameter Estimates (94% HDI)")
plt.savefig("forest_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# Posterior distribution of treatment effect with reference lines
fig, ax = plt.subplots(figsize=(8, 5))
te_samples = te_posterior.values.flatten()
ax.hist(te_samples, bins=80, density=True, alpha=0.6, color="steelblue",
        label="Posterior")
ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="No effect")
ax.axvline(-5, color="orange", linestyle="--", linewidth=1.5,
           label="Clinically meaningful (-5 mmHg)")
ax.axvline(te_mean, color="black", linewidth=2, label=f"Mean: {te_mean:.1f}")
ax.fill_betweenx(
    [0, ax.get_ylim()[1] * 0.5],
    te_hdi_low, te_hdi_high,
    alpha=0.15, color="steelblue",
    label=f"94% HDI: [{te_hdi_low:.1f}, {te_hdi_high:.1f}]"
)
ax.set_xlabel("Treatment Effect (mmHg)")
ax.set_ylabel("Density")
ax.set_title("Posterior Distribution of Drug Effect on Blood Pressure")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("treatment_effect_posterior.png", dpi=150, bbox_inches="tight")
plt.show()

# Model graph
try:
    graph = pm.model_to_graphviz(bp_model)
    graph.render("model_graph", format="png", cleanup=True)
    print("Model graph saved to model_graph.png")
except Exception as e:
    print(f"Could not render model graph (graphviz may not be installed): {e}")

# Pair plot to check parameter correlations
az.plot_pair(
    idata,
    var_names=["mu_control", "treatment_effect", "sigma"],
    divergences=True,
)
plt.suptitle("Pair Plot (with divergences)", y=1.02)
plt.savefig("pair_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================================
# 13. FINAL SAVE
# ============================================================================
idata.to_netcdf("drug_trial_idata.nc")
print("\nFinal InferenceData saved to drug_trial_idata.nc")
print("Analysis complete.")
