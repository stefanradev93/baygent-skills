"""
Workplace Safety Program Effect on Injury Rates
================================================

Hierarchical Poisson model estimating the effect of a safety program on
monthly injury counts across 30 factories, with prior sensitivity analysis
to address whether informative priors drive the conclusions.

Generative story:
    Each factory has a baseline injury rate (injuries per employee per month).
    The safety program, introduced at month 24, multiplies that rate by a
    treatment effect (rate ratio). We observe monthly injury counts, which
    are Poisson-distributed with rate = baseline_rate * n_employees * treatment_effect.
    Factory-level variation in baseline rates is captured hierarchically.
"""

import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz_stats as azs
from arviz_stats import psense_summary
from arviz_plots import plot_psense_dist

# ---------------------------------------------------------------------------
# Reproducible seed derived from analysis name
# ---------------------------------------------------------------------------
RANDOM_SEED = sum(map(ord, "safety-program-injury-rates"))
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Step 0: Generate synthetic data
# ---------------------------------------------------------------------------
N_FACTORIES = 30
N_MONTHS_PRE = 24
N_MONTHS_POST = 24
N_MONTHS = N_MONTHS_PRE + N_MONTHS_POST

# Factory sizes (number of employees)
n_employees = rng.integers(50, 500, size=N_FACTORIES)

# True parameters
true_mu_log_rate = np.log(0.02)  # ~2 injuries per 100 employees per month
true_sigma_factory = 0.3         # factory-level variation in log-rate
true_treatment_rr = 0.72         # true rate ratio (28% reduction)

# Factory-specific baseline log-rates
true_log_rate_factory = rng.normal(true_mu_log_rate, true_sigma_factory, size=N_FACTORIES)

# Build dataset
records = []
for f in range(N_FACTORIES):
    for m in range(N_MONTHS):
        treated = int(m >= N_MONTHS_PRE)
        log_mu = true_log_rate_factory[f] + np.log(n_employees[f])
        if treated:
            log_mu += np.log(true_treatment_rr)
        count = rng.poisson(np.exp(log_mu))
        records.append({
            "factory": f,
            "month": m,
            "treated": treated,
            "n_employees": n_employees[f],
            "injuries": count,
        })

df = pd.DataFrame(records)

# Coordinates and indices
factory_idx = df["factory"].values
month_idx = df["month"].values
treated = df["treated"].values.astype(float)
log_exposure = np.log(df["n_employees"].values)
observed_injuries = df["injuries"].values

coords = {
    "factory": np.arange(N_FACTORIES),
    "obs": np.arange(len(df)),
}

print(f"Dataset: {len(df)} observations, {N_FACTORIES} factories, "
      f"{N_MONTHS_PRE} pre + {N_MONTHS_POST} post months")
print(f"Total injuries: {observed_injuries.sum()}")
print(f"Mean injuries/month/factory: {observed_injuries.mean():.2f}")

# ---------------------------------------------------------------------------
# Step 1: Formulate the generative story
# ---------------------------------------------------------------------------
# Injuries_fm ~ Poisson(lambda_fm)
# log(lambda_fm) = log(n_employees_f) + alpha_f + beta * treated_fm
# alpha_f ~ Normal(mu_alpha, sigma_alpha)     [factory baseline log-rate]
# mu_alpha ~ Normal(log(0.02), 0.5)           [grand mean baseline]
# sigma_alpha ~ Gamma(2, 5)                   [factory variation, avoiding 0]
# beta ~ Normal(log(0.75), 0.1)               [INFORMATIVE: prior study says ~25% reduction]
#
# rate_ratio = exp(beta), so beta = log(RR)
# Prior on beta: Normal(log(0.75), 0.1) => 94% of prior mass on RR in ~[0.61, 0.92]
# This encodes the strong prior belief from the previous study.

# ---------------------------------------------------------------------------
# Step 2 & 3: Specify priors and implement in PyMC
# ---------------------------------------------------------------------------
with pm.Model(coords=coords) as informative_model:
    # --- Data containers ---
    factory_data = pm.Data("factory_idx", factory_idx, dims="obs")
    treated_data = pm.Data("treated", treated, dims="obs")
    exposure_data = pm.Data("log_exposure", log_exposure, dims="obs")

    # --- Hyperpriors ---
    # Grand mean log-rate: centered on log(0.02) ~ -3.9, i.e. ~2% injury rate
    # sigma=0.5 allows rates roughly from 1% to 4% (plausible range)
    mu_alpha = pm.Normal("mu_alpha", mu=np.log(0.02), sigma=0.5)

    # Factory-level SD: Gamma(2, 5) has mean 0.4, avoids near-zero
    # (if there's no factory variation, we don't need hierarchy)
    sigma_alpha = pm.Gamma("sigma_alpha", alpha=2, beta=5)

    # --- Factory-level intercepts (non-centered to avoid funnels) ---
    alpha_offset = pm.Normal("alpha_offset", mu=0, sigma=1, dims="factory")
    alpha = pm.Deterministic("alpha", mu_alpha + alpha_offset * sigma_alpha, dims="factory")

    # --- Treatment effect (INFORMATIVE prior from previous study) ---
    # log(0.75) ~ -0.288. sigma=0.1 gives tight prior: 94% HDI on RR ~ [0.61, 0.92]
    # Justification: previous study found ~25% reduction with similar program
    beta = pm.Normal("beta", mu=np.log(0.75), sigma=0.1)

    # Derived quantity: rate ratio for interpretability
    rate_ratio = pm.Deterministic("rate_ratio", pm.math.exp(beta))

    # --- Linear predictor ---
    log_mu = exposure_data + alpha[factory_data] + beta * treated_data

    # --- Likelihood ---
    injuries = pm.Poisson("injuries", mu=pm.math.exp(log_mu), observed=observed_injuries, dims="obs")

    # --- Step 4: Prior predictive check ---
    prior_pred = pm.sample_prior_predictive(random_seed=rng)

# Visualize prior predictive
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Prior predictive distribution of total injuries per month
prior_injuries = prior_pred.prior_predictive["injuries"].values.reshape(
    prior_pred.prior_predictive.sizes["chain"],
    prior_pred.prior_predictive.sizes["draw"],
    -1
)
prior_monthly_total = prior_injuries.sum(axis=-1).flatten()
axes[0].hist(prior_monthly_total[prior_monthly_total < np.percentile(prior_monthly_total, 99)],
             bins=50, alpha=0.7, color="steelblue")
axes[0].axvline(observed_injuries.reshape(N_FACTORIES, N_MONTHS).sum(axis=0).mean(),
                color="red", linestyle="--", label="Observed mean monthly total")
axes[0].set_title("Prior Predictive: Total injuries per month")
axes[0].set_xlabel("Total injuries (all factories)")
axes[0].legend()

# Prior on rate ratio
prior_rr = np.exp(prior_pred.prior["beta"].values.flatten())
axes[1].hist(prior_rr, bins=50, alpha=0.7, color="steelblue")
axes[1].axvline(1.0, color="gray", linestyle="--", label="No effect")
axes[1].axvline(0.75, color="red", linestyle="--", label="Prior center (RR=0.75)")
axes[1].set_title("Prior on Rate Ratio (informative)")
axes[1].set_xlabel("Rate Ratio (exp(beta))")
axes[1].legend()

plt.tight_layout()
plt.savefig("prior_predictive_informative.png", dpi=150, bbox_inches="tight")
plt.close()
print("Prior predictive check saved: prior_predictive_informative.png")

# ---------------------------------------------------------------------------
# Step 5: Inference
# ---------------------------------------------------------------------------
with informative_model:
    idata = pm.sample(nuts_sampler="nutpie", random_seed=rng)
    idata.extend(prior_pred)

    # Posterior predictive check
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=rng))

    # Compute log-likelihood and log-prior (nutpie silently ignores idata_kwargs)
    pm.compute_log_likelihood(idata, model=informative_model)
    pm.compute_log_prior(idata, model=informative_model)

# Save immediately after sampling -- late crashes can destroy valid results
idata.to_netcdf("informative_model_output.nc")
print("InferenceData saved: informative_model_output.nc")

# ---------------------------------------------------------------------------
# Step 6: Diagnose convergence
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CONVERGENCE DIAGNOSTICS (informative prior model)")
print("=" * 60)

# Primary diagnostic: arviz_stats.diagnose
assert "log_likelihood" in idata, "Missing log_likelihood -- run pm.compute_log_likelihood"
assert "log_prior" in idata, "Missing log_prior -- run pm.compute_log_prior"

has_errors = azs.diagnose(idata)

# Visual diagnostics: rank plots
az.plot_trace(idata, var_names=["mu_alpha", "sigma_alpha", "beta", "rate_ratio"],
              kind="rank_vlines")
plt.tight_layout()
plt.savefig("trace_rank_informative.png", dpi=150, bbox_inches="tight")
plt.close()
print("Trace/rank plots saved: trace_rank_informative.png")

# Energy diagnostic
az.plot_energy(idata)
plt.savefig("energy_informative.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Step 7: Criticize the model
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL CRITICISM")
print("=" * 60)

# Posterior predictive check
az.plot_ppc(idata, num_pp_samples=100)
plt.savefig("ppc_informative.png", dpi=150, bbox_inches="tight")
plt.close()
print("PPC saved: ppc_informative.png")

# Calibration check (PIT)
from arviz_plots import plot_ppc_pit
plot_ppc_pit(idata)
plt.savefig("pit_informative.png", dpi=150, bbox_inches="tight")
plt.close()
print("PIT calibration plot saved: pit_informative.png")

# Parameter summary
print("\nParameter estimates (informative prior model):")
summary_table = az.summary(idata, var_names=["mu_alpha", "sigma_alpha", "beta", "rate_ratio"],
                           hdi_prob=0.94)
print(summary_table)

# ---------------------------------------------------------------------------
# Step 8: Prior sensitivity analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PRIOR SENSITIVITY ANALYSIS")
print("=" * 60)

# Focus on the key parameters: beta (treatment effect) and hyperpriors
sensitivity = psense_summary(idata, var_names=["beta", "rate_ratio", "mu_alpha", "sigma_alpha"])
print("\nPower-scaling sensitivity summary:")
print(sensitivity)

# Visual: how the posterior shifts under prior/likelihood perturbation
plot_psense_dist(idata, var_names=["beta", "rate_ratio"])
plt.savefig("psense_dist_informative.png", dpi=150, bbox_inches="tight")
plt.close()
print("Prior sensitivity distribution plot saved: psense_dist_informative.png")

# ---------------------------------------------------------------------------
# Step 8b: Formal sensitivity comparison -- refit with weakly informative prior
# ---------------------------------------------------------------------------
# To directly address the colleague's challenge, we refit the same model
# with a weakly informative prior on beta and compare posteriors.

print("\n" + "=" * 60)
print("SENSITIVITY COMPARISON: WEAKLY INFORMATIVE PRIOR")
print("=" * 60)

with pm.Model(coords=coords) as weak_model:
    # --- Data containers ---
    factory_data_w = pm.Data("factory_idx", factory_idx, dims="obs")
    treated_data_w = pm.Data("treated", treated, dims="obs")
    exposure_data_w = pm.Data("log_exposure", log_exposure, dims="obs")

    # --- Hyperpriors (same as informative model) ---
    mu_alpha_w = pm.Normal("mu_alpha", mu=np.log(0.02), sigma=0.5)
    sigma_alpha_w = pm.Gamma("sigma_alpha", alpha=2, beta=5)

    # --- Factory-level intercepts (non-centered) ---
    alpha_offset_w = pm.Normal("alpha_offset", mu=0, sigma=1, dims="factory")
    alpha_w = pm.Deterministic("alpha", mu_alpha_w + alpha_offset_w * sigma_alpha_w, dims="factory")

    # --- Treatment effect (WEAKLY INFORMATIVE prior) ---
    # Normal(0, 0.5) on log-scale: centered at no effect, but allows
    # rate ratios roughly from 0.37 to 2.72 (94% HDI)
    # This is skeptical -- no prior belief about direction or magnitude
    beta_w = pm.Normal("beta", mu=0, sigma=0.5)

    rate_ratio_w = pm.Deterministic("rate_ratio", pm.math.exp(beta_w))

    # --- Linear predictor ---
    log_mu_w = exposure_data_w + alpha_w[factory_data_w] + beta_w * treated_data_w

    # --- Likelihood ---
    injuries_w = pm.Poisson("injuries", mu=pm.math.exp(log_mu_w),
                            observed=observed_injuries, dims="obs")

    # Prior predictive
    prior_pred_w = pm.sample_prior_predictive(random_seed=rng)

    # Inference
    idata_weak = pm.sample(nuts_sampler="nutpie", random_seed=rng)
    idata_weak.extend(prior_pred_w)
    idata_weak.extend(pm.sample_posterior_predictive(idata_weak, random_seed=rng))

    pm.compute_log_likelihood(idata_weak, model=weak_model)
    pm.compute_log_prior(idata_weak, model=weak_model)

idata_weak.to_netcdf("weak_model_output.nc")
print("Weak prior model saved: weak_model_output.nc")

# Diagnose weak model
print("\nConvergence diagnostics (weak prior model):")
has_errors_w = azs.diagnose(idata_weak)

# Sensitivity check on weak model
sensitivity_weak = psense_summary(idata_weak, var_names=["beta", "rate_ratio", "mu_alpha", "sigma_alpha"])
print("\nPower-scaling sensitivity (weak prior):")
print(sensitivity_weak)

# ---------------------------------------------------------------------------
# Also fit a skeptical prior model (centered at no effect, tight)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SENSITIVITY COMPARISON: SKEPTICAL PRIOR")
print("=" * 60)

with pm.Model(coords=coords) as skeptical_model:
    factory_data_s = pm.Data("factory_idx", factory_idx, dims="obs")
    treated_data_s = pm.Data("treated", treated, dims="obs")
    exposure_data_s = pm.Data("log_exposure", log_exposure, dims="obs")

    mu_alpha_s = pm.Normal("mu_alpha", mu=np.log(0.02), sigma=0.5)
    sigma_alpha_s = pm.Gamma("sigma_alpha", alpha=2, beta=5)

    alpha_offset_s = pm.Normal("alpha_offset", mu=0, sigma=1, dims="factory")
    alpha_s = pm.Deterministic("alpha", mu_alpha_s + alpha_offset_s * sigma_alpha_s, dims="factory")

    # SKEPTICAL prior: centered at no effect, moderately tight
    # Normal(0, 0.15) on log-scale: 94% HDI on RR ~ [0.74, 1.35]
    # Skeptical toward any large effect
    beta_s = pm.Normal("beta", mu=0, sigma=0.15)

    rate_ratio_s = pm.Deterministic("rate_ratio", pm.math.exp(beta_s))

    log_mu_s = exposure_data_s + alpha_s[factory_data_s] + beta_s * treated_data_s
    injuries_s = pm.Poisson("injuries", mu=pm.math.exp(log_mu_s),
                            observed=observed_injuries, dims="obs")

    prior_pred_s = pm.sample_prior_predictive(random_seed=rng)
    idata_skeptical = pm.sample(nuts_sampler="nutpie", random_seed=rng)
    idata_skeptical.extend(prior_pred_s)
    idata_skeptical.extend(pm.sample_posterior_predictive(idata_skeptical, random_seed=rng))

    pm.compute_log_likelihood(idata_skeptical, model=skeptical_model)
    pm.compute_log_prior(idata_skeptical, model=skeptical_model)

idata_skeptical.to_netcdf("skeptical_model_output.nc")

print("\nConvergence diagnostics (skeptical prior model):")
has_errors_s = azs.diagnose(idata_skeptical)

sensitivity_skeptical = psense_summary(idata_skeptical, var_names=["beta", "rate_ratio"])
print("\nPower-scaling sensitivity (skeptical prior):")
print(sensitivity_skeptical)

# ---------------------------------------------------------------------------
# Step 8c: Visual comparison of posteriors across all three priors
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("POSTERIOR COMPARISON ACROSS PRIOR SPECIFICATIONS")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Rate ratio posteriors ---
ax = axes[0]
rr_informative = np.exp(idata.posterior["beta"].values.flatten())
rr_weak = np.exp(idata_weak.posterior["beta"].values.flatten())
rr_skeptical = np.exp(idata_skeptical.posterior["beta"].values.flatten())

ax.hist(rr_informative, bins=80, alpha=0.5, density=True, color="C0",
        label=f"Informative (N(log(0.75), 0.1))\nMean={rr_informative.mean():.3f}")
ax.hist(rr_weak, bins=80, alpha=0.5, density=True, color="C1",
        label=f"Weakly informative (N(0, 0.5))\nMean={rr_weak.mean():.3f}")
ax.hist(rr_skeptical, bins=80, alpha=0.5, density=True, color="C2",
        label=f"Skeptical (N(0, 0.15))\nMean={rr_skeptical.mean():.3f}")
ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="No effect")
ax.axvline(true_treatment_rr, color="black", linestyle=":", alpha=0.7,
           label=f"True RR = {true_treatment_rr}")
ax.set_xlabel("Rate Ratio (exp(beta))")
ax.set_ylabel("Density")
ax.set_title("Posterior Rate Ratio Under Different Priors")
ax.legend(fontsize=8)

# --- Beta (log-scale) posteriors ---
ax = axes[1]
beta_inf = idata.posterior["beta"].values.flatten()
beta_wk = idata_weak.posterior["beta"].values.flatten()
beta_sk = idata_skeptical.posterior["beta"].values.flatten()

ax.hist(beta_inf, bins=80, alpha=0.5, density=True, color="C0", label="Informative")
ax.hist(beta_wk, bins=80, alpha=0.5, density=True, color="C1", label="Weakly informative")
ax.hist(beta_sk, bins=80, alpha=0.5, density=True, color="C2", label="Skeptical")
ax.axvline(0.0, color="gray", linestyle="--", alpha=0.7, label="No effect")
ax.axvline(np.log(true_treatment_rr), color="black", linestyle=":", alpha=0.7, label="True beta")
ax.set_xlabel("beta (log rate ratio)")
ax.set_ylabel("Density")
ax.set_title("Posterior beta Under Different Priors")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("prior_sensitivity_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Prior sensitivity comparison plot saved: prior_sensitivity_comparison.png")

# ---------------------------------------------------------------------------
# Summary statistics across models
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY: RATE RATIO ESTIMATES ACROSS PRIOR SPECIFICATIONS")
print("=" * 60)

for label, trace in [("Informative", idata),
                     ("Weakly informative", idata_weak),
                     ("Skeptical", idata_skeptical)]:
    rr_samples = np.exp(trace.posterior["beta"].values.flatten())
    hdi = az.hdi(trace.posterior["beta"], hdi_prob=0.94)
    rr_hdi = np.exp(hdi["beta"].values)
    prob_reduction = (rr_samples < 1.0).mean()
    print(f"\n{label} prior:")
    print(f"  Rate ratio mean: {rr_samples.mean():.3f}")
    print(f"  Rate ratio 94% HDI: [{rr_hdi[0]:.3f}, {rr_hdi[1]:.3f}]")
    print(f"  P(program reduces injuries): {prob_reduction:.3f}")

# ---------------------------------------------------------------------------
# Step 10: Model graph for reporting
# ---------------------------------------------------------------------------
graph = pm.model_to_graphviz(informative_model)
graph.render("model_graph_informative", format="png", cleanup=True)
print("\nModel graph saved: model_graph_informative.png")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
