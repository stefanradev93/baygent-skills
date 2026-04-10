"""
Workplace Safety Program Effect on Injury Rates
================================================

Generative story:
    Monthly injury counts at 30 factories follow a Poisson process whose rate
    depends on factory-level baseline risk, the number of employees (exposure),
    and whether the safety program has been introduced. Each factory has its own
    baseline injury rate (partial pooling via a hierarchical prior). The program
    effect is a multiplicative shift in the rate, modeled as exp(beta_program)
    on the log-rate scale. A rate ratio of 0.75 corresponds to beta ~ -0.29.

    We fit the model three times with different priors on beta_program to assess
    prior sensitivity:
        1. Informative prior: Normal(-0.29, 0.10) -- encodes the previous study
        2. Weakly informative prior: Normal(0, 0.50) -- allows wide range of effects
        3. Skeptical prior: Normal(0, 0.15) -- mildly skeptical, centered at no effect

    We then compare posteriors and run power-scaling sensitivity analysis on each.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import arviz_stats as azs
from arviz_stats import psense_summary
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# 0. Reproducible seed
# ============================================================================
RANDOM_SEED = sum(map(ord, "workplace-safety-program"))
rng = np.random.default_rng(RANDOM_SEED)

# ============================================================================
# 1. Generate synthetic data
# ============================================================================
N_FACTORIES = 30
N_MONTHS_PER_PERIOD = 24  # 2 years before & after
TRUE_PROGRAM_EFFECT = np.log(0.75)  # true rate ratio = 0.75 => log(0.75) ~ -0.29

# Factory-level attributes
employees = rng.integers(50, 500, size=N_FACTORIES)

# True factory-level baseline log-rates (per employee per month)
# These represent the underlying heterogeneity across factories
true_mu_log_rate = np.log(0.005)  # ~5 injuries per 1000 employees per month
true_sigma_factory = 0.40
true_factory_offsets = rng.normal(0, true_sigma_factory, size=N_FACTORIES)
true_log_rates = true_mu_log_rate + true_factory_offsets

# Build the full dataset: each factory observed for 48 months
factory_ids = np.repeat(np.arange(N_FACTORIES), 2 * N_MONTHS_PER_PERIOD)
month_ids = np.tile(np.arange(2 * N_MONTHS_PER_PERIOD), N_FACTORIES)
post_program = (month_ids >= N_MONTHS_PER_PERIOD).astype(int)
employees_long = np.repeat(employees, 2 * N_MONTHS_PER_PERIOD)

# Expected rate = employees * exp(factory_log_rate + beta_program * post)
log_mu = (
    true_log_rates[factory_ids]
    + TRUE_PROGRAM_EFFECT * post_program
    + np.log(employees_long)
)
injuries = rng.poisson(np.exp(log_mu))

df = pd.DataFrame({
    "factory": factory_ids,
    "month": month_ids,
    "post_program": post_program,
    "employees": employees_long,
    "injuries": injuries,
})

print(f"Dataset: {len(df)} rows, {N_FACTORIES} factories, "
      f"{2 * N_MONTHS_PER_PERIOD} months each")
print(f"Injury count range: [{injuries.min()}, {injuries.max()}]")
print(f"Mean injuries per factory-month: {injuries.mean():.2f}")
print(f"True program effect (log RR): {TRUE_PROGRAM_EFFECT:.3f}")
print(f"True rate ratio: {np.exp(TRUE_PROGRAM_EFFECT):.3f}")

# ============================================================================
# 2. Define three prior scenarios for beta_program
# ============================================================================
PRIOR_SCENARIOS = {
    "informative": {
        "mu": -0.29,
        "sigma": 0.10,
        "label": "Informative: N(-0.29, 0.10) -- previous study",
    },
    "weakly_informative": {
        "mu": 0.0,
        "sigma": 0.50,
        "label": "Weakly informative: N(0, 0.50)",
    },
    "skeptical": {
        "mu": 0.0,
        "sigma": 0.15,
        "label": "Skeptical: N(0, 0.15) -- centered at no effect",
    },
}

# ============================================================================
# 3. Coords (shared across all models)
# ============================================================================
coords = {
    "factory": np.arange(N_FACTORIES),
    "obs_id": np.arange(len(df)),
}

# ============================================================================
# 4. Build, check, fit, and diagnose each model
# ============================================================================
results = {}

for scenario_name, prior_spec in PRIOR_SCENARIOS.items():
    print(f"\n{'='*70}")
    print(f"Scenario: {prior_spec['label']}")
    print(f"{'='*70}")

    with pm.Model(coords=coords) as model:
        # --- Data containers ---
        factory_idx = pm.Data("factory_idx", df["factory"].values, dims="obs_id")
        post = pm.Data("post_program", df["post_program"].values, dims="obs_id")
        exposure = pm.Data("exposure", df["employees"].values, dims="obs_id")
        observed_injuries = pm.Data(
            "observed_injuries", df["injuries"].values, dims="obs_id"
        )

        # --- Hyperpriors for factory-level variation ---
        # Population mean log-rate (per employee per month)
        mu_factory = pm.Normal(
            "mu_factory", mu=-5.0, sigma=1.0
        )  # log(0.005) ~ -5.3; weakly informative around plausible rates

        # Between-factory SD: Gamma avoids zero, allows moderate heterogeneity
        sigma_factory = pm.Gamma(
            "sigma_factory", alpha=2, beta=5
        )  # mode ~0.2, mean 0.4 -- moderate factory variation

        # --- Factory-level random intercepts (non-centered) ---
        factory_offset_raw = pm.Normal(
            "factory_offset_raw", mu=0, sigma=1, dims="factory"
        )
        factory_offset = pm.Deterministic(
            "factory_offset",
            factory_offset_raw * sigma_factory,
            dims="factory",
        )

        # --- Program effect: THIS IS THE PRIOR WE VARY ---
        beta_program = pm.Normal(
            "beta_program",
            mu=prior_spec["mu"],
            sigma=prior_spec["sigma"],
        )  # Prior justification varies by scenario (see PRIOR_SCENARIOS)

        # --- Log-rate and likelihood ---
        log_rate = (
            mu_factory
            + factory_offset[factory_idx]
            + beta_program * post
            + pm.math.log(exposure)
        )

        pm.Poisson(
            "injuries",
            mu=pm.math.exp(log_rate),
            observed=observed_injuries,
            dims="obs_id",
        )

        # ---- Step 4: Prior predictive check ----
        print("  Running prior predictive check...")
        prior_pred = pm.sample_prior_predictive(random_seed=rng)

        prior_injuries = prior_pred.prior_predictive["injuries"].values.flatten()
        pct_above_1000 = (prior_injuries > 1000).mean() * 100
        print(f"  Prior pred: mean={np.nanmean(prior_injuries):.1f}, "
              f"median={np.nanmedian(prior_injuries):.1f}, "
              f"% > 1000 injuries/month: {pct_above_1000:.1f}%")

        # ---- Step 5: Inference ----
        print("  Sampling posterior...")
        idata = pm.sample(nuts_sampler="nutpie", random_seed=rng)
        idata.extend(prior_pred)

        # ---- Posterior predictive check ----
        print("  Sampling posterior predictive...")
        idata.extend(pm.sample_posterior_predictive(idata, random_seed=rng))

        # ---- Compute log-likelihood and log-prior (required for psense) ----
        print("  Computing log-likelihood and log-prior...")
        pm.compute_log_likelihood(idata, model=model)
        pm.compute_log_prior(idata, model=model)

        # ---- Save immediately after sampling ----
        output_path = f"idata_{scenario_name}.nc"
        idata.to_netcdf(output_path)
        print(f"  Saved InferenceData to {output_path}")

    # ---- Step 6: Diagnose convergence ----
    print("\n  --- Convergence diagnostics ---")
    has_errors = azs.diagnose(idata)
    if has_errors:
        print("  WARNING: Convergence issues detected. Investigate before interpreting.")
    else:
        print("  All convergence diagnostics passed.")

    # ---- Trace / rank plots ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    az.plot_trace(
        idata,
        var_names=["beta_program", "mu_factory", "sigma_factory"],
        kind="rank_vlines",
    )
    plt.suptitle(f"Rank plots -- {scenario_name}", y=1.02)
    plt.tight_layout()
    plt.savefig(f"trace_{scenario_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Step 7: Posterior predictive check ----
    fig, ax = plt.subplots(figsize=(10, 5))
    az.plot_ppc(idata, num_pp_samples=100, ax=ax)
    ax.set_title(f"Posterior predictive check -- {scenario_name}")
    plt.savefig(f"ppc_{scenario_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Step 8: Prior sensitivity analysis ----
    print("\n  --- Prior sensitivity (psense_summary) ---")
    assert "log_likelihood" in idata, (
        "Missing log_likelihood -- run pm.compute_log_likelihood(idata, model=model)"
    )
    assert "log_prior" in idata, (
        "Missing log_prior -- run pm.compute_log_prior(idata, model=model)"
    )

    psense = psense_summary(
        idata, var_names=["beta_program", "mu_factory", "sigma_factory"]
    )
    print(psense)
    print()

    # ---- Store results for comparison ----
    beta_post = idata.posterior["beta_program"]
    rate_ratio_samples = np.exp(beta_post.values.flatten())
    results[scenario_name] = {
        "idata": idata,
        "psense": psense,
        "beta_mean": float(beta_post.mean()),
        "beta_hdi": az.hdi(beta_post.values.flatten(), hdi_prob=0.94),
        "rate_ratio_mean": float(rate_ratio_samples.mean()),
        "rate_ratio_hdi": az.hdi(rate_ratio_samples, hdi_prob=0.94),
    }

# ============================================================================
# 5. Comparison across prior scenarios
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON: Program effect across prior scenarios")
print("=" * 70)
print(f"{'Scenario':<25} {'beta_program':<18} {'94% HDI':<25} "
      f"{'Rate Ratio':<12} {'RR 94% HDI':<25}")
print("-" * 105)

for name, res in results.items():
    beta_hdi = res["beta_hdi"]
    rr_hdi = res["rate_ratio_hdi"]
    print(
        f"{name:<25} {res['beta_mean']:>8.3f}          "
        f"[{beta_hdi[0]:.3f}, {beta_hdi[1]:.3f}]"
        f"       {res['rate_ratio_mean']:>6.3f}      "
        f"[{rr_hdi[0]:.3f}, {rr_hdi[1]:.3f}]"
    )

print(f"\nTrue value: beta = {TRUE_PROGRAM_EFFECT:.3f}, "
      f"rate ratio = {np.exp(TRUE_PROGRAM_EFFECT):.3f}")

# ============================================================================
# 6. Visual comparison of posteriors
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: beta_program posteriors
ax = axes[0]
for name, res in results.items():
    samples = res["idata"].posterior["beta_program"].values.flatten()
    ax.hist(samples, bins=60, alpha=0.4, density=True,
            label=PRIOR_SCENARIOS[name]["label"])
ax.axvline(TRUE_PROGRAM_EFFECT, color="black", linestyle="--",
           linewidth=2, label=f"True value ({TRUE_PROGRAM_EFFECT:.3f})")
ax.axvline(0, color="red", linestyle=":", linewidth=1, label="No effect")
ax.set_xlabel("beta_program (log rate ratio)")
ax.set_ylabel("Density")
ax.set_title("Posterior of program effect (log scale)")
ax.legend(fontsize=8)

# Right panel: rate ratio posteriors
ax = axes[1]
for name, res in results.items():
    samples = np.exp(res["idata"].posterior["beta_program"].values.flatten())
    ax.hist(samples, bins=60, alpha=0.4, density=True,
            label=PRIOR_SCENARIOS[name]["label"])
ax.axvline(np.exp(TRUE_PROGRAM_EFFECT), color="black", linestyle="--",
           linewidth=2, label=f"True RR ({np.exp(TRUE_PROGRAM_EFFECT):.3f})")
ax.axvline(1.0, color="red", linestyle=":", linewidth=1, label="No effect (RR=1)")
ax.set_xlabel("Rate Ratio (program vs. baseline)")
ax.set_ylabel("Density")
ax.set_title("Posterior of rate ratio")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("prior_sensitivity_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved comparison plot: prior_sensitivity_comparison.png")

# ============================================================================
# 7. Summary table of sensitivity diagnostics
# ============================================================================
print("\n" + "=" * 70)
print("PRIOR SENSITIVITY DIAGNOSTICS (psense_summary)")
print("=" * 70)
for name, res in results.items():
    print(f"\n--- {PRIOR_SCENARIOS[name]['label']} ---")
    print(res["psense"])

print("\n\nInterpretation guide:")
print("  CJS < 0.05  =>  Low sensitivity (robust)")
print("  Prior CJS > 0.05, Likelihood CJS > 0.05  =>  Prior-data conflict")
print("  Prior CJS > 0.05, Likelihood CJS < 0.05  =>  Strong prior / weak likelihood")
print("  Prior CJS < 0.05, Likelihood CJS > 0.05  =>  Likelihood-driven (usually fine)")
print("\nDone.")
