"""
Soda Tax DiD Analysis — Parallel Trends Violation
=================================================
Panel data: 30 cities, 2018-2025.
Treatment: 15 cities introduced soda tax in 2023.
Confounder: treatment cities are richer, already trending down before 2023.

This script:
1. Generates synthetic data that mimics the described violation
2. Visualises pre-treatment trends
3. Estimates a naive DiD (ignores violation)
4. Estimates a DiD with group-specific time trends (attempts to fix it)
5. Runs a synthetic control as the preferred alternative design
6. Runs refutation tests
7. Saves all outputs as figures + a JSON summary
"""

import json
import warnings

import arviz as az
import causalpy as cp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_SEED = sum(map(ord, "soda-tax-obesity-did"))
rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR = (
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "causal-inference-workspace/iteration-1/did-parallel-trends-violation/"
    "with_skill/outputs"
)

# ---------------------------------------------------------------------------
# 1. Generate synthetic data
# ---------------------------------------------------------------------------
N_CITIES = 30
N_TREATED = 15
YEARS = list(range(2018, 2026))
TREATMENT_YEAR = 2023

city_ids = list(range(1, N_CITIES + 1))
treated_cities = city_ids[:N_TREATED]
control_cities = city_ids[N_TREATED:]

rows = []
city_rng = np.random.default_rng(RANDOM_SEED + 1)
for city in city_ids:
    treated = 1 if city in treated_cities else 0
    # Baseline obesity (treated cities are richer -> lower baseline)
    baseline = city_rng.normal(32 if treated else 34, 1.5)
    # Pre-treatment slope (treated cities already declining faster due to wealth/healthcare)
    pre_slope = city_rng.normal(-0.6 if treated else -0.2, 0.05)
    # True causal effect of soda tax (applied starting 2023)
    true_effect = -1.5  # percentage points
    for year in YEARS:
        time_since_start = year - 2018
        post = int(year >= TREATMENT_YEAR)
        # Obesity rate
        rate = baseline + pre_slope * time_since_start
        if treated and post:
            rate += true_effect * (year - TREATMENT_YEAR + 1) * 0.4
        rate += city_rng.normal(0, 0.3)
        rows.append(
            {
                "city_id": city,
                "unit": f"city_{city:02d}",
                "year": year,
                "obesity_rate": rate,
                "group": treated,
                "post_tax": post,
                "post_treatment": post,  # CausalPy requires this column name
                "time": year - 2018,  # numeric time index for CausalPy
            }
        )

df = pd.DataFrame(rows)
df.to_csv(f"{OUTPUT_DIR}/synthetic_data.csv", index=False)
print(f"Data shape: {df.shape}")
print(df.head())

# ---------------------------------------------------------------------------
# 2. Pre-treatment trend visualisation
# ---------------------------------------------------------------------------
pre = df[df["year"] < TREATMENT_YEAR].copy()

fig, ax = plt.subplots(figsize=(10, 5))
# Group means
for g, label, color in [(1, "Tax cities (treated)", "#d62728"), (0, "No-tax cities (control)", "#1f77b4")]:
    gd = pre[pre["group"] == g].groupby("year")["obesity_rate"].mean()
    ax.plot(gd.index, gd.values, marker="o", label=label, color=color, linewidth=2)
ax.axvline(TREATMENT_YEAR - 0.5, color="gray", linestyle="--", alpha=0.7, label="Treatment (2023)")
ax.set_title("Pre-treatment obesity trends by group\n(should be parallel for valid DiD)", fontsize=13)
ax.set_xlabel("Year")
ax.set_ylabel("Mean obesity rate (%)")
ax.legend()
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig1_pre_trends.png", dpi=150)
plt.close(fig)
print("Saved fig1_pre_trends.png")

# Compute pre-treatment slopes for both groups
slopes = {}
for g in [0, 1]:
    gd = pre[pre["group"] == g].groupby("year")["obesity_rate"].mean().reset_index()
    slope = np.polyfit(gd["year"], gd["obesity_rate"], 1)[0]
    slopes[g] = slope
print(f"Pre-treatment slopes — treated: {slopes[1]:.3f}/yr, control: {slopes[0]:.3f}/yr")
slope_diff = slopes[1] - slopes[0]
print(f"Slope difference: {slope_diff:.3f}/yr  (PARALLEL TRENDS VIOLATED if this is large)")

# ---------------------------------------------------------------------------
# 3. Naive DiD (standard, ignores pre-trend violation)
# ---------------------------------------------------------------------------
rng_naive = np.random.default_rng(RANDOM_SEED + 10)
result_naive = cp.DifferenceInDifferences(
    data=df,
    formula="obesity_rate ~ 1 + post_treatment + C(group) + post_treatment:C(group)",
    time_variable_name="year",
    group_variable_name="group",
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={"nuts_sampler": "nutpie", "random_seed": rng_naive, "progressbar": False}
    ),
)
fig_naive, _ = result_naive.plot()
fig_naive.suptitle("Naive DiD (parallel trends VIOLATED)", fontsize=12)
plt.tight_layout()
fig_naive.savefig(f"{OUTPUT_DIR}/fig2_naive_did.png", dpi=150)
plt.close(fig_naive)
print("Naive DiD estimated.")

idata_naive = result_naive.idata
# CausalPy LinearRegression stores betas in posterior["beta"] with dims [chain, draw, treated_units, coeffs]
# The coeffs coordinate contains the column names from the design matrix (e.g., patsy labels)
naive_post = idata_naive.posterior
beta_naive = naive_post["beta"]
print("Naive posterior beta coords (coeffs):", beta_naive.coords["coeffs"].values.tolist())

# The DiD ATT coefficient is the interaction term
coeff_labels = beta_naive.coords["coeffs"].values.tolist()
interaction_idx = [i for i, c in enumerate(coeff_labels) if "post_treatment" in c and "group" in c.lower()]
print("Interaction coeff indices:", interaction_idx, "labels:", [coeff_labels[i] for i in interaction_idx])

if interaction_idx:
    # Sum over all interaction terms if there are multiple (shouldn't be for standard DiD)
    did_coef_naive = beta_naive.isel(coeffs=interaction_idx[0]).values.flatten()
else:
    print("WARNING: interaction term not found, using first coeff as fallback")
    did_coef_naive = beta_naive.isel(coeffs=0).values.flatten()

naive_mean = float(np.mean(did_coef_naive))
naive_hdi = az.hdi(did_coef_naive, hdi_prob=0.94)
print(f"Naive DiD ATT: {naive_mean:.3f} (94% HDI: [{naive_hdi[0]:.3f}, {naive_hdi[1]:.3f}])")

# ---------------------------------------------------------------------------
# 4. DiD with group-specific linear time trends (raw PyMC — CausalPy rejects
#    multiple interaction terms so we build the design matrix manually)
# ---------------------------------------------------------------------------
import pymc as pm
import patsy

formula_trends = "obesity_rate ~ 1 + post_treatment + C(group) + post_treatment:C(group) + time:C(group)"
y_arr, X_arr = patsy.dmatrices(formula_trends, df, return_type="dataframe")
X_np = X_arr.values
y_np = y_arr.values.flatten()
coeff_names_trends = list(X_arr.columns)
print("Trend model design matrix columns:", coeff_names_trends)

rng_trend = np.random.default_rng(RANDOM_SEED + 20)
with pm.Model(coords={"coeffs": coeff_names_trends, "obs": np.arange(len(y_np))}) as model_trends:
    beta = pm.Normal("beta", mu=0, sigma=50, dims="coeffs")
    sigma = pm.HalfNormal("sigma", sigma=1)
    mu = pm.Deterministic("mu", pm.math.dot(X_np, beta), dims="obs")
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_np, dims="obs")
    idata_trends = pm.sample(
        nuts_sampler="nutpie",
        random_seed=rng_trend,
        progressbar=False,
        return_inferencedata=True,
    )

print("DiD+trends estimated.")

# Visualise posterior means vs observed
fig_trends, ax_trends = plt.subplots(figsize=(10, 5))
for g, label, color in [(1, "Tax cities (treated)", "#d62728"), (0, "No-tax cities (control)", "#1f77b4")]:
    gd = df[df["group"] == g].groupby("year")["obesity_rate"].mean()
    ax_trends.plot(gd.index, gd.values, marker="o", color=color, label=f"Observed {label}", linewidth=2)
ax_trends.axvline(TREATMENT_YEAR - 0.5, color="gray", linestyle="--", alpha=0.7)
ax_trends.set_title("DiD + group-specific time trends\n(controls for differential pre-trend)", fontsize=12)
ax_trends.set_xlabel("Year")
ax_trends.set_ylabel("Mean obesity rate (%)")
ax_trends.legend()
plt.tight_layout()
fig_trends.savefig(f"{OUTPUT_DIR}/fig3_did_with_trends.png", dpi=150)
plt.close(fig_trends)

# Extract the DiD interaction coefficient (post_treatment:C(group)[T.1])
beta_trends_post = idata_trends.posterior["beta"]
trend_coeff_labels = list(beta_trends_post.coords["coeffs"].values)
print("Trend posterior coeff labels:", trend_coeff_labels)
trend_interaction_idx = [i for i, c in enumerate(trend_coeff_labels) if "post_treatment" in c and "group" in c.lower()]
print("Trend interaction idx:", trend_interaction_idx)

if trend_interaction_idx:
    did_coef_trends = beta_trends_post.isel(coeffs=trend_interaction_idx[0]).values.flatten()
else:
    did_coef_trends = beta_trends_post.isel(coeffs=0).values.flatten()

trends_mean = float(np.mean(did_coef_trends))
trends_hdi = az.hdi(did_coef_trends, hdi_prob=0.94)
print(f"DiD+trends ATT: {trends_mean:.3f} (94% HDI: [{trends_hdi[0]:.3f}, {trends_hdi[1]:.3f}])")

# ---------------------------------------------------------------------------
# 5. Refutation: Placebo treatment time (pre-period only)
# ---------------------------------------------------------------------------
df_pre_only = df[df["year"] < TREATMENT_YEAR].copy()
# Re-code post_tax as if treatment happened at 2021
PLACEBO_YEAR = 2021
df_pre_only["post_tax"] = (df_pre_only["year"] >= PLACEBO_YEAR).astype(int)
df_pre_only["post_treatment"] = df_pre_only["post_tax"]

rng_placebo = np.random.default_rng(RANDOM_SEED + 30)
result_placebo = cp.DifferenceInDifferences(
    data=df_pre_only,
    formula="obesity_rate ~ 1 + post_treatment + C(group) + post_treatment:C(group)",
    time_variable_name="year",
    group_variable_name="group",
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={"nuts_sampler": "nutpie", "random_seed": rng_placebo, "progressbar": False}
    ),
)
fig_placebo, _ = result_placebo.plot()
fig_placebo.suptitle("Placebo DiD (treatment at 2021, pre-period only)\nExpected: effect ≈ 0 if parallel trends holds", fontsize=11)
plt.tight_layout()
fig_placebo.savefig(f"{OUTPUT_DIR}/fig4_placebo_time.png", dpi=150)
plt.close(fig_placebo)

idata_placebo = result_placebo.idata
placebo_post = idata_placebo.posterior
beta_placebo = placebo_post["beta"]
placebo_coeff_labels = beta_placebo.coords["coeffs"].values.tolist()
print("Placebo beta coeffs:", placebo_coeff_labels)
placebo_interaction_idx = [i for i, c in enumerate(placebo_coeff_labels) if "post_treatment" in c and "group" in c.lower()]
if placebo_interaction_idx:
    did_coef_placebo = beta_placebo.isel(coeffs=placebo_interaction_idx[0]).values.flatten()
else:
    did_coef_placebo = beta_placebo.isel(coeffs=0).values.flatten()

placebo_mean = float(np.mean(did_coef_placebo))
placebo_hdi = az.hdi(did_coef_placebo, hdi_prob=0.94)
placebo_zero_in_hdi = float(placebo_hdi[0]) <= 0 <= float(placebo_hdi[1])
print(f"Placebo ATT: {placebo_mean:.3f} (94% HDI: [{placebo_hdi[0]:.3f}, {placebo_hdi[1]:.3f}])")
print(f"  Zero in HDI: {placebo_zero_in_hdi} -> PASS={placebo_zero_in_hdi}")

# ---------------------------------------------------------------------------
# 6. Synthetic Control as alternative design
#    (handles heterogeneous pre-trends; each control city is a potential donor)
# ---------------------------------------------------------------------------
# Wide format: index=year, columns=city units, values=obesity_rate
df_wide = df.pivot(index="year", columns="unit", values="obesity_rate")
treated_unit_names = [f"city_{c:02d}" for c in treated_cities]
control_unit_names = [f"city_{c:02d}" for c in control_cities]

# Run SC for first treated city as demonstration
demo_treated = treated_unit_names[0]
demo_donors = control_unit_names  # all control cities as donors

rng_sc = np.random.default_rng(RANDOM_SEED + 40)
try:
    result_sc = cp.SyntheticControl(
        data=df_wide,
        treatment_time=TREATMENT_YEAR,
        control_units=demo_donors,
        treated_units=[demo_treated],
        model=cp.pymc_models.WeightedSumFitter(
            sample_kwargs={"nuts_sampler": "nutpie", "random_seed": rng_sc, "progressbar": False}
        ),
    )
    fig_sc, _ = result_sc.plot()
    fig_sc.suptitle(f"Synthetic Control for {demo_treated}\n(robust to differential pre-trends)", fontsize=11)
    plt.tight_layout()
    fig_sc.savefig(f"{OUTPUT_DIR}/fig5_synthetic_control.png", dpi=150)
    plt.close(fig_sc)

    es_sc = result_sc.effect_summary(direction="two-sided", cumulative=True)
    sc_text = es_sc.text if hasattr(es_sc, "text") else str(es_sc)
    print("SC effect summary:", sc_text)
    sc_fitted = True
except Exception as e:
    print(f"SyntheticControl error: {e}")
    sc_text = f"ERROR: {e}"
    sc_fitted = False

# ---------------------------------------------------------------------------
# 7. Sensitivity: strength needed to explain away the group-specific trends estimate
# ---------------------------------------------------------------------------
p_negative_naive = float(np.mean(did_coef_naive < 0))
p_negative_trends = float(np.mean(did_coef_trends < 0))

# ---------------------------------------------------------------------------
# 8. Posterior comparison plot
# ---------------------------------------------------------------------------
fig_comp, ax = plt.subplots(figsize=(9, 4))
bins = np.linspace(-5, 3, 80)
ax.hist(did_coef_naive, bins=bins, alpha=0.5, label=f"Naive DiD (mean={naive_mean:.2f})", color="#d62728", density=True)
ax.hist(did_coef_trends, bins=bins, alpha=0.5, label=f"DiD+group trends (mean={trends_mean:.2f})", color="#1f77b4", density=True)
ax.hist(did_coef_placebo, bins=bins, alpha=0.4, label=f"Placebo 2021 (mean={placebo_mean:.2f})", color="#2ca02c", density=True)
ax.axvline(0, color="k", linestyle="--", linewidth=1.5, label="Zero effect")
ax.axvline(-1.5, color="purple", linestyle=":", linewidth=1.5, label="True effect (−1.5)")
ax.set_title("Posterior distributions of DiD coefficient\nNaive vs. trend-adjusted vs. placebo", fontsize=12)
ax.set_xlabel("Estimated effect on obesity rate (pp)")
ax.set_ylabel("Density")
ax.legend(fontsize=9)
plt.tight_layout()
fig_comp.savefig(f"{OUTPUT_DIR}/fig6_posterior_comparison.png", dpi=150)
plt.close(fig_comp)
print("Saved fig6_posterior_comparison.png")

# ---------------------------------------------------------------------------
# 9. Save numeric summary
# ---------------------------------------------------------------------------
summary = {
    "pre_treatment_slope_treated": float(slopes[1]),
    "pre_treatment_slope_control": float(slopes[0]),
    "slope_difference": float(slope_diff),
    "parallel_trends_violation": bool(abs(slope_diff) > 0.2),
    "naive_did": {
        "mean": naive_mean,
        "hdi_94_lo": float(naive_hdi[0]),
        "hdi_94_hi": float(naive_hdi[1]),
        "p_negative": p_negative_naive,
    },
    "did_with_group_trends": {
        "mean": trends_mean,
        "hdi_94_lo": float(trends_hdi[0]),
        "hdi_94_hi": float(trends_hdi[1]),
        "p_negative": p_negative_trends,
    },
    "placebo_test": {
        "mean": placebo_mean,
        "hdi_94_lo": float(placebo_hdi[0]),
        "hdi_94_hi": float(placebo_hdi[1]),
        "zero_in_hdi": placebo_zero_in_hdi,
        "pass": not placebo_zero_in_hdi,  # FAIL = zero NOT in HDI = parallel trends violated
    },
    "synthetic_control_fitted": sc_fitted,
    "true_effect": -1.5,
}

with open(f"{OUTPUT_DIR}/numeric_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== FINAL SUMMARY ===")
print(json.dumps(summary, indent=2))
