"""
Difference-in-Differences analysis: causal effect of new checkout system on monthly revenue.
50 stores, 2 years monthly panel data, treatment starts January 2024.

Key modeling note: a secular time trend ($200/month per store) is present in the data
generation process. The DiD formula must control for it to recover the true ATT.
We use time_since_treatment (centered at treatment date) as the trend control, which
avoids perfect collinearity between the trend variable and the post_treatment dummy.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import causalpy as cp
import pymc as pm
import arviz as az
import dowhy
import networkx as nx
import patsy
from scipy import stats

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = sum(map(ord, "checkout-did-analysis"))
rng = np.random.default_rng(RANDOM_SEED)
print(f"Random seed: {RANDOM_SEED}")

OUTPUT_DIR = (
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "causal-inference-workspace/iteration-1/did-policy-evaluation/with_skill/outputs"
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — GENERATE SYNTHETIC DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 1: Generating synthetic panel data")
print("="*70)

n_stores = 50
n_treated = 25
n_months = 24        # Jan 2023 – Dec 2024
treatment_month_idx = 12   # January 2024 = month index 12 (0-indexed)
TRUE_ATT = 8_000     # injected causal effect: +$8,000/month for treated stores

store_ids = [f"store_{i:03d}" for i in range(1, n_stores + 1)]
months = pd.date_range("2023-01", periods=n_months, freq="MS")
store_re = rng.normal(0, 15_000, n_stores)   # store-level baseline heterogeneity

rows = []
for s_idx, store in enumerate(store_ids):
    is_treated = s_idx < n_treated
    group = 1 if is_treated else 0
    for t_idx, month in enumerate(months):
        post = int(t_idx >= treatment_month_idx)
        baseline = 100_000 + store_re[s_idx] + t_idx * 200  # +$200/month secular trend
        treatment_effect = TRUE_ATT * is_treated * post
        noise = rng.normal(0, 5_000)
        revenue = baseline + treatment_effect + noise
        rows.append({
            "store_id": store,
            "unit": store,
            "month": month,
            "time": t_idx,
            # time_centered: 0 at treatment date, negative pre, positive post
            # this variable is orthogonal to post_treatment, avoiding collinearity
            "time_centered": t_idx - treatment_month_idx,
            "revenue": revenue,
            "group": group,
            "post_treatment": post,
        })

df = pd.DataFrame(rows)
print(f"Data shape: {df.shape}")
print(f"Treatment group stores: {df[df['group']==1]['store_id'].nunique()}")
print(f"Control group stores:   {df[df['group']==0]['store_id'].nunique()}")
print(f"True ATT injected:      ${TRUE_ATT:,.0f}/month")
print(df.groupby(["group", "post_treatment"])["revenue"].mean().round(0))

# Manual sanity check
T_post = df[(df.group==1) & (df.post_treatment==1)]["revenue"].mean()
T_pre  = df[(df.group==1) & (df.post_treatment==0)]["revenue"].mean()
C_post = df[(df.group==0) & (df.post_treatment==1)]["revenue"].mean()
C_pre  = df[(df.group==0) & (df.post_treatment==0)]["revenue"].mean()
raw_did = (T_post - T_pre) - (C_post - C_pre)
print(f"\nRaw (unadjusted) 2×2 DiD: ${raw_did:,.0f}")
print("This is approximately correct because the trend is parallel.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PARALLEL TRENDS VISUAL CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 2: Parallel trends visual check (pre-treatment period)")
print("="*70)

pre = df[df["time"] < treatment_month_idx].copy()
pre_mean = pre.groupby(["month", "group"])["revenue"].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
for g, label, color in [(1, "Treatment (new checkout)", "#2196F3"), (0, "Control", "#FF5722")]:
    d = pre_mean[pre_mean["group"] == g]
    ax.plot(d["month"], d["revenue"], marker="o", label=label, color=color)
ax.set_title("Pre-treatment mean revenue by group\n(parallel slopes required for DiD validity)")
ax.set_xlabel("Month")
ax.set_ylabel("Mean monthly revenue ($)")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/parallel_trends_check.png", dpi=150)
plt.close()
print("Saved: parallel_trends_check.png")

slopes = {}
for g, label in [(1, "Treatment"), (0, "Control")]:
    d = pre[pre["group"] == g].groupby("time")["revenue"].mean()
    slope, _, _, p, _ = stats.linregress(d.index, d.values)
    slopes[g] = slope
    print(f"  {label} pre-trend slope: ${slope:,.0f}/month (p={p:.3f})")
print(f"  Slope difference: ${abs(slopes[1] - slopes[0]):,.0f}/month — parallel trends plausible.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — MAIN DiD MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 3: Fitting Bayesian DiD model via CausalPy")
print("="*70)
print("""
Formula: revenue ~ 1 + time_centered + C(group) + post_treatment + post_treatment:C(group)

Design choices:
  - time_centered absorbs the secular trend. Centered at treatment date (month 12),
    so at treatment onset time_centered = 0, pre-period is negative, post is positive.
    This eliminates collinearity with post_treatment.
  - post_treatment absorbs the level shift at treatment time for ALL stores.
  - post_treatment:C(group)[T.1] is the DiD estimator = ATT.
  - C(group) absorbs baseline revenue differences between groups.
""")

result = cp.DifferenceInDifferences(
    data=df,
    formula=(
        "revenue ~ 1 + time_centered + C(group) "
        "+ post_treatment + post_treatment:C(group)"
    ),
    time_variable_name="time",
    group_variable_name="group",
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={
            "nuts_sampler": "nutpie",
            "random_seed": rng,
            "draws": 2000,
            "tune": 1000,
            "target_accept": 0.9,
        }
    ),
)

fig_did, ax_did = result.plot()
fig_did.suptitle(
    "Bayesian DiD: Causal Effect of New Checkout System\non Monthly Revenue",
    y=1.01
)
plt.tight_layout()
fig_did.savefig(f"{OUTPUT_DIR}/did_main_result.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: did_main_result.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 4: MCMC diagnostics")
print("="*70)

idata = result.idata
summary = az.summary(idata, var_names=["~mu"], filter_vars="like")
print(summary.to_string())

n_divergences = int(idata.sample_stats["diverging"].sum())
print(f"\nDivergences: {n_divergences}")

rhat_vals = az.rhat(idata)
rhat_max = max(
    float(rhat_vals[v].max())
    for v in rhat_vals.data_vars
    if rhat_vals[v].dtype in [float, "float32", "float64"]
)
print(f"Max R-hat: {rhat_max:.4f} (threshold: < 1.01)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — EFFECT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 5: Effect summary")
print("="*70)

es = result.effect_summary(direction="two-sided")
print(es.text)
print(es.table.to_string())

# Extract DiD coefficient posterior
posterior_betas = idata.posterior["beta"]
print(f"\nPosterior beta shape: {posterior_betas.shape}")
print(f"Beta dims: {posterior_betas.dims}")

formula_str = (
    "revenue ~ 1 + time_centered + C(group) "
    "+ post_treatment + post_treatment:C(group)"
)
_, X_dm = patsy.dmatrices(formula_str, df, return_type="dataframe")
coef_names = list(X_dm.columns)
print(f"Formula matrix columns: {coef_names}")

did_coef_name = "post_treatment:C(group)[T.1]"
did_coef_idx = coef_names.index(did_coef_name)
print(f"DiD coefficient: '{did_coef_name}' at index {did_coef_idx}")

did_posterior = posterior_betas.isel({"coeffs": did_coef_idx}).values.flatten()

p_positive = float((did_posterior > 0).mean())
hdi_50 = az.hdi(did_posterior, hdi_prob=0.50)
hdi_89 = az.hdi(did_posterior, hdi_prob=0.89)
hdi_95 = az.hdi(did_posterior, hdi_prob=0.95)
point_est = float(np.median(did_posterior))

print(f"\nDiD estimate (ATT): ${point_est:,.0f}/month")
print(f"  50% HDI: [${hdi_50[0]:,.0f}, ${hdi_50[1]:,.0f}]")
print(f"  89% HDI: [${hdi_89[0]:,.0f}, ${hdi_89[1]:,.0f}]")
print(f"  95% HDI: [${hdi_95[0]:,.0f}, ${hdi_95[1]:,.0f}]")
print(f"  P(effect > 0) = {p_positive:.4f}")
print(f"\nTrue injected ATT: ${TRUE_ATT:,.0f}")
print(f"Recovery: {point_est / TRUE_ATT * 100:.1f}%")

# Posterior density plot
fig_post, ax_post = plt.subplots(figsize=(9, 4))
ax_post.hist(did_posterior, bins=60, density=True, alpha=0.6, color="#2196F3", label="Posterior")
ax_post.axvline(point_est, color="#1565C0", lw=2, label=f"Median: ${point_est:,.0f}")
ax_post.axvline(TRUE_ATT, color="red", lw=2, linestyle="--", label=f"True ATT: ${TRUE_ATT:,.0f}")
ax_post.axvline(0, color="gray", lw=1, linestyle=":", label="Null (zero effect)")
for lo, hi, lab, alpha_val in [
    (hdi_50[0], hdi_50[1], "50% HDI", 0.25),
    (hdi_89[0], hdi_89[1], "89% HDI", 0.12),
]:
    ax_post.axvspan(lo, hi, alpha=alpha_val, color="#2196F3", label=lab)
ax_post.set_title(
    "Posterior distribution of DiD coefficient (ATT)\n"
    "Causal effect of new checkout system on monthly revenue"
)
ax_post.set_xlabel("Revenue effect ($/month per treated store)")
ax_post.set_ylabel("Posterior density")
ax_post.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax_post.legend(fontsize=8)
plt.tight_layout()
fig_post.savefig(f"{OUTPUT_DIR}/did_posterior_density.png", dpi=150)
plt.close()
print("Saved: did_posterior_density.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — REFUTATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 6: Refutation — Mandatory tests for DiD validity")
print("="*70)

refutation_results = {}

# ── 6a. Placebo treatment time ──────────────────────────────────────────────
print("\n--- 6a. Placebo treatment time ---")
print("Logic: use pre-treatment data only, pretend treatment happened at month 6.")
print("If parallel trends holds and there is no real pre-treatment effect, the")
print("DiD coefficient should be indistinguishable from zero.")

placebo_time_idx = 6
df_pre_only = df[df["time"] < treatment_month_idx].copy()
df_pre_only = df_pre_only.assign(
    post_treatment=(df_pre_only["time"] >= placebo_time_idx).astype(int),
    time_centered=df_pre_only["time"] - placebo_time_idx,
)

result_placebo = cp.DifferenceInDifferences(
    data=df_pre_only,
    formula=(
        "revenue ~ 1 + time_centered + C(group) "
        "+ post_treatment + post_treatment:C(group)"
    ),
    time_variable_name="time",
    group_variable_name="group",
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={
            "nuts_sampler": "nutpie",
            "random_seed": rng,
            "draws": 1000,
            "tune": 500,
        }
    ),
)

placebo_betas = result_placebo.idata.posterior["beta"]
_, X_dm_p = patsy.dmatrices(
    "revenue ~ 1 + time_centered + C(group) + post_treatment + post_treatment:C(group)",
    df_pre_only, return_type="dataframe"
)
placebo_did_idx = list(X_dm_p.columns).index("post_treatment:C(group)[T.1]")
placebo_did_posterior = placebo_betas.isel({"coeffs": placebo_did_idx}).values.flatten()

placebo_median = float(np.median(placebo_did_posterior))
placebo_hdi95 = az.hdi(placebo_did_posterior, hdi_prob=0.95)
placebo_includes_zero = bool(placebo_hdi95[0] <= 0 <= placebo_hdi95[1])

print(f"Placebo DiD estimate: ${placebo_median:,.0f}/month")
print(f"95% HDI: [${placebo_hdi95[0]:,.0f}, ${placebo_hdi95[1]:,.0f}]")
print(f"Includes zero: {placebo_includes_zero}")
placebo_result = "PASS" if placebo_includes_zero else "FAIL"
print(f"RESULT: {placebo_result}")
refutation_results["placebo_time"] = {
    "result": placebo_result,
    "estimate": placebo_median,
    "hdi_lo": float(placebo_hdi95[0]),
    "hdi_hi": float(placebo_hdi95[1]),
}

# ── 6b. DoWhy general refutations ──────────────────────────────────────────
print("\n--- 6b. DoWhy general refutations ---")

df_dowhy = df.copy()
# Interaction term: 1 only for treated stores in post-treatment period
df_dowhy["treatment"] = (df_dowhy["group"] * df_dowhy["post_treatment"]).astype(int)

dag_simple = nx.DiGraph()
dag_simple.add_edges_from([
    ("treatment", "revenue"),
    ("group", "treatment"),          # which stores got treatment
    ("post_treatment", "treatment"), # when treatment happened
    ("group", "revenue"),            # baseline group revenue differences
    ("post_treatment", "revenue"),   # time trend / post-period level shift
    ("time_centered", "revenue"),    # secular trend
])

dowhy_model = dowhy.CausalModel(
    data=df_dowhy,
    treatment="treatment",
    outcome="revenue",
    graph=dag_simple,
)
identified = dowhy_model.identify_effect(proceed_when_unidentifiable=False)
print(f"Estimand identified: {identified.estimand_type}")
estimate = dowhy_model.estimate_effect(
    identified,
    method_name="backdoor.linear_regression",
    control_value=0,
    treatment_value=1,
    confidence_intervals=False,
)
print(f"DoWhy linear regression estimate: ${estimate.value:,.0f}")

# Refutation 1: Random common cause
ref_rcc = dowhy_model.refute_estimate(
    identified, estimate, method_name="random_common_cause", random_seed=RANDOM_SEED
)
print(f"\nRandom common cause refuter:")
print(f"  Original: ${ref_rcc.estimated_effect:,.0f}   New: ${ref_rcc.new_effect:,.0f}")
rcc_diff_pct = abs(ref_rcc.new_effect - ref_rcc.estimated_effect) / abs(ref_rcc.estimated_effect) * 100
rcc_pass = rcc_diff_pct < 10
rcc_result = "PASS" if rcc_pass else "FAIL"
print(f"  Relative change: {rcc_diff_pct:.1f}%   RESULT: {rcc_result}")
refutation_results["random_common_cause"] = {"result": rcc_result, "pct_change": rcc_diff_pct}

# Refutation 2: Placebo treatment (permute treatment column)
ref_placebo_t = dowhy_model.refute_estimate(
    identified, estimate, method_name="placebo_treatment_refuter",
    placebo_type="permute", random_seed=RANDOM_SEED
)
print(f"\nPlacebo treatment refuter:")
print(f"  Original: ${ref_placebo_t.estimated_effect:,.0f}   New: ${ref_placebo_t.new_effect:,.0f}")
placebo_pct = abs(ref_placebo_t.new_effect) / abs(ref_placebo_t.estimated_effect) * 100
placebo_t_pass = placebo_pct < 10
placebo_t_result = "PASS" if placebo_t_pass else "FAIL"
print(f"  Placebo estimate as % of real: {placebo_pct:.1f}%   RESULT: {placebo_t_result}")
refutation_results["placebo_treatment"] = {"result": placebo_t_result, "pct_of_real": placebo_pct}

# Refutation 3: Data subset (80%)
ref_subset = dowhy_model.refute_estimate(
    identified, estimate, method_name="data_subset_refuter",
    subset_fraction=0.8, random_seed=RANDOM_SEED, num_simulations=10
)
print(f"\nData subset refuter (80%, 10 simulations):")
print(f"  Original: ${ref_subset.estimated_effect:,.0f}   New: ${ref_subset.new_effect:,.0f}")
subset_diff_pct = abs(ref_subset.new_effect - ref_subset.estimated_effect) / abs(ref_subset.estimated_effect) * 100
subset_pass = subset_diff_pct < 15
subset_result = "PASS" if subset_pass else "FAIL"
print(f"  Relative change: {subset_diff_pct:.1f}%   RESULT: {subset_result}")
refutation_results["data_subset"] = {"result": subset_result, "pct_change": subset_diff_pct}

# Refutation 4: Sensitivity to unobserved confounding
print(f"\nSensitivity to unobserved confounding (tipping-point analysis):")
tipping_point = None
sensitivity_rows = []
for strength in [0.05, 0.10, 0.20, 0.30, 0.50]:
    ref_conf = dowhy_model.refute_estimate(
        identified,
        estimate,
        method_name="add_unobserved_common_cause",
        confounders_effect_on_treatment="binary_flip",
        confounders_effect_on_outcome="linear",
        effect_strength_on_treatment=strength,
        effect_strength_on_outcome=strength,
        random_seed=RANDOM_SEED,
    )
    flipped = (ref_conf.new_effect * estimate.value) < 0
    flag = " [SIGN FLIP — tipping point!]" if flipped else ""
    print(f"  Strength {strength:.2f}: ${ref_conf.new_effect:,.0f}{flag}")
    sensitivity_rows.append({"strength": strength, "new_estimate": ref_conf.new_effect, "flipped": flipped})
    if tipping_point is None and flipped:
        tipping_point = strength

tipping_label = str(tipping_point) if tipping_point else "> 0.50"
print(f"\n  Tipping point (sign flip): {tipping_label}")
refutation_results["confounding_sensitivity"] = {
    "tipping_point": tipping_label,
    "rows": sensitivity_rows
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — COLLECT QUANTITATIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 7: Final quantitative summary")
print("="*70)

print(f"""
============================================================
QUANTITATIVE RESULTS SUMMARY
============================================================
PRIMARY ESTIMATE (Bayesian DiD, CausalPy)
  DiD coefficient (ATT):  ${point_est:,.0f}/month per treated store
  50% HDI:                [${hdi_50[0]:,.0f}, ${hdi_50[1]:,.0f}]
  89% HDI:                [${hdi_89[0]:,.0f}, ${hdi_89[1]:,.0f}]
  95% HDI:                [${hdi_95[0]:,.0f}, ${hdi_95[1]:,.0f}]
  P(effect > 0):          {p_positive:.4f}
  True injected ATT:      ${TRUE_ATT:,.0f}
  Recovery accuracy:      {point_est / TRUE_ATT * 100:.1f}%

CROSS-CHECK (DoWhy linear regression):
  Estimate:               ${estimate.value:,.0f}

MCMC HEALTH
  Divergences:            {n_divergences}
  Max R-hat:              {rhat_max:.4f}  (OK: < 1.01)

REFUTATION SUMMARY
  Test                    Result   Key metric
  ─────────────────────── ──────   ─────────────────────────────────────
  Placebo time            {refutation_results['placebo_time']['result']:<6}   HDI [{refutation_results['placebo_time']['hdi_lo']:,.0f}, {refutation_results['placebo_time']['hdi_hi']:,.0f}] (zero {'inside' if refutation_results['placebo_time']['result']=='PASS' else 'OUTSIDE'})
  Random common cause     {refutation_results['random_common_cause']['result']:<6}   {refutation_results['random_common_cause']['pct_change']:.1f}% change from original
  Placebo treatment       {refutation_results['placebo_treatment']['result']:<6}   {refutation_results['placebo_treatment']['pct_of_real']:.1f}% of original estimate survives
  Data subset (80%)       {refutation_results['data_subset']['result']:<6}   {refutation_results['data_subset']['pct_change']:.1f}% change from original
  Confounding tipping pt  N/A      strength needed to flip sign: {tipping_label}
============================================================
""")

print("All output files saved to:", OUTPUT_DIR)
