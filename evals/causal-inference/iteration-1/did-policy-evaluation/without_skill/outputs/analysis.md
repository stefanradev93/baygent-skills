# Causal Analysis: New Checkout System Effect on Monthly Revenue

## Approach: Difference-in-Differences (DiD)

This is a classic panel data setup suited for **Difference-in-Differences** estimation. The core idea: we compare the change in revenue for treated stores (before vs. after January 2024) to the change in revenue for control stores over the same period. The "double difference" removes time-invariant store-level confounders and common time trends.

### Key Assumptions

1. **Parallel trends**: In the absence of treatment, treated and control stores would have followed the same revenue trend. This is the critical identifying assumption — we can partially validate it using pre-treatment data.
2. **No spillovers (SUTVA)**: Treatment of one store doesn't affect control stores' revenue.
3. **Common support**: Treated and control stores are comparable in baseline characteristics.
4. **No anticipation**: Stores didn't change behavior before the treatment date in anticipation of it.

---

## Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
import scipy.stats as stats

# ─── 1. Generate Synthetic Data ───────────────────────────────────────────────

np.random.seed(42)

n_stores = 50
n_months = 24  # 12 pre (Jan 2023 – Dec 2023) + 12 post (Jan 2024 – Dec 2024)
treatment_month_idx = 12  # January 2024 is month index 12 (0-based)

months = pd.date_range("2023-01", periods=n_months, freq="MS")
store_ids = np.arange(1, n_stores + 1)
treatment_group = np.where(store_ids <= 25, 1, 0)

# Store-level fixed effects (baseline revenue heterogeneity)
store_fe = np.random.normal(0, 15, n_stores)  # in $000s

# Time trend (common to all stores — slight upward drift)
time_trend = np.linspace(0, 8, n_months)

# True ATT (Average Treatment effect on the Treated)
TRUE_ATT = 12.0  # $12k/month increase due to checkout system

records = []
for i, store in enumerate(store_ids):
    for t, month in enumerate(months):
        post = int(t >= treatment_month_idx)
        treated = treatment_group[i]

        # Revenue = baseline + store FE + common time trend + treatment effect + noise
        noise = np.random.normal(0, 8)
        revenue = (
            100
            + store_fe[i]
            + time_trend[t]
            + (TRUE_ATT if (treated == 1 and post == 1) else 0)
            + noise
        )

        records.append({
            "store_id": store,
            "month": month,
            "revenue": revenue,
            "group": treated,
            "post_treatment": post,
        })

df = pd.DataFrame(records)
df["month_num"] = df.groupby("store_id")["month"].transform(
    lambda x: (x - x.min()).dt.days // 30
)

print(f"Dataset shape: {df.shape}")
print(f"Stores: {df['store_id'].nunique()}, Months: {df['month'].nunique()}")
print(f"Treatment group size: {df[df['group']==1]['store_id'].nunique()} stores")
print(f"Control group size: {df[df['group']==0]['store_id'].nunique()} stores")
print(df.head(10))


# ─── 2. Exploratory Data Analysis ─────────────────────────────────────────────

monthly_avg = (
    df.groupby(["month", "group"])["revenue"]
    .mean()
    .reset_index()
    .assign(group_label=lambda x: x["group"].map({1: "Treatment", 0: "Control"}))
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Revenue trends
for label, grp in monthly_avg.groupby("group_label"):
    axes[0].plot(grp["month"], grp["revenue"], marker="o", markersize=3, label=label)
axes[0].axvline(pd.Timestamp("2024-01"), color="red", linestyle="--", label="Treatment start")
axes[0].set_title("Average Monthly Revenue by Group")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Revenue ($000s)")
axes[0].legend()
axes[0].tick_params(axis="x", rotation=45)

# Pre-treatment parallel trends check: plot difference (Treatment - Control)
pre_df = monthly_avg[monthly_avg["month"] < pd.Timestamp("2024-01")]
pre_pivot = pre_df.pivot(index="month", columns="group_label", values="revenue")
pre_diff = pre_pivot["Treatment"] - pre_pivot["Control"]
axes[1].plot(pre_diff.index, pre_diff.values, marker="s", color="purple")
axes[1].axhline(pre_diff.mean(), color="gray", linestyle=":", label=f"Mean diff = {pre_diff.mean():.2f}")
axes[1].set_title("Pre-Treatment Difference (Treatment − Control)\n[Parallel Trends Check]")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Revenue Difference ($000s)")
axes[1].legend()
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("revenue_trends.png", dpi=150)
plt.show()
print("Figure saved: revenue_trends.png")


# ─── 3. Simple 2x2 DiD Estimate ───────────────────────────────────────────────
# This is the textbook estimator: (Treat_Post - Treat_Pre) - (Control_Post - Control_Pre)

cell_means = df.groupby(["group", "post_treatment"])["revenue"].mean()
treat_pre  = cell_means.loc[(1, 0)]
treat_post = cell_means.loc[(1, 1)]
ctrl_pre   = cell_means.loc[(0, 0)]
ctrl_post  = cell_means.loc[(0, 1)]

did_simple = (treat_post - treat_pre) - (ctrl_post - ctrl_pre)
print(f"\n=== 2x2 DiD Estimate ===")
print(f"Treated  Pre:  {treat_pre:.2f}")
print(f"Treated  Post: {treat_post:.2f}  (change: {treat_post - treat_pre:.2f})")
print(f"Control  Pre:  {ctrl_pre:.2f}")
print(f"Control  Post: {ctrl_post:.2f}  (change: {ctrl_post - ctrl_pre:.2f})")
print(f"DiD estimate: {did_simple:.2f}  (true ATT = {TRUE_ATT})")


# ─── 4. OLS DiD with Two-Way Fixed Effects ────────────────────────────────────
# More rigorous: absorb store FE and time FE, cluster SEs at store level

# Encode store and month as categorical for fixed effects
df["store_fe_cat"] = df["store_id"].astype("category")
df["month_fe_cat"] = df["month"].astype("str").astype("category")

# Model: revenue ~ group*post + store FE + month FE
# The interaction term group:post_treatment is our DiD coefficient
formula = "revenue ~ group:post_treatment + C(store_id) + C(month)"
model = smf.ols(formula, data=df).fit()

# Cluster standard errors at the store level (within-store correlation)
cluster_se = cov_cluster(model, df["store_id"].values)
from statsmodels.stats.sandwich_covariance import se_cov
clustered_std_errors = se_cov(cluster_se)

# Extract the DiD coefficient
did_coef = model.params["group:post_treatment"]
did_se_clustered = clustered_std_errors[model.params.index.get_loc("group:post_treatment")]
t_stat = did_coef / did_se_clustered
p_value = 2 * stats.t.sf(abs(t_stat), df=model.df_resid)
ci_low  = did_coef - 1.96 * did_se_clustered
ci_high = did_coef + 1.96 * did_se_clustered

print(f"\n=== Two-Way Fixed Effects DiD ===")
print(f"DiD coefficient:    {did_coef:.3f}")
print(f"Clustered SE:       {did_se_clustered:.3f}")
print(f"t-statistic:        {t_stat:.3f}")
print(f"p-value:            {p_value:.4f}")
print(f"95% CI:             [{ci_low:.3f}, {ci_high:.3f}]")
print(f"True ATT:           {TRUE_ATT:.3f}")
print(f"R²:                 {model.rsquared:.4f}")


# ─── 5. Parallel Trends Test (Event Study) ────────────────────────────────────
# Create relative-time dummies (omit t=-1 as reference period)

df["rel_time"] = df["month_num"] - (treatment_month_idx - 1)  # -1 = Dec 2023 reference

# Keep only treated units for event study
treated_df = df[df["group"] == 1].copy()

# Create dummies for each relative time period (excluding reference = -1)
rel_times = sorted(df["rel_time"].unique())
rel_times_no_ref = [t for t in rel_times if t != -1]

for t in rel_times_no_ref:
    df[f"d_rel_{t}"] = ((df["group"] == 1) & (df["rel_time"] == t)).astype(int)

dummy_terms = " + ".join([f"d_rel_{t}" for t in rel_times_no_ref])
event_formula = f"revenue ~ {dummy_terms} + C(store_id) + C(month)"
event_model = smf.ols(event_formula, data=df).fit()

# Extract coefficients for event study plot
event_coefs = []
for t in rel_times_no_ref:
    coef = event_model.params.get(f"d_rel_{t}", np.nan)
    se   = event_model.bse.get(f"d_rel_{t}", np.nan)
    event_coefs.append({"rel_time": t, "coef": coef, "se": se})

event_coefs_df = pd.DataFrame(event_coefs)
# Add reference period (t=-1, coef=0)
ref_row = pd.DataFrame([{"rel_time": -1, "coef": 0.0, "se": 0.0}])
event_coefs_df = pd.concat([event_coefs_df, ref_row]).sort_values("rel_time").reset_index(drop=True)

# Plot event study
fig, ax = plt.subplots(figsize=(12, 5))
pre_mask  = event_coefs_df["rel_time"] < 0
post_mask = event_coefs_df["rel_time"] >= 0

for mask, color, label in [(pre_mask, "steelblue", "Pre-treatment"), (post_mask, "darkorange", "Post-treatment")]:
    sub = event_coefs_df[mask]
    ax.errorbar(
        sub["rel_time"], sub["coef"],
        yerr=1.96 * sub["se"],
        fmt="o", color=color, capsize=4, label=label
    )

ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(-0.5, color="red", linestyle="--", label="Treatment (Jan 2024)")
ax.set_xlabel("Months Relative to Treatment")
ax.set_ylabel("Coefficient (Revenue Effect, $000s)")
ax.set_title("Event Study: Dynamic Treatment Effects\n(Pre-period flat = parallel trends support)")
ax.legend()
plt.tight_layout()
plt.savefig("event_study.png", dpi=150)
plt.show()
print("Figure saved: event_study.png")

# Check parallel trends: pre-period coefficients should be ~0
pre_coefs = event_coefs_df[event_coefs_df["rel_time"] < -1]["coef"]
print(f"\nPre-treatment coefficients (should be ~0 if parallel trends hold):")
print(pre_coefs.describe().round(3))

pre_t_stats = event_coefs_df[event_coefs_df["rel_time"] < -1].apply(
    lambda r: abs(r["coef"]) / r["se"] if r["se"] > 0 else np.nan, axis=1
)
n_sig_pre = (pre_t_stats > 1.96).sum()
print(f"Pre-period coefficients with |t| > 1.96: {n_sig_pre} / {len(pre_t_stats)}")


# ─── 6. Robustness Check: Placebo Test ────────────────────────────────────────
# Falsification: pretend treatment happened 6 months earlier (July 2023)
# If we find an effect there, our design is suspect.

df["post_placebo"] = (df["month"] >= pd.Timestamp("2023-07")).astype(int)

# Only use pre-actual-treatment data
pre_actual = df[df["month"] < pd.Timestamp("2024-01")].copy()
placebo_formula = "revenue ~ group:post_placebo + C(store_id) + C(month)"
placebo_model = smf.ols(placebo_formula, data=pre_actual).fit()
placebo_coef = placebo_model.params["group:post_placebo"]
placebo_se   = placebo_model.bse["group:post_placebo"]
placebo_t    = placebo_coef / placebo_se
placebo_p    = 2 * stats.t.sf(abs(placebo_t), df=placebo_model.df_resid)

print(f"\n=== Placebo Test (Fake Treatment: July 2023) ===")
print(f"Placebo DiD coefficient: {placebo_coef:.3f}")
print(f"SE:                      {placebo_se:.3f}")
print(f"t-stat:                  {placebo_t:.3f}")
print(f"p-value:                 {placebo_p:.4f}")
print("Interpretation: p > 0.05 means no spurious effect at fake date — good sign.")


# ─── 7. Summary ───────────────────────────────────────────────────────────────

print(f"""
╔══════════════════════════════════════════════════════╗
║           RESULTS SUMMARY                           ║
╠══════════════════════════════════════════════════════╣
║  True ATT (ground truth):    {TRUE_ATT:.1f}k/month           ║
║  Simple 2x2 DiD:             {did_simple:.2f}k/month          ║
║  TWFE DiD (main estimate):   {did_coef:.3f}k/month          ║
║    Clustered SE:             {did_se_clustered:.3f}                   ║
║    95% CI:                   [{ci_low:.2f}, {ci_high:.2f}]     ║
║    p-value:                  {p_value:.4f}                 ║
║  Placebo test p-value:       {placebo_p:.4f}                 ║
╚══════════════════════════════════════════════════════╝
""")
```

---

## What This Analysis Does

### Step 1: Data Generation
Synthetic panel data for 50 stores × 24 months with store-level fixed effects (baseline revenue heterogeneity) and a common upward time trend. The true treatment effect (ATT) is $12k/month — we use this as ground truth to validate our estimator.

### Step 2: Simple 2×2 DiD
The textbook estimator:
```
DiD = (Treat_Post − Treat_Pre) − (Control_Post − Control_Pre)
```
This removes store-level confounders (via the first difference) and common time shocks (via the second difference).

### Step 3: Two-Way Fixed Effects (TWFE) Regression
The workhorse regression:
```
revenue_it = α_i + γ_t + β(group_i × post_t) + ε_it
```
- `α_i` = store fixed effects (absorb time-invariant store differences)
- `γ_t` = month fixed effects (absorb common shocks hitting all stores)
- `β` = the DiD coefficient = **Average Treatment Effect on the Treated (ATT)**

**Clustered standard errors** at the store level correct for within-store serial correlation (observations for the same store across months are correlated).

### Step 4: Event Study (Parallel Trends Validation)
We estimate separate treatment-interaction dummies for each relative time period. This allows us to:
1. **Test parallel trends**: Pre-treatment coefficients should be statistically indistinguishable from zero
2. **See treatment dynamics**: Does the effect appear immediately? Does it grow over time?

A flat pre-trend + a jump at t=0 is the ideal pattern.

### Step 5: Placebo Test
We pretend the treatment happened in July 2023 and test for an effect using only pre-treatment data. If no spurious effect is found (p > 0.05), this supports the validity of our design.

---

## Interpretation

The TWFE estimate should be close to the true ATT of $12k/month. A statistically significant coefficient with a non-significant placebo test and flat pre-trends provides strong evidence that the new checkout system caused the revenue increase.

**Important caveats in practice** (with real data):
- The parallel trends assumption is **untestable** — the event study only tests it in the pre-period
- With staggered adoption timings (different stores treated at different times), standard TWFE can be biased — use `did2s` or `csdid` estimators instead (Callaway & Sant'Anna, 2021)
- Spillovers (e.g., customers switching between stores) would violate SUTVA
- Heterogeneous treatment effects across stores are averaged in the ATT — explore heterogeneity if relevant
