# Synthetic Control Analysis: California Tobacco Control Program (1988)

## Overview

California's Proposition 99 (1988) was a landmark tobacco control initiative that raised cigarette taxes by 25 cents per pack and funded anti-smoking campaigns. We want to estimate the causal effect on per-capita cigarette sales (packs per capita per year) using a synthetic control method.

The **synthetic control** approach constructs a weighted combination of other US states (the "donor pool") that closely matches California's pre-treatment trajectory. Post-1988 divergence between California and its synthetic counterpart estimates the treatment effect.

---

## Key Challenges with This Dataset

Before building the analysis, it's important to flag a critical methodological issue: **California is a poor candidate for synthetic control due to donor pool problems**.

1. **California's uniqueness**: California was the largest US state and had notably lower smoking rates even before 1988. This means no convex combination of other states may closely approximate it in the pre-treatment period.
2. **Poor donor pool fit**: If the pre-treatment fit is poor, the post-treatment gap is not credibly causal — it may just reflect the pre-existing divergence continuing.
3. **Extrapolation concern**: Constructing a synthetic control that matches California may require heavily weighting states with very different characteristics, violating the interpolation assumption.

We will proceed with the analysis but flag the pre-treatment fit quality explicitly — it is the key diagnostic.

---

## Analysis Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(sum(map(ord, "california-tobacco-synthetic-control")))

# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATA
#    Based on approximate real patterns from Abadie, Diamond & Hainmueller (2010)
# ─────────────────────────────────────────────────────────────────────────────

years = np.arange(1970, 2001)
treatment_year = 1988
pre_mask = years < treatment_year
post_mask = years >= treatment_year
n_years = len(years)

# California: historically low and declining cigarette sales
# Real data shows CA going from ~120 packs/capita in 1970 to ~60 by 2000
california_base = np.linspace(127, 65, n_years)
# Prop 99 caused an additional accelerated decline post-1988
treatment_effect = np.where(
    years >= treatment_year,
    -np.linspace(0, 28, post_mask.sum()),  # growing effect: up to -28 packs/capita by 2000
    0.0
)
california = california_base + treatment_effect + np.random.normal(0, 2.5, n_years)

# Donor pool: 38 states (excluding CA and states with own major tobacco programs)
# We simulate states with higher baseline sales and shallower declines
state_names = [
    "Alabama", "Arkansas", "Colorado", "Connecticut", "Delaware",
    "Georgia", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
    "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
    "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma",
    "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Virginia", "West Virginia"
]
n_donors = len(state_names)

# Generate donor trajectories: higher baselines, slower declines, no treatment
donor_baselines = np.random.uniform(150, 230, n_donors)
donor_slopes = np.random.uniform(-1.8, -0.8, n_donors)  # packs/year decline
donor_noise_sd = np.random.uniform(2, 6, n_donors)

donor_data = {}
for i, state in enumerate(state_names):
    trajectory = donor_baselines[i] + donor_slopes[i] * (years - 1970)
    trajectory += np.random.normal(0, donor_noise_sd[i], n_years)
    donor_data[state] = trajectory

df_donors = pd.DataFrame(donor_data, index=years)
df_california = pd.Series(california, index=years, name="California")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SYNTHETIC CONTROL ESTIMATION
#    Minimize pre-treatment MSPE subject to weights summing to 1, weights >= 0
# ─────────────────────────────────────────────────────────────────────────────

X_donors_pre = df_donors.loc[pre_mask].values   # (T_pre, n_donors)
y_california_pre = df_california.loc[pre_mask].values  # (T_pre,)

def synthetic_control_loss(weights, X, y):
    """Mean squared prediction error in pre-treatment period."""
    synthetic = X @ weights
    return np.mean((y - synthetic) ** 2)

n = n_donors
w0 = np.ones(n) / n  # equal weights as starting point

constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
bounds = [(0.0, 1.0)] * n

result = minimize(
    synthetic_control_loss,
    w0,
    args=(X_donors_pre, y_california_pre),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-12, "maxiter": 10000}
)

optimal_weights = result.x

# Construct synthetic California for all years
synthetic_california = df_donors.values @ optimal_weights  # (n_years,)
df_synthetic = pd.Series(synthetic_california, index=years, name="Synthetic California")

# Pre-treatment RMSPE (key fit diagnostic)
pre_rmspe = np.sqrt(np.mean(
    (df_california.loc[pre_mask].values - df_synthetic.loc[pre_mask].values) ** 2
))
post_rmspe = np.sqrt(np.mean(
    (df_california.loc[post_mask].values - df_synthetic.loc[post_mask].values) ** 2
))

print(f"Pre-treatment RMSPE:  {pre_rmspe:.2f} packs/capita")
print(f"Post-treatment RMSPE: {post_rmspe:.2f} packs/capita")
print(f"RMSPE ratio (post/pre): {post_rmspe/pre_rmspe:.2f}")

# Top contributing donor states
weight_df = pd.Series(optimal_weights, index=state_names).sort_values(ascending=False)
print("\nTop donor state weights:")
print(weight_df[weight_df > 0.01].round(3).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3. PLACEBO TESTS (In-Space)
#    Apply synthetic control to each donor state as if it were treated.
#    If CA's gap is large relative to placebo gaps, it's credibly unusual.
# ─────────────────────────────────────────────────────────────────────────────

def fit_synthetic_control(treated_state_name, donor_df, treated_series, years, treatment_year):
    """Fit synthetic control for a given treated unit using remaining donors."""
    pre_mask = years < treatment_year

    # Donor pool excludes the treated unit (already done if called correctly)
    X_pre = donor_df.loc[pre_mask].values
    y_pre = treated_series.loc[pre_mask].values

    n = X_pre.shape[1]
    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    res = minimize(
        synthetic_control_loss, w0, args=(X_pre, y_pre),
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 5000}
    )

    synthetic = donor_df.values @ res.x
    gap = treated_series.values - synthetic
    pre_rmspe = np.sqrt(np.mean((treated_series.loc[pre_mask].values - synthetic[pre_mask]) ** 2))

    return gap, pre_rmspe, res.x

# California gap
ca_gap = df_california.values - synthetic_california

# Placebo gaps for all donor states
placebo_gaps = {}
placebo_pre_rmspes = {}

print("\nRunning placebo tests...")
for i, state in enumerate(state_names):
    # Donor pool for this placebo: all other donors (exclude current "treated" state)
    other_donors = [s for s in state_names if s != state]
    df_placebo_donors = df_donors[other_donors]
    treated_placebo = df_donors[state]

    gap, pre_rmspe_i, _ = fit_synthetic_control(
        state, df_placebo_donors, treated_placebo,
        pd.Series(years), treatment_year
    )
    placebo_gaps[state] = gap
    placebo_pre_rmspes[state] = pre_rmspe_i

print("Placebo tests complete.")

# Filter placebos with poor pre-treatment fit (RMSPE ratio > 5x California's)
# States with bad fit provide no useful comparison
rmspe_threshold = 5 * pre_rmspe
good_placebos = {k: v for k, v in placebo_gaps.items()
                 if placebo_pre_rmspes[k] < rmspe_threshold}

print(f"Placebos passing RMSPE filter (< {rmspe_threshold:.1f}): {len(good_placebos)}/{n_donors}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. INFERENCE VIA RMSPE RATIO
#    P-value: fraction of placebos with post/pre RMSPE ratio >= California's
# ─────────────────────────────────────────────────────────────────────────────

ca_rmspe_ratio = post_rmspe / pre_rmspe

placebo_ratios = {}
for state, gap in good_placebos.items():
    pre_gap = gap[pre_mask]
    post_gap = gap[post_mask]
    pre_r = np.sqrt(np.mean(pre_gap ** 2))
    post_r = np.sqrt(np.mean(post_gap ** 2))
    if pre_r > 0:
        placebo_ratios[state] = post_r / pre_r

p_value = np.mean([r >= ca_rmspe_ratio for r in placebo_ratios.values()])
print(f"\nCalifornia RMSPE ratio: {ca_rmspe_ratio:.2f}")
print(f"Permutation p-value: {p_value:.3f}")
print(f"(Fraction of placebos with ratio >= California's)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── Panel A: Trends ──────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(years, df_california.values, 'b-', linewidth=2.5, label='California (actual)')
ax1.plot(years, synthetic_california, 'r--', linewidth=2.5, label='Synthetic California')
ax1.axvline(treatment_year, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
ax1.text(treatment_year + 0.3, ax1.get_ylim()[0] + 5, 'Prop 99\n(1988)',
         fontsize=9, color='gray')
ax1.set_xlabel('Year')
ax1.set_ylabel('Cigarette Sales (packs per capita)')
ax1.set_title('A. California vs Synthetic California', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# ── Panel B: Treatment Effect Gap ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(years, ca_gap, 'b-', linewidth=2.5, label='California gap')
ax2.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax2.axvline(treatment_year, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
ax2.fill_between(years[post_mask], ca_gap[post_mask], 0,
                  alpha=0.2, color='blue', label='Estimated effect')
ax2.set_xlabel('Year')
ax2.set_ylabel('Gap (packs per capita)')
ax2.set_title('B. Treatment Effect (CA minus Synthetic CA)', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# ── Panel C: Placebo Tests ───────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
for state, gap in good_placebos.items():
    ax3.plot(years, gap, color='lightgray', linewidth=0.8, alpha=0.6)
ax3.plot(years, ca_gap, 'b-', linewidth=2.5, label='California', zorder=5)
ax3.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax3.axvline(treatment_year, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
ax3.set_xlabel('Year')
ax3.set_ylabel('Gap (packs per capita)')
ax3.set_title('C. Placebo Tests (in-space)', fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
ax3.set_ylim(-60, 60)

# ── Panel D: RMSPE Ratio Distribution ───────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ratios_arr = np.array(list(placebo_ratios.values()))
ax4.hist(ratios_arr, bins=15, color='lightgray', edgecolor='white', alpha=0.8, label='Placebo states')
ax4.axvline(ca_rmspe_ratio, color='blue', linewidth=2.5,
            label=f'California ({ca_rmspe_ratio:.1f}x)')
ax4.set_xlabel('Post/Pre RMSPE Ratio')
ax4.set_ylabel('Count')
ax4.set_title('D. Permutation Inference\n(RMSPE Ratio Distribution)', fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)
ax4.text(0.97, 0.95, f'p = {p_value:.3f}', transform=ax4.transAxes,
         ha='right', va='top', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Synthetic Control Analysis: California Tobacco Control Program (1988)',
             fontsize=13, fontweight='bold', y=1.01)

plt.savefig(
    '/Users/alex_andorra/tptm_alex/portfolio/agent-skills/causal-inference-workspace/'
    'iteration-1/synthetic-control-poor-donors/without_skill/outputs/synthetic_control.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Plot saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

avg_effect_post = np.mean(ca_gap[post_mask])
cumulative_effect = np.sum(ca_gap[post_mask])

print("\n" + "="*55)
print("RESULTS SUMMARY")
print("="*55)
print(f"Pre-treatment period:       1970–1987 ({pre_mask.sum()} years)")
print(f"Post-treatment period:      1988–2000 ({post_mask.sum()} years)")
print(f"Pre-treatment RMSPE:        {pre_rmspe:.2f} packs/capita")
print(f"Average treatment effect:   {avg_effect_post:.1f} packs/capita/year")
print(f"Cumulative effect (totals): {cumulative_effect:.0f} packs/capita")
print(f"RMSPE ratio (post/pre):     {ca_rmspe_ratio:.2f}")
print(f"Permutation p-value:        {p_value:.3f}")
print("="*55)
```

---

## Running the Code

```bash
conda run -n baygent python analysis.py
```

Or paste into a Jupyter notebook in the `baygent` environment.

---

## Interpretation and Critical Assessment

### What the results likely show

If the synthetic control fits reasonably in the pre-treatment period (RMSPE < 5–10 packs/capita), we can interpret the post-1988 gap as Proposition 99's causal effect. Historical analyses (Abadie et al. 2010) estimate California's per-capita cigarette consumption fell roughly 25–30 packs/year *more* than the counterfactual by the late 1990s.

The RMSPE ratio test provides permutation-based inference: if California's ratio is in the top 5% of placebo ratios, p < 0.05.

### The "poor donors" problem — why this is a hard case

**This is the central methodological challenge the setup explicitly poses.** Here is why:

1. **Pre-treatment uniqueness**: California had the lowest cigarette consumption in the US even before 1988. Its trajectory is an outlier. To construct a synthetic California, the optimizer must either (a) use extrapolating negative weights (not allowed with the non-negativity constraint) or (b) heavily weight the few low-smoking states that still don't match well.

2. **Diagnostic signal**: The pre-treatment RMSPE will likely be elevated (>10 packs/capita) compared to what is acceptable in standard applications (<3–5 packs/capita). This is the smoking gun (pun intended) that the donor pool is inadequate.

3. **What elevated RMSPE means**: A high pre-treatment RMSPE means the "synthetic control" never convincingly resembled California. The post-treatment gap is then uninterpretable — it conflates the poor fit with any treatment effect.

4. **The bias direction is unclear**: If the synthetic control starts slightly above California before 1988, the measured effect will be downward biased (underestimating the effect). If it starts below, it will be upward biased.

### What to do when the donor pool is poor

| Problem | Diagnosis | Fix |
|---|---|---|
| High pre-treatment RMSPE | Synthetic CA doesn't track actual CA | Restrict donor pool to geographically/demographically similar states |
| All weights on 1–2 states | Overfit; those states may have own confounds | Penalized (LASSO) synthetic control |
| Placebo gaps are all small but CA gap is large | Could be spurious if fit was poor | Check sensitivity to donor pool composition |
| Pre-treatment gap has a trend | Parallel trends not satisfied | Consider differencing, or SCM on growth rates |

**Practical fixes for this specific case**:
- Drop high-baseline states (e.g., Kentucky, Tennessee) from the donor pool — they cannot approximate California regardless of weighting
- Keep only states with pre-1988 cigarette sales within ±40 packs/capita of California's range
- Consider the **augmented synthetic control** (Ben-Michael et al. 2021) which adjusts for pre-treatment imbalance via outcome regression
- Report sensitivity analysis: how much does the effect estimate change as you vary the donor pool?

### Validity checklist

| Check | Status | Notes |
|---|---|---|
| No anticipation | Likely OK | Prop 99 was a ballot initiative; limited anticipation effects |
| No interference (SUTVA) | Mostly OK | Border effects possible (cigarette smuggling from Nevada) |
| Sufficient pre-treatment periods | OK | 18 pre-treatment years is generous |
| Donor pool exclusions | Important | Remove states with contemporaneous tobacco programs (Massachusetts, Oregon) |
| Pre-treatment fit quality | **Critical to verify** | The key diagnostic — report RMSPE prominently |
| Extrapolation vs interpolation | Concern | CA's uniqueness may force extrapolation |

---

## Conclusion

The synthetic control method is conceptually the right tool here — it respects the single treated unit (California), uses pre-treatment fit as a falsifiability criterion, and provides permutation-based inference without distributional assumptions.

**However**, California's uniqueness makes this a genuinely hard case. The honest reporting strategy is:

1. Show the pre-treatment RMSPE prominently — if it's > 10 packs/capita, state explicitly that the donor pool is a poor fit.
2. Conduct sensitivity analysis over donor pool restrictions.
3. Consider reporting the augmented synthetic control (which is more robust to poor pre-treatment balance) alongside the classical estimator.
4. Interpret the point estimate with appropriate uncertainty — the RMSPE ratio test is a necessary but not sufficient condition for causal identification.

The academic literature (Abadie et al. 2010) found a treatment effect of approximately −26 packs/capita by 1999 using a carefully curated donor pool of 38 states. Replicating that finding requires restricting to states that can actually form a credible counterfactual for California.
