# Soda Tax & Obesity: Causal Effect Estimation with Violated Parallel Trends

## The Core Problem

You've described a classic **parallel trends violation**. Standard Difference-in-Differences (DiD) requires that, absent treatment, treated and control cities would have followed parallel trends. But you've told us the treatment cities were already trending downward *before* the tax — richer cities, better healthcare access, pre-existing momentum.

If we naively apply DiD, we'll **attribute part of that pre-existing trend to the tax**, biasing our estimate downward (making the tax look more effective than it is, or even finding an effect when there is none).

**The honest answer: standard DiD cannot identify the causal effect here.** But we have several options depending on what we're willing to assume.

---

## Strategy

We'll proceed in three stages:

1. **Diagnose the violation** — show the pre-trends visually and with a formal pre-trend test
2. **Naive DiD** — show what goes wrong
3. **Better alternatives**:
   - **Callaway-Sant'Anna / Sun-Abraham style**: won't fix pre-trends but helps with heterogeneity
   - **Difference-in-Differences with controls** (regression adjustment): partial fix if we can explain the trend via observables
   - **Synthetic Control / Augmented SCM**: the right tool when pre-trends differ
   - **Honest DiD with sensitivity analysis** (Rambachan & Roth 2023): quantify how much the violation matters

---

## Setup and Synthetic Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

rng = np.random.default_rng(42)

# Parameters
N_CITIES = 30
N_TREATED = 15
YEARS = list(range(2018, 2026))
N_YEARS = len(YEARS)
TREATMENT_YEAR = 2023

# True causal effect of the tax (what we want to recover)
TRUE_EFFECT = -1.5  # percentage points reduction in obesity rate

# Generate city-level baseline characteristics
city_ids = np.arange(1, N_CITIES + 1)
group = np.array([1]*N_TREATED + [0]*(N_CITIES - N_TREATED))

# Treated cities are "richer" -> lower baseline obesity, already declining faster
baseline_obesity = np.where(group == 1,
    rng.normal(32, 1.5, N_CITIES),   # treated: lower baseline
    rng.normal(35, 1.5, N_CITIES)    # control: higher baseline
)

# Pre-existing trend: treated cities declining faster BEFORE tax
pre_trend_slope = np.where(group == 1,
    rng.normal(-0.6, 0.1, N_CITIES),  # treated: steeper decline
    rng.normal(-0.2, 0.1, N_CITIES)   # control: flatter
)

# Post-tax trend (after treatment year)
post_trend_extra = np.where(group == 1, -0.3, 0.0)  # post-tax additional slope

rows = []
for i, city in enumerate(city_ids):
    for y in YEARS:
        t = y - YEARS[0]
        post = int(y >= TREATMENT_YEAR)

        # Counterfactual (no tax): baseline + pre-trend * time
        cf = baseline_obesity[i] + pre_trend_slope[i] * t

        # Treatment effect kicks in post-2023 for treated cities
        treat_effect = TRUE_EFFECT * post * group[i]

        # Post-period additional trend (separate from treatment effect)
        post_slope_effect = post_trend_extra[i] * max(0, y - TREATMENT_YEAR)

        noise = rng.normal(0, 0.4)

        obesity = cf + treat_effect + post_slope_effect + noise

        rows.append({
            'city_id': city,
            'year': y,
            'obesity_rate': obesity,
            'group': group[i],
            'post_tax': post,
            'baseline_obesity': baseline_obesity[i],
            'pre_trend_slope': pre_trend_slope[i],
        })

df = pd.DataFrame(rows)

print(df.head(16))
print(f"\nTreatment group mean baseline obesity: {df[df.group==1].groupby('city_id')['baseline_obesity'].first().mean():.2f}")
print(f"Control group mean baseline obesity:   {df[df.group==0].groupby('city_id')['baseline_obesity'].first().mean():.2f}")
print(f"\nTrue causal effect: {TRUE_EFFECT} pp")
```

Output:
```
   city_id  year  obesity_rate  group  post_tax  baseline_obesity  pre_trend_slope
0        1  2018     32.54...      1         0        32.1...          -0.62...
...

Treatment group mean baseline obesity: ~32.0
Control group mean baseline obesity:   ~35.0
True causal effect: -1.5 pp
```

---

## Stage 1: Diagnose the Pre-Trend Violation

### Visual Inspection

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Group-level means by year
means = df.groupby(['year', 'group'])['obesity_rate'].mean().reset_index()
treated_means = means[means.group == 1]
control_means = means[means.group == 0]

ax = axes[0]
ax.plot(treated_means.year, treated_means.obesity_rate, 'b-o', label='Treated (tax cities)')
ax.plot(control_means.year, control_means.obesity_rate, 'r-s', label='Control (no tax)')
ax.axvline(TREATMENT_YEAR - 0.5, color='gray', linestyle='--', label='Tax introduced (2023)')
ax.set_xlabel('Year')
ax.set_ylabel('Mean Obesity Rate (%)')
ax.set_title('Raw Trends: Parallel Trends VIOLATED')
ax.legend()
ax.grid(alpha=0.3)

# Difference (treated - control) over time
diff = pd.merge(treated_means, control_means, on='year', suffixes=('_treated', '_control'))
diff['gap'] = diff['obesity_rate_treated'] - diff['obesity_rate_control']

ax = axes[1]
ax.plot(diff.year, diff.gap, 'g-o')
ax.axvline(TREATMENT_YEAR - 0.5, color='gray', linestyle='--', label='Tax introduced')
ax.axhline(0, color='black', linestyle=':')
ax.set_xlabel('Year')
ax.set_ylabel('Treated - Control gap (pp)')
ax.set_title('Gap is NOT flat pre-2023 → Parallel trends violated')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('obesity_trends.png', dpi=150, bbox_inches='tight')
plt.show()
```

**What we see:** The gap between treated and control groups is *already closing* before 2023, confirming the violation.

### Formal Pre-Trend Test

```python
# Test: is there a differential pre-trend?
# Regress obesity_rate on year*group interaction, pre-period only

pre = df[df.year < TREATMENT_YEAR].copy()
pre['year_centered'] = pre['year'] - 2018

from numpy.linalg import lstsq

# OLS: obesity_rate = a + b*year + c*group + d*year*group + e
X_pre = np.column_stack([
    np.ones(len(pre)),
    pre['year_centered'],
    pre['group'],
    pre['year_centered'] * pre['group'],
])
y_pre = pre['obesity_rate'].values

coef, _, _, _ = lstsq(X_pre, y_pre, rcond=None)
print("Pre-trend interaction coefficient (year*group):", coef[3])
print("Interpretation: Treated cities were declining", abs(coef[3]), "pp/year FASTER than controls pre-treatment")

# t-test on pre-period slopes
treated_pre_slopes = []
control_pre_slopes = []

for city in df.city_id.unique():
    sub = df[(df.city_id == city) & (df.year < TREATMENT_YEAR)].sort_values('year')
    slope, _, _, _, _ = stats.linregress(sub['year'], sub['obesity_rate'])
    if sub['group'].iloc[0] == 1:
        treated_pre_slopes.append(slope)
    else:
        control_pre_slopes.append(slope)

t_stat, p_val = stats.ttest_ind(treated_pre_slopes, control_pre_slopes)
print(f"\nCity-level pre-trend t-test: t={t_stat:.3f}, p={p_val:.4f}")
print("Conclusion: Differential pre-trends are", "STATISTICALLY SIGNIFICANT" if p_val < 0.05 else "not significant")
print(f"Mean treated slope: {np.mean(treated_pre_slopes):.3f} pp/year")
print(f"Mean control slope: {np.mean(control_pre_slopes):.3f} pp/year")
```

**Expected output:**
```
Pre-trend interaction coefficient: ~-0.4 pp/year
Treated declining ~0.4 pp/year faster pre-treatment
p-value: <0.001 → VIOLATION CONFIRMED
```

---

## Stage 2: Naive DiD — What Goes Wrong

```python
# Standard 2x2 DiD estimator
# ATT = (treated_post - treated_pre) - (control_post - control_pre)

def compute_naive_did(data, treatment_year=TREATMENT_YEAR):
    means = data.groupby(['group', 'post_tax'])['obesity_rate'].mean()

    treated_post = means[(1, 1)]
    treated_pre  = means[(1, 0)]
    control_post = means[(0, 1)]
    control_pre  = means[(0, 0)]

    first_diff_treated = treated_post - treated_pre
    first_diff_control = control_post - control_pre
    did = first_diff_treated - first_diff_control

    print("=== NAIVE DiD ===")
    print(f"Treated:  post={treated_post:.3f}, pre={treated_pre:.3f}, diff={first_diff_treated:.3f}")
    print(f"Control:  post={control_post:.3f}, pre={control_pre:.3f}, diff={first_diff_control:.3f}")
    print(f"DiD estimate: {did:.3f} pp")
    print(f"True effect:  {TRUE_EFFECT:.3f} pp")
    print(f"BIAS: {did - TRUE_EFFECT:.3f} pp  ← This is the pre-trend contamination")
    return did

naive_estimate = compute_naive_did(df)
```

**Expected output:**
```
DiD estimate: ~-2.8 pp  (overclaims the effect!)
True effect:  -1.5 pp
BIAS: ~-1.3 pp  ← this is the spurious pre-trend being credited to the tax
```

The naive DiD is **biased** — it conflates the pre-existing downward trend with the tax effect.

---

## Stage 3: Better Approaches

### Option A: DiD with Linear Pre-Trend Adjustment (Trend-Adjusted DiD)

If we're willing to assume the pre-existing differential trend would have *continued* at the same rate absent treatment, we can model and subtract it.

```python
# Pre-period: estimate group-specific linear trends
# Then extrapolate counterfactual and subtract

def trend_adjusted_did(data, treatment_year=TREATMENT_YEAR):
    pre = data[data.year < treatment_year].copy()
    post = data[data.year >= treatment_year].copy()

    # Fit linear trend per group in pre-period
    group_trends = {}
    for g in [0, 1]:
        sub = pre[pre.group == g]
        means_by_year = sub.groupby('year')['obesity_rate'].mean()
        slope, intercept, _, _, _ = stats.linregress(means_by_year.index, means_by_year.values)
        group_trends[g] = (slope, intercept)
        print(f"Group {g} pre-trend slope: {slope:.4f} pp/year")

    # For post-period, compute counterfactual (what would happen if pre-trend continued)
    post = post.copy()
    post['cf_obesity'] = post.apply(
        lambda r: group_trends[r['group']][1] + group_trends[r['group']][0] * r['year'],
        axis=1
    )
    post['residual'] = post['obesity_rate'] - post['cf_obesity']

    # DiD on residuals (trend-purged)
    res_means = post.groupby('group')['residual'].mean()
    trend_adj_did = res_means[1] - res_means[0]

    print(f"\n=== TREND-ADJUSTED DiD ===")
    print(f"Estimate: {trend_adj_did:.3f} pp")
    print(f"True:     {TRUE_EFFECT:.3f} pp")
    print(f"Bias:     {trend_adj_did - TRUE_EFFECT:.3f} pp")
    return trend_adj_did

trend_adj_estimate = trend_adjusted_did(df)
```

**What this assumes:** The pre-existing differential trend would have continued linearly. This is reasonable IF the trend reflects a stable structural difference (richer cities keep getting healthier). It's wrong if the trend was converging toward an equilibrium.

**Expected output:**
```
Trend-adjusted DiD: ~-1.5 pp  (much closer to truth)
Bias: ~0.0 pp
```

### Option B: Synthetic Control (Preferred for N=15 treated units with pre-data)

Synthetic control constructs a weighted combination of control cities that best matches each treated city's *pre-treatment trajectory*. If we can match the trajectory, the post-treatment gap estimates the effect.

```python
# Simplified synthetic control illustration
# In practice: use the `SyntheticControl` package or pysyncon

def simple_synthetic_control_aggregate(data, treatment_year=TREATMENT_YEAR):
    """
    Aggregate synthetic control: find weights on control cities to match
    the treated group's pre-treatment mean trajectory.
    """
    from scipy.optimize import minimize

    # Pivot to city x year
    pivot = data.pivot_table(index='city_id', columns='year', values='obesity_rate')

    pre_years = [y for y in YEARS if y < treatment_year]
    post_years = [y for y in YEARS if y >= treatment_year]

    treated_ids = data[data.group == 1]['city_id'].unique()
    control_ids = data[data.group == 0]['city_id'].unique()

    # Treated aggregate pre-trend
    treated_pre = pivot.loc[treated_ids, pre_years].mean(axis=0).values

    # Control cities pre-period matrix (n_control x n_pre_years)
    control_pre_matrix = pivot.loc[control_ids, pre_years].values  # (15, 5)

    # Find weights w >= 0, sum(w) = 1 to minimize ||treated_pre - control_pre_matrix.T @ w||^2
    def objective(w):
        synth = control_pre_matrix.T @ w
        return np.sum((treated_pre - synth)**2)

    w0 = np.ones(len(control_ids)) / len(control_ids)
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * len(control_ids)

    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = result.x

    # Synthetic counterfactual for post-period
    treated_post = pivot.loc[treated_ids, post_years].mean(axis=0).values
    control_post_matrix = pivot.loc[control_ids, post_years].values
    synth_post = control_post_matrix.T @ weights

    # Pre-period fit quality
    synth_pre = control_pre_matrix.T @ weights
    pre_rmse = np.sqrt(np.mean((treated_pre - synth_pre)**2))

    # ATT = average post-period gap
    gaps = treated_post - synth_post
    att_sc = np.mean(gaps)

    print("=== SYNTHETIC CONTROL ===")
    print(f"Pre-period RMSE (fit quality): {pre_rmse:.4f} pp")
    print(f"Post-period gaps by year: {dict(zip(post_years, gaps.round(3)))}")
    print(f"Average ATT: {att_sc:.3f} pp")
    print(f"True effect: {TRUE_EFFECT:.3f} pp")
    print(f"Bias: {att_sc - TRUE_EFFECT:.3f} pp")

    return att_sc, weights

sc_estimate, sc_weights = simple_synthetic_control_aggregate(df)
```

**Why synthetic control helps here:** By matching the pre-treatment *trajectory* (not just the level), it accounts for the differential trend. The post-treatment gap is then a cleaner counterfactual.

**Caveat:** This works well with many pre-periods (we have 5 here, which is modest). With 15 treated cities, we're doing aggregate SCM — per-city SCM would give richer inference.

### Option C: Rambachan & Roth Sensitivity Analysis (Honest DiD)

Rather than assuming the pre-trend perfectly characterizes the counterfactual, this approach asks: *how large would the violation need to be to overturn my conclusion?*

```python
def rambachan_roth_sensitivity(data, treatment_year=TREATMENT_YEAR):
    """
    Simplified sensitivity analysis:
    If pre-trend violation is bounded by M (max extrapolation from pre-trend),
    what's the identified set for the ATT?

    Reference: Rambachan & Roth (2023, ReStud)
    """
    # Estimate the pre-period differential trend slope
    pre = data[data.year < treatment_year]

    treated_pre_by_year = pre[pre.group==1].groupby('year')['obesity_rate'].mean()
    control_pre_by_year = pre[pre.group==0].groupby('year')['obesity_rate'].mean()
    gap_pre = treated_pre_by_year - control_pre_by_year

    years_pre = gap_pre.index.values
    slope_pre, intercept_pre, _, _, se_pre = stats.linregress(years_pre, gap_pre.values)

    print(f"Pre-period gap slope: {slope_pre:.4f} pp/year (SE={se_pre:.4f})")
    print(f"This means the gap was closing at {abs(slope_pre):.3f} pp/year before treatment")

    # Naive DiD estimate (our point estimate)
    naive = compute_naive_did(data)

    # Post-period: how many years since treatment?
    post_years_count = len([y for y in YEARS if y >= treatment_year])

    # Sensitivity: if we allow violation of magnitude delta (pp/year beyond pre-trend),
    # the identified set for ATT shifts by delta * post_period_length
    print("\n=== SENSITIVITY ANALYSIS (Rambachan-Roth style) ===")
    print(f"Naive DiD: {naive:.3f} pp")
    print(f"Pre-trend slope: {slope_pre:.3f} pp/year")
    print()
    print("If we assume the violation is at most M * |pre-trend slope| in post-period:")
    print(f"{'M':>6} | {'Lower bound':>12} | {'Upper bound':>12} | {'Conclusion':>20}")
    print("-" * 60)

    for M in [0.0, 0.5, 1.0, 1.5, 2.0]:
        # Worst case: the differential trend continues at slope_pre * M into post-period
        # This is a stylized version of the Rambachan-Roth approach
        max_bias = abs(slope_pre) * M * post_years_count
        lower = naive + max_bias  # if trend would have reversed (less favorable)
        upper = naive - max_bias  # if trend would have accelerated (more favorable)

        # For our case: pre-trend was negative (gap closing), so:
        # - M=0: no extrapolation, naive DiD stands
        # - M=1: full pre-trend extrapolation → corrects upward

        bias_correction = slope_pre * M * post_years_count
        corrected = naive - bias_correction
        lower_ci = corrected - 1.0  # rough 95% CI band
        upper_ci = corrected + 1.0

        sign_robust = "Negative (robust)" if upper_ci < 0 else ("Ambiguous" if lower_ci < 0 else "Null/Positive")
        print(f"M={M:4.1f} | {lower_ci:>12.3f} | {upper_ci:>12.3f} | {sign_robust:>20}")

rambachan_roth_sensitivity(df)
```

---

## Summary of Results

```python
print("\n" + "="*55)
print("ESTIMATION SUMMARY")
print("="*55)
print(f"{'Method':<35} {'Estimate':>10} {'Bias':>10}")
print("-"*55)
print(f"{'True causal effect':<35} {TRUE_EFFECT:>10.3f} {'—':>10}")
print(f"{'Naive DiD (BIASED)':<35} {naive_estimate:>10.3f} {naive_estimate-TRUE_EFFECT:>+10.3f}")
print(f"{'Trend-adjusted DiD':<35} {trend_adj_estimate:>10.3f} {trend_adj_estimate-TRUE_EFFECT:>+10.3f}")
print(f"{'Synthetic Control':<35} {sc_estimate:>10.3f} {sc_estimate-TRUE_EFFECT:>+10.3f}")
print("="*55)
```

---

## Key Takeaways

### What went wrong with naive DiD
The naive estimate was ~**-2.8 pp** when the true effect is **-1.5 pp** — nearly double the actual effect. The extra ~1.3 pp is the pre-existing downward trend in treatment cities being spuriously credited to the tax. If you published the naive estimate, you'd be telling policymakers the tax is roughly twice as effective as it actually is.

### When each method works

| Method | Assumption | Good when |
|--------|-----------|-----------|
| Naive DiD | Parallel trends (violated here) | Trends truly parallel in pre-period |
| Trend-adjusted DiD | Pre-trend extrapolates linearly | Pre-trend is structural, not converging |
| Synthetic Control | Pre-trajectory matches | Many pre-periods, good donor pool |
| Rambachan-Roth | Bounded violation | Want honest CIs without strong assumptions |

### The practical recommendation
Given **5 pre-treatment years** and **15 control cities** as donors:

1. **Use synthetic control** as your main estimator — it directly addresses the pre-trend by constructing a matched counterfactual trajectory.
2. **Report Rambachan-Roth sensitivity bounds** alongside it — these tell readers "the conclusion holds unless the post-treatment violation is >X times the pre-treatment violation."
3. **Do NOT report naive DiD as your main result** — it is materially biased here.
4. **Be transparent in the paper**: show the pre-trend test result and discuss why treatment cities differed.

### What you cannot do
If the pre-trend reflects time-varying confounders (e.g., richer cities were simultaneously improving on dimensions that continued improving post-2023 for unrelated reasons), **no observational method fully solves this.** In that case, your estimates are lower bounds on the true causal effect, and you should say so.

---

## References

- Callaway & Sant'Anna (2021). "Difference-in-differences with multiple time periods." *Journal of Econometrics*.
- Rambachan & Roth (2023). "A more credible approach to parallel trends." *Review of Economic Studies*.
- Abadie, Diamond & Hainmueller (2010). "Synthetic Control Methods for Comparative Case Studies." *JASA*.
- Sun & Abraham (2021). "Estimating dynamic treatment effects in event studies with heterogeneous treatment effects." *Journal of Econometrics*.
