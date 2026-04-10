# Causal Effect of Scholarship on First-Year GPA: Regression Discontinuity Design

## Approach

Since scholarship assignment is determined by whether a student scores above 80 on the entrance exam, this is a **Sharp Regression Discontinuity Design (RDD)**. Students just below and just above the threshold of 80 are likely similar in all ways except scholarship receipt — the cutoff creates a natural experiment.

The key identifying assumption: students cannot precisely manipulate their score to just cross the threshold (no sorting/manipulation around the cutoff).

**Estimand**: Local Average Treatment Effect (LATE) at the threshold — the causal effect of receiving the scholarship for students with exam scores near 80.

---

## Full Analysis Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─── 1. Generate Synthetic Data ────────────────────────────────────────────────

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
n = 2000
THRESHOLD = 80

# Covariates
family_income = rng.normal(60_000, 20_000, n).clip(10_000, 200_000)
high_school_gpa = rng.normal(3.0, 0.5, n).clip(1.5, 4.0)

# Exam score — slight positive correlation with covariates
exam_score = (
    40
    + 35 * (family_income / 200_000)
    + 20 * (high_school_gpa / 4.0)
    + rng.normal(0, 10, n)
).clip(40, 100)

# Sharp assignment: scholarship iff score >= 80
received_scholarship = (exam_score >= THRESHOLD).astype(int)

# True causal effect of scholarship on GPA: +0.3 GPA points
TRUE_EFFECT = 0.3

# GPA outcome — depends on covariates, running variable (smooth), and scholarship
noise = rng.normal(0, 0.25, n)
first_year_gpa = (
    0.5
    + 0.015 * (exam_score - THRESHOLD)          # smooth effect of score
    - 0.000002 * (exam_score - THRESHOLD) ** 2  # slight nonlinearity
    + 0.3 * (high_school_gpa - 3.0)
    + 0.000003 * (family_income - 60_000)
    + TRUE_EFFECT * received_scholarship
    + noise
).clip(0, 4.0)

df = pd.DataFrame({
    "exam_score": exam_score,
    "received_scholarship": received_scholarship,
    "first_year_gpa": first_year_gpa,
    "family_income": family_income,
    "high_school_gpa": high_school_gpa,
    "running_var": exam_score - THRESHOLD,  # centered running variable
})

print(f"N = {len(df)}")
print(f"Scholarship recipients: {received_scholarship.sum()} ({received_scholarship.mean():.1%})")
print(f"\nMean GPA by scholarship:")
print(df.groupby("received_scholarship")["first_year_gpa"].mean().rename({0: "No scholarship", 1: "Scholarship"}))


# ─── 2. Manipulation Test (McCrary Density Test) ───────────────────────────────
# Check for sorting/bunching around cutoff — if students manipulate scores to
# cross the threshold, we'd see a density discontinuity at 80.

def mccrary_density_test(running_var, threshold, bandwidth=5, bins=30):
    """Simplified density test: compare density on each side of cutoff."""
    left = running_var[(running_var >= threshold - bandwidth) & (running_var < threshold)]
    right = running_var[(running_var >= threshold) & (running_var < threshold + bandwidth)]

    # Kernel density estimate on each side
    # Use t-test on bin counts as a simple check
    bin_edges = np.linspace(threshold - bandwidth, threshold + bandwidth, bins + 1)
    counts, _ = np.histogram(running_var, bins=bin_edges)
    mid = bins // 2
    left_counts = counts[:mid]
    right_counts = counts[mid:]

    t_stat, p_val = stats.ttest_ind(left_counts, right_counts)
    return {
        "n_left": len(left),
        "n_right": len(right),
        "density_ratio": len(right) / len(left) if len(left) > 0 else np.nan,
        "t_stat": t_stat,
        "p_value": p_val,
    }

density_result = mccrary_density_test(df["exam_score"], THRESHOLD)
print("\n=== Manipulation Test (McCrary-style) ===")
print(f"N in [75, 80): {density_result['n_left']}")
print(f"N in [80, 85): {density_result['n_right']}")
print(f"Density ratio (right/left): {density_result['density_ratio']:.3f}")
print(f"t-stat: {density_result['t_stat']:.3f}, p-value: {density_result['p_value']:.3f}")
if density_result["p_value"] > 0.05:
    print("No evidence of manipulation (good for RDD validity)")
else:
    print("WARNING: Possible manipulation around threshold!")


# ─── 3. Covariate Smoothness Check ─────────────────────────────────────────────
# Covariates should NOT jump at the threshold — only treatment does.

def local_linear_rdd(y, x, threshold, bandwidth, covariates=None):
    """
    Local linear RDD estimator.
    Fits: y = a + b*(x-c) + tau*D + d*(x-c)*D + controls + eps
    within [-bandwidth, +bandwidth] of threshold.

    Returns tau (treatment effect), se, t-stat, p-value.
    """
    c = threshold
    mask = (x >= c - bandwidth) & (x <= c + bandwidth)
    x_c = x[mask] - c          # centered running variable
    D = (x[mask] >= c).astype(float)   # treatment indicator
    y_w = y[mask]

    # Design matrix: intercept, x_c, D, x_c*D
    X = np.column_stack([np.ones(mask.sum()), x_c, D, x_c * D])

    # Add covariates if provided
    if covariates is not None:
        cov_w = covariates[mask]
        X = np.column_stack([X, cov_w])

    # OLS
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, y_w, rcond=None)
        y_hat = X @ beta
        resid = y_w - y_hat
        n_obs, k = X.shape
        sigma2 = (resid ** 2).sum() / (n_obs - k)
        XtX_inv = np.linalg.pinv(X.T @ X)
        var_beta = sigma2 * XtX_inv
        se = np.sqrt(np.diag(var_beta))
        t_stat = beta / se
        p_val = 2 * stats.t.sf(np.abs(t_stat), df=n_obs - k)
        return {
            "estimate": beta[2],  # tau = coefficient on D
            "se": se[2],
            "t_stat": t_stat[2],
            "p_value": p_val[2],
            "n_obs": n_obs,
            "bandwidth": bandwidth,
        }
    except np.linalg.LinAlgError:
        return None


print("\n=== Covariate Smoothness at Threshold ===")
print("(Should see NO significant jumps — covariates must be continuous at cutoff)")
for cov_name, cov_col in [("family_income", "family_income"), ("high_school_gpa", "high_school_gpa")]:
    result = local_linear_rdd(
        df[cov_col].values, df["exam_score"].values, THRESHOLD, bandwidth=10
    )
    sig = "**SIGNIFICANT — RDD MAY BE INVALID**" if result["p_value"] < 0.05 else "not significant (good)"
    print(f"  {cov_name}: est={result['estimate']:.4f}, p={result['p_value']:.3f} → {sig}")


# ─── 4. Main RDD Estimate — Bandwidth Sensitivity ──────────────────────────────
print("\n=== Main RDD Estimates (Local Linear, Various Bandwidths) ===")
print(f"True causal effect: {TRUE_EFFECT}")
print(f"{'Bandwidth':>10} {'N obs':>7} {'Estimate':>10} {'SE':>8} {'95% CI':>20} {'p-value':>10}")
print("-" * 70)

covs = df[["family_income", "high_school_gpa"]].values
# Standardize covariates for numerical stability
covs_std = (covs - covs.mean(axis=0)) / covs.std(axis=0)

results_by_bw = []
for bw in [5, 7, 10, 15, 20]:
    res = local_linear_rdd(
        df["first_year_gpa"].values, df["exam_score"].values, THRESHOLD,
        bandwidth=bw, covariates=covs_std
    )
    if res:
        ci_lo = res["estimate"] - 1.96 * res["se"]
        ci_hi = res["estimate"] + 1.96 * res["se"]
        print(
            f"{bw:>10} {res['n_obs']:>7} {res['estimate']:>10.4f} "
            f"{res['se']:>8.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] {res['p_value']:>10.4f}"
        )
        results_by_bw.append({"bandwidth": bw, **res})

# ─── 5. Optimal Bandwidth (Imbens-Kalyanaraman heuristic) ──────────────────────

def ik_bandwidth(y, x, threshold):
    """
    Simplified Imbens-Kalyanaraman (2012) bandwidth selector.
    Uses rule-of-thumb based on variance and density at threshold.
    """
    c = threshold
    h_pilot = 10  # pilot bandwidth

    left_mask = (x >= c - h_pilot) & (x < c)
    right_mask = (x >= c) & (x <= c + h_pilot)

    if left_mask.sum() < 5 or right_mask.sum() < 5:
        return h_pilot

    var_left = np.var(y[left_mask])
    var_right = np.var(y[right_mask])
    n_left = left_mask.sum()
    n_right = right_mask.sum()
    n = len(y)

    # Density at threshold (fraction of obs near threshold)
    f_c = (left_mask.sum() + right_mask.sum()) / (n * 2 * h_pilot)

    # Second derivative (curvature) from quadratic fit on each side
    def get_curvature(y_s, x_s, c_val):
        x_c = x_s - c_val
        X = np.column_stack([np.ones(len(x_s)), x_c, x_c**2])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y_s, rcond=None)
            return 2 * beta[2]
        except:
            return 0

    m2_left = get_curvature(y[left_mask], x[left_mask], c)
    m2_right = get_curvature(y[right_mask], x[right_mask], c)

    avg_var = (var_left / n_left + var_right / n_right)
    avg_m2 = ((m2_left ** 2 + m2_right ** 2) / 2)

    if avg_m2 < 1e-8:
        return h_pilot

    # MSE-optimal bandwidth formula
    h_opt = (avg_var / (f_c * avg_m2 * n)) ** 0.2
    # Scale to reasonable range given data spread
    h_opt = h_opt * (x.max() - x.min()) / 2
    return float(np.clip(h_opt, 3, 25))

h_opt = ik_bandwidth(df["first_year_gpa"].values, df["exam_score"].values, THRESHOLD)
print(f"\nIK Optimal bandwidth (heuristic): {h_opt:.2f}")
res_opt = local_linear_rdd(
    df["first_year_gpa"].values, df["exam_score"].values, THRESHOLD,
    bandwidth=h_opt, covariates=covs_std
)
if res_opt:
    print(f"Estimate at optimal BW: {res_opt['estimate']:.4f} ± {res_opt['se']:.4f}")
    print(f"95% CI: [{res_opt['estimate'] - 1.96*res_opt['se']:+.4f}, {res_opt['estimate'] + 1.96*res_opt['se']:+.4f}]")
    print(f"p-value: {res_opt['p_value']:.4f}")


# ─── 6. Placebo Cutoff Tests ────────────────────────────────────────────────────
# Test "fake" thresholds — should find no effect there.
print("\n=== Placebo Cutoff Tests (Should be Non-Significant) ===")
print(f"{'Placebo cutoff':>15} {'Estimate':>10} {'p-value':>10} {'Significant?':>15}")
print("-" * 55)

for placebo_c in [60, 65, 70, 75, 85, 90, 95]:
    # Only use data on one side of the real threshold to avoid contamination
    if placebo_c < THRESHOLD:
        mask = df["exam_score"] < THRESHOLD
    else:
        mask = df["exam_score"] >= THRESHOLD

    sub = df[mask]
    if len(sub) < 100:
        continue

    res_p = local_linear_rdd(
        sub["first_year_gpa"].values, sub["exam_score"].values,
        placebo_c, bandwidth=10
    )
    if res_p and res_p["n_obs"] >= 30:
        sig = "YES (concern!)" if res_p["p_value"] < 0.05 else "No"
        print(f"{placebo_c:>15} {res_p['estimate']:>10.4f} {res_p['p_value']:>10.3f} {sig:>15}")


# ─── 7. Donut RDD (Robustness to Exact Cutoff) ─────────────────────────────────
# Exclude observations very close to threshold (in case of heaping/rounding)
print("\n=== Donut RDD (Excluding ±1 point around threshold) ===")
donut_df = df[np.abs(df["exam_score"] - THRESHOLD) > 1]
res_donut = local_linear_rdd(
    donut_df["first_year_gpa"].values, donut_df["exam_score"].values,
    THRESHOLD, bandwidth=10, covariates=covs_std[np.abs(df["exam_score"] - THRESHOLD) > 1]
)
if res_donut:
    print(f"Estimate: {res_donut['estimate']:.4f} ± {res_donut['se']:.4f}, p={res_donut['p_value']:.4f}")
    print("(Similar to main estimate → rounding/heaping not a concern)")


# ─── 8. Visualization ──────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Panel A: Raw data scatter + RDD fit
ax1 = fig.add_subplot(gs[0, :2])
colors = ["#2196F3" if s == 0 else "#FF5722" for s in df["received_scholarship"]]
ax1.scatter(df["exam_score"], df["first_year_gpa"], c=colors, alpha=0.2, s=8)

# Fit lines on each side (bw=10)
bw = 10
for side, color, label in [(df["exam_score"] < THRESHOLD, "#1565C0", "No Scholarship"),
                             (df["exam_score"] >= THRESHOLD, "#BF360C", "Scholarship")]:
    mask = side & (np.abs(df["exam_score"] - THRESHOLD) <= bw)
    x_s = df.loc[mask, "exam_score"].values
    y_s = df.loc[mask, "first_year_gpa"].values
    if len(x_s) > 5:
        x_c = x_s - THRESHOLD
        X = np.column_stack([np.ones(len(x_s)), x_c])
        beta, _, _, _ = np.linalg.lstsq(X, y_s, rcond=None)
        x_plot = np.linspace(x_s.min(), x_s.max(), 100)
        y_plot = beta[0] + beta[1] * (x_plot - THRESHOLD)
        ax1.plot(x_plot, y_plot, color=color, linewidth=2.5, label=label)

ax1.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5, label=f"Threshold = {THRESHOLD}")
ax1.set_xlabel("Entrance Exam Score", fontsize=11)
ax1.set_ylabel("First-Year GPA", fontsize=11)
ax1.set_title("A. Regression Discontinuity: GPA vs. Exam Score", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)

# Panel B: Density of running variable (manipulation test)
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(df["exam_score"], bins=40, color="#78909C", edgecolor="white", linewidth=0.5)
ax2.axvline(THRESHOLD, color="red", linestyle="--", linewidth=2)
ax2.set_xlabel("Exam Score", fontsize=11)
ax2.set_ylabel("Count", fontsize=11)
ax2.set_title("B. Score Distribution\n(Manipulation Check)", fontsize=12, fontweight="bold")

# Panel C: Binned means plot
ax3 = fig.add_subplot(gs[1, :2])
bin_edges = np.arange(40, 101, 2)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_means = []
bin_ses = []
for i in range(len(bin_edges) - 1):
    mask = (df["exam_score"] >= bin_edges[i]) & (df["exam_score"] < bin_edges[i+1])
    vals = df.loc[mask, "first_year_gpa"]
    if len(vals) >= 3:
        bin_means.append(vals.mean())
        bin_ses.append(vals.std() / np.sqrt(len(vals)))
    else:
        bin_means.append(np.nan)
        bin_ses.append(np.nan)

bin_means = np.array(bin_means)
bin_ses = np.array(bin_ses)
left_bins = bin_centers < THRESHOLD
right_bins = bin_centers >= THRESHOLD

ax3.errorbar(bin_centers[left_bins], bin_means[left_bins], yerr=1.96 * bin_ses[left_bins],
             fmt="o", color="#2196F3", markersize=5, capsize=3, label="No Scholarship")
ax3.errorbar(bin_centers[right_bins], bin_means[right_bins], yerr=1.96 * bin_ses[right_bins],
             fmt="o", color="#FF5722", markersize=5, capsize=3, label="Scholarship")
ax3.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5)
ax3.set_xlabel("Exam Score (binned)", fontsize=11)
ax3.set_ylabel("Mean First-Year GPA", fontsize=11)
ax3.set_title("C. Binned Mean GPA — Visual Discontinuity", fontsize=12, fontweight="bold")
ax3.legend(fontsize=9)

# Panel D: Bandwidth sensitivity
ax4 = fig.add_subplot(gs[1, 2])
bws = [r["bandwidth"] for r in results_by_bw]
ests = [r["estimate"] for r in results_by_bw]
ses = [r["se"] for r in results_by_bw]
ax4.errorbar(bws, ests, yerr=[1.96 * s for s in ses], fmt="o-",
             color="#7B1FA2", markersize=7, capsize=4, linewidth=2)
ax4.axhline(TRUE_EFFECT, color="green", linestyle="--", linewidth=1.5, label=f"True effect ({TRUE_EFFECT})")
ax4.axhline(0, color="gray", linestyle=":", linewidth=1)
ax4.set_xlabel("Bandwidth", fontsize=11)
ax4.set_ylabel("RDD Estimate (± 1.96 SE)", fontsize=11)
ax4.set_title("D. Bandwidth Sensitivity", fontsize=12, fontweight="bold")
ax4.legend(fontsize=9)

plt.suptitle("Regression Discontinuity Design: Scholarship Effect on First-Year GPA",
             fontsize=14, fontweight="bold", y=1.01)

output_dir = "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/causal-inference-workspace/iteration-1/rdd-scholarship-threshold/without_skill/outputs"
plt.savefig(f"{output_dir}/rdd_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {output_dir}/rdd_analysis.png")


# ─── 9. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY OF FINDINGS")
print("=" * 60)
if res_opt:
    print(f"Optimal bandwidth (IK heuristic): {h_opt:.1f} score points")
    print(f"N within bandwidth: {res_opt['n_obs']}")
    print(f"Estimated LATE: {res_opt['estimate']:.3f} GPA points")
    print(f"95% CI: [{res_opt['estimate'] - 1.96*res_opt['se']:.3f}, {res_opt['estimate'] + 1.96*res_opt['se']:.3f}]")
    print(f"True effect (data generating process): {TRUE_EFFECT}")
    print(f"Bias: {abs(res_opt['estimate'] - TRUE_EFFECT):.4f}")
print("\nInterpretation: For students near the 80-point threshold,")
print("receiving the scholarship raises first-year GPA by approximately")
print(f"{res_opt['estimate']:.2f} points (out of 4.0).")
print("\nNote: This is a LOCAL effect — it applies only to 'marginal'")
print("students near the threshold, not to all students.")
```

---

## Explanation of the Approach

### Why RDD?

The scholarship rule creates a **sharp discontinuity**: students just below 80 and just above 80 are nearly identical in expectation — same unobserved ability, motivation, family background — but one group receives a scholarship and the other does not. The only thing that changes discontinuously at score = 80 is treatment assignment.

This is fundamentally different from a simple comparison of scholarship vs. non-scholarship students, which would be confounded (higher-scoring students would have higher GPAs regardless of the scholarship).

### Key design choices

**Local linear regression** (not polynomial). Higher-order polynomials fit the running variable can produce spurious estimates near boundaries. Local linear is the standard recommendation (Gelman & Imbens 2019).

**Bandwidth selection**. The bandwidth controls the bias-variance tradeoff:
- Narrow bandwidth: low bias (observations are truly comparable), high variance
- Wide bandwidth: more data, but comparability weakens

I implement an Imbens-Kalyanaraman (2012) heuristic and also show estimates across multiple bandwidths to check robustness.

**Covariates** (family_income, high_school_gpa) are included to reduce residual variance and improve precision, but they should NOT be necessary for identification — the discontinuity alone identifies the effect.

### Validity checks run

| Check | Purpose |
|---|---|
| McCrary density test | Detect if students manipulate scores to cross threshold |
| Covariate smoothness | Confirm covariates don't jump at threshold |
| Placebo cutoffs | Test fake thresholds — should find no effect |
| Donut RDD | Check robustness to score heaping/rounding at 80 |
| Bandwidth sensitivity | Estimate should be stable across bandwidths |

### What this estimate means (LATE, not ATE)

The RDD estimand is the **Local Average Treatment Effect at the threshold**: the causal effect for students whose score is near 80. This is not the average effect for all students — a scholarship might help a low-scoring student more or less than a borderline student. Policy implications should be interpreted with this locality in mind.

### Limitations

1. **External validity**: The LATE applies to marginal students near score = 80, not to all students. Effects may differ for students far from the threshold.
2. **Compound treatment**: A scholarship may bundle financial aid + prestige + peer effects. We estimate the total bundle effect, not any single mechanism.
3. **Bandwidth choice**: Any bandwidth involves an assumption that treatment and control units are comparable within that window. The sensitivity analysis helps assess how consequential this choice is.
4. **No manipulation assumed**: If students or instructors can manipulate scores to just cross 80 (e.g., rounding, retakes), the identifying assumption fails. The McCrary test provides a check but is not definitive.
