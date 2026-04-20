# Diagnostic Report Template

After every training + diagnostics run, generate `report.md` inside a dedicated results folder. See "Output structure" below.

## Results folder naming

All artifacts for a single analysis go into `<slug>/`, where `<slug>` is a short lowercase-hyphenated descriptor of the problem (e.g., `churn-model`, `ar2-timeseries`). Choose the most informative 1–3 word name for the specific analysis. When iterating on the same problem, append a version: `churn-model-v2/`.

```python
import os

results_dir = "<slug>"  # e.g., "churn-model" or "churn-model-v2" for iterations
os.makedirs(results_dir, exist_ok=True)
```

## Output structure

```
<slug>/
├── checkpoints/model.keras     # model checkpoint (saved by BasicWorkflow)
├── loss.png                    # training loss curve
├── recovery.png                # parameter recovery scatter
├── calibration_ecdf.png        # simulation-based calibration ECDF
├── coverage.png                # credible interval coverage
├── z_score_contraction.png     # z-score vs posterior contraction
├── metrics.csv                 # numerical diagnostics table
├── history.json                # training loss history
└── report.md                   # diagnostic report
```

## Figure naming convention

Save figures from `plot_default_diagnostics()` using these exact names:

```python
figure_names = {
    "losses": "loss.png",
    "recovery": "recovery.png",
    "calibration_ecdf": "calibration_ecdf.png",
    "coverage": "coverage.png",
    "z_score_contraction": "z_score_contraction.png",
}

figures = workflow.plot_default_diagnostics(test_data=test_data)
for key, fig in figures.items():
    fig.savefig(os.path.join(results_dir, figure_names[key]), dpi=150, bbox_inches="tight")
    plt.close(fig)
```

## Report template

Below is the template for `report.md`. Copy it as-is and fill in the `<placeholders>`. Every description paragraph is **static** — include it verbatim in every report. Every `<placeholder>` is **dynamic** — fill it in from the actual run.

If real data is available, include the optional sections at the end (marked **[IF REAL DATA]**). Otherwise, omit them entirely.

````markdown
# Amortized Inference — Diagnostic Report

## Training and Network Configuration

| Setting | Value |
|---------|-------|
| Inference network | <e.g., FlowMatching (Base)> |
| Summary network | <e.g., SetTransformer (Base, summary_dim=15)> |
| Epochs | <e.g., 500> |
| Batch size | <e.g., 32> |
| Batches per epoch | <e.g., 100 (online only)> |
| Validation data | <e.g., 300 simulations> |
| Training mode | <online / offline / disk> |
| Simulation budget | <total training simulations used> |

## Convergence

![Training loss](loss.png)

The training loss curve shows the optimization objective over epochs. A healthy curve decreases smoothly and plateaus. Key warning signs: (i) a growing gap between training and validation loss indicates overfitting; (ii) loss still visibly decreasing at the final epoch means the network could benefit from more epochs; (iii) NaN spikes indicate numerical instability, often caused by extreme simulator outputs or missing standardization.

**Assessment:** <1–3 sentences based on inspect_history() results — state whether training converged, whether overfitting was detected, and whether more epochs might help.>

## Parameter Recovery

![Parameter recovery](recovery.png)

Each panel plots the posterior median (point estimate) against the true parameter value across held-out simulations. Points falling on the diagonal indicate perfect recovery. Vertical bars represent 95% posterior credible intervals — their width reflects estimation uncertainty. Systematic deviations from the diagonal reveal bias; wide intervals indicate the data are only weakly informative for that parameter.

**Assessment:** <1–3 sentences based on check_diagnostics() recovery ratings — state which parameters show excellent or good recovery and which show fair or poor recovery. Mention any visible bias or wide intervals from the plot.>

## Calibration and Coverage

![Calibration ECDF](calibration_ecdf.png)

**Calibration ECDF** — Simulation-based calibration (SBC) plots show the empirical CDF of posterior ranks. Well-calibrated posteriors produce ECDFs close to the uniform diagonal. Lines consistently above the diagonal indicate overconfident (too narrow) posteriors; lines below indicate underconfident (too wide) posteriors.

![Coverage](coverage.png)

**Coverage** — Shows the fraction of held-out true values falling within nominal credible intervals (e.g., 50%, 80%, 95%). Well-calibrated models yield empirical coverage matching the nominal level. Under-coverage means the credible intervals are too narrow; over-coverage means they are too wide.

**Assessment:** <1–3 sentences based on check_diagnostics() calibration ratings — state which parameters have excellent calibration and which are fair or poor. Note any over- or under-coverage patterns visible in the plots.>

## Posterior Z-Score and Contraction

![Z-score and contraction](z_score_contraction.png)

The z-score–contraction plot summarizes posterior quality in two dimensions. The x-axis shows **posterior contraction** — the fraction by which the posterior variance has shrunk relative to the prior variance. Values near 0 indicate no information gain (the data are uninformative for that parameter); values near 1 indicate near-complete information gain. The y-axis shows the **posterior z-score** — the average standardized deviation between the posterior mean and the true value. Symmetric values around 0 indicate an unbiased (Gaussian-like) posterior. The ideal region is the middle-right corner (z-scores distributed around 0, high contraction).

**Assessment:** <1–3 sentences based on check_diagnostics() contraction ratings — state which parameters show high or medium contraction and which show low contraction or overconfidence.>

## Numerical Diagnostic Summary

| Metric | <param_1> | <param_2> | ... |
|--------|-----------|-----------|-----|
| NRMSE  | <0.042>   | <0.035>   | ... |
| Log-gamma | <...>  | <...>     | ... |
| ECE    | <0.031>   | <0.048>   | ... |
| Post. Contraction | <0.920> | <0.870> | ... |

<For each parameter, copy the qualitative summary from check_diagnostics() output and expand with problem-specific context. Use the exact ratings ("excellent", "good", "fair", "poor", "high", "medium", "low", "overconfident") — do NOT expose numeric thresholds.>

**<param_1>** — <check_diagnostics() summary for param_1, e.g., "excellent calibration; good recovery; high contraction">
**<param_2>** — <check_diagnostics() summary for param_2>

## Suggested Next Steps

<Provide 1–5 concrete, actionable steps. Use suggest_next_steps() output as the starting point, then expand with problem-specific context.>

1. <step>
2. <step>
3. <step>
---

<!-- Sections below are OPTIONAL — include only when real data is available -->

## Real-Data Inference

<Describe the observed data: source, sample size, preprocessing applied. State how many posterior samples were drawn.>

| Parameter | Posterior Median | 95% CI |
|-----------|-----------------|--------|
| <param_1> | <value> | [<lower>, <upper>] |
| <param_2> | <value> | [<lower>, <upper>] |

<Brief interpretation of the posterior — what do the estimates mean in the context of the problem?>

## Posterior Predictive Checks

<Describe which test quantities or visual comparisons were used. State how many posterior draws were passed through the simulator (default: 50).>

<Include or link to any PPC figures produced.>

**Assessment:** <1–3 sentences — do the replicated data match the observations? Are there systematic discrepancies? If so, what aspects of the data does the simulator fail to capture?>
````

## Common "Suggested Next Steps" patterns

Use these as a reference when filling in the final section:

- All diagnostics healthy → "Proceed to real-data inference and posterior predictive checks."
- Under-trained → "Increase the number of epochs (e.g., double the current value) and re-train."
- Calibration poor → "Consider increasing summary network capacity or training for more epochs."
- Recovery poor for specific parameters → "Those parameters may be weakly identifiable under the current prior and data design. Consider a more informative prior or a richer summary network."
- Contraction very high with poor calibration → "The posterior may be overconfident. Inspect the simulator for potential issues and increase the simulation budget."
- Overfitting detected → "Reduce network capacity, add regularization, or increase the simulation budget."
