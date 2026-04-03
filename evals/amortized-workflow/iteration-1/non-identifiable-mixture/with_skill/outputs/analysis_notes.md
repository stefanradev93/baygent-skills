# Analysis Notes: Two-Component Gaussian Mixture with Label Switching

## Model specification

The generative model draws N observations from a two-component Gaussian mixture:

- `w ~ Beta(2, 2)` -- mixing weight
- `mu1 ~ Normal(0, 5)` -- mean of component 1
- `mu2 ~ Normal(0, 5)` -- mean of component 2
- `sigma ~ Gamma(2, 1)` -- shared standard deviation
- For each observation: with probability `w`, draw from `N(mu1, sigma)`; otherwise from `N(mu2, sigma)`

## The label-switching problem

This model has a fundamental **symmetry** (also called a **permutation invariance**): swapping `(w, mu1, mu2)` with `(1-w, mu2, mu1)` produces the exact same likelihood for any dataset. This means the true posterior is **bimodal** -- it has two equally valid modes related by the swap.

When `w` is near 0.5, the two modes are nearly equally weighted and the symmetry is strongest. When `w` is far from 0.5, one mode dominates and the effective non-identifiability is weaker (though still present in principle).

This is not a bug in the model or the estimator -- it is a genuine structural property of mixture models without ordering constraints.

## Expected diagnostic behavior

### Parameters we expect to recover well

- **sigma**: The shared standard deviation is invariant to label switching. Both modes of the posterior agree on its value. We expect low NRMSE, good calibration (ECE < 0.05), and strong contraction.
- **w**: Although `w` and `1-w` are both valid, the Beta(2, 2) prior and the constraint to (0, 1) may allow partial recovery. However, when `w` is near 0.5, the posterior for `w` will be broad and centered around 0.5, giving weak contraction.

### Parameters we expect to show poor recovery

- **mu1 and mu2**: These are the parameters most affected by label switching. The amortized estimator will try to learn a single mapping from data to posterior, but the true posterior is bimodal. A unimodal approximation (flow matching) will either:
  - Spread probability across both modes, producing high NRMSE and low contraction (the posterior is wide and centered between the two modes)
  - Collapse to one mode arbitrarily, producing poor calibration (ECE > 0.10) because which mode it picks may not match the "ground truth" labeling

In practice, we expect to see:
- **NRMSE > 0.15** for mu1 and/or mu2 -- triggering a STOP
- **ECE > 0.10** for mu1 and/or mu2 -- confirming miscalibration
- **Contraction < 0.80** for mu1 and/or mu2 -- indicating weak information gain
- **Recovery plots** showing a "cloud" or "X" pattern for mu1 and mu2, where the network sometimes confuses which component is which

## Interpreting the diagnostics

### If the diagnostics report says STOP

This is the **expected outcome** for this model. It confirms that label switching is causing the amortized estimator to fail on mu1 and mu2. This is a correct diagnosis -- you should NOT proceed to real-data inference with these parameter estimates.

Look at the per-parameter breakdown:
- If mu1 and mu2 both show FAIL on NRMSE or ECE: classic label switching
- If w also shows weak contraction: the weight is near 0.5 too often for the network to disambiguate
- If sigma passes all checks: the shared parameter is correctly identified as invariant to the symmetry

### If the diagnostics report says WARN

Partial label switching. The prior on w may be pushing it away from 0.5 often enough that the network learns a reasonable mapping for most simulations, but fails on the near-symmetric cases.

### If the diagnostics report says GO

This would be surprising and worth investigating. Possible explanations:
- The network learned to always assign mu1 < mu2 (an implicit ordering) -- check recovery plots
- The prior on w rarely produces values near 0.5 -- check the test simulation distribution of w
- The sample sizes (N) are large enough that even near w = 0.5, the means are distinguishable

## Remediation strategies

If diagnostics confirm non-identifiability (the expected outcome), there are several principled fixes:

### 1. Impose an ordering constraint (recommended)

Force `mu1 < mu2` in the prior and simulator. This breaks the symmetry and makes the model identifiable:

```python
def prior_ordered():
    w = rng.beta(2, 2)
    mu_low = rng.normal(0, 5)
    mu_gap = rng.exponential(2)  # always positive
    mu1 = mu_low
    mu2 = mu_low + mu_gap         # guarantees mu2 > mu1
    sigma = rng.gamma(2, 1)
    return dict(w=w, mu1=mu1, mu2=mu2, sigma=sigma)
```

This is the simplest and most effective fix. After retraining with this prior, mu1 and mu2 should show good recovery.

### 2. Reparameterize to identifiable quantities

Instead of inferring (w, mu1, mu2), infer quantities that are invariant to the permutation:
- The overall mean: `w * mu1 + (1-w) * mu2`
- The spread between means: `|mu1 - mu2|`
- The overall variance contribution: `sigma`

This requires changing the simulator output and adapter but avoids the identifiability issue entirely.

### 3. Post-hoc relabeling

After inference, sort the posterior samples so that mu1 < mu2 within each draw. This is a post-processing step and does not require retraining. However, it is fragile when the means are close together.

### 4. Increase model capacity

Switching from FlowMatching to a more expressive multimodal approximator could in principle capture both modes. However, this is harder to train and diagnose, and the ordering constraint (strategy 1) is almost always preferable.

## Recommendation

Start with strategy 1 (ordering constraint). Retrain with the ordered prior, rerun diagnostics, and verify that all parameters now pass the house thresholds. Only then proceed to real-data inference.

## Simulation budget

| Item | Value |
|------|-------|
| Training mode | Online |
| Training sims | 150 epochs x 100 batches x 32 = 480,000 |
| Validation sims | 300 (auto-simulated) |
| Diagnostic sims | 300 held-out |
| Summary network | SetTransformer (Small) |
| Inference network | FlowMatching (Small) |
| Parameters | 4 (w, mu1, mu2, sigma) |
| Observations per sim | 30--200 (varied via meta_fn) |

## Limitations

- The Beta(2, 2) prior on w concentrates mass near 0.5, which maximizes the label-switching problem. A more informative prior on w would reduce (but not eliminate) the issue.
- The shared sigma assumption is a simplification; with component-specific sigmas, the identifiability picture changes.
- FlowMatching learns a unimodal approximate posterior; it cannot represent the true bimodal posterior of a symmetric mixture without ordering constraints.
