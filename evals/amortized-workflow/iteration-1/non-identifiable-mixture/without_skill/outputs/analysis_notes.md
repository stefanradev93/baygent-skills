# Analysis Notes: Non-Identifiable Gaussian Mixture Model

## Model Description

Two-component Gaussian mixture with parameters:
- **w** ~ Beta(2, 2): mixing weight in (0, 1)
- **mu1** ~ Normal(0, 5): mean of component 1
- **mu2** ~ Normal(0, 5): mean of component 2
- **sigma** ~ Gamma(2, 1): shared standard deviation (> 0)

Observation model: x_i ~ w * N(mu1, sigma) + (1-w) * N(mu2, sigma), for i = 1..80.

## The Non-Identifiability Problem

### Label switching

The core issue is a **discrete symmetry** in the likelihood. For any parameter vector (w, mu1, mu2, sigma), the vector (1-w, mu2, mu1, sigma) produces an identical data distribution. This is called **label switching** -- the two component labels ("component 1" and "component 2") are arbitrary.

This symmetry is **exact** when w = 0.5 (the components are equally weighted, so swapping them changes nothing). When w is far from 0.5, the symmetry is broken by the prior -- one labeling has higher prior probability -- but the likelihood still has the symmetry.

### Why this matters for amortized inference

A neural density estimator (like FlowMatching) trained on simulations from this model will see both labelings in the training data with roughly equal frequency (because the Beta(2, 2) prior is symmetric around 0.5). The network must learn to represent a **bimodal posterior** for mu1 and mu2, which is fundamentally harder than a unimodal one. In practice:

1. The network may learn the average of the two modes (a poor point estimate)
2. It may collapse to one mode inconsistently across simulations
3. It may learn a broad, poorly calibrated distribution that covers both modes

## Expected Diagnostic Behavior

### Recovery plots (posterior z-scores / true vs. estimated)

| Parameter | Expected behavior | Why |
|-----------|------------------|-----|
| **w** | Moderate recovery; may cluster around 0.5 | When w ~ 0.5, it is well-defined but the mixture is maximally ambiguous |
| **mu1** | Poor recovery; scattered, low R2 | Label switching means the network cannot consistently assign the correct value |
| **mu2** | Poor recovery; scattered, low R2 | Same as mu1 -- the two means are interchangeable |
| **sigma** | Good recovery; tight correlation | Shared across components, unaffected by label switching |

### Numerical metrics

- **NRMSE**: Expect mu1 and mu2 to have NRMSE > 0.8 (possibly near 1.0). w should be moderate (0.3-0.6). sigma should be low (< 0.3).
- **R2 (coefficient of determination)**: Expect mu1 and mu2 near 0 or negative (worse than predicting the prior mean). sigma should be > 0.7.
- **Calibration (ECDF-based)**: May show poor calibration for mu1 and mu2 -- the credible intervals are either too wide (covering both modes) or too narrow (covering only one mode and missing the true value half the time).

### What the joint posterior looks like

For a test dataset generated with w = 0.5, the joint posterior of (mu1, mu2) should show a characteristic **bimodal pattern**: two clusters at (true_mu1, true_mu2) and (true_mu2, true_mu1). If the network has learned well, these two clusters should be distinct. If the network has averaged over the modes, you will see a single broad blob centered between the two true solutions.

## Interpreting Poor Recovery

Poor recovery for mu1 and mu2 does **not** mean the network failed to learn. It means the model has a fundamental identifiability problem that no amount of training can fix. The diagnostics are working correctly by flagging this.

Key signs that the issue is non-identifiability (not training failure):
1. **sigma recovers well** -- the network learned the data-generating process; it just cannot resolve the label ambiguity
2. **w recovers reasonably** -- especially when w is far from 0.5
3. **Training loss converged** -- the network reached a stable loss plateau
4. **The bimodal pattern appears in the joint posterior** -- the network actually learned both modes

If ALL parameters show poor recovery, the problem is more likely training failure (too few epochs, too small architecture, or a bug in the simulator).

## Remediation Strategies

### 1. Ordering constraint (recommended)

Impose mu1 < mu2 (or mu1 > mu2) as a hard constraint. This breaks the symmetry by defining "component 1" as the one with the smaller mean. Implementation:

```python
# In the prior, sort the means
def prior(rng=None):
    rng = rng or np.random.default_rng()
    w = rng.beta(2, 2)
    mu_raw = rng.normal(0, 5, size=2)
    mu1, mu2 = np.sort(mu_raw)  # enforce mu1 < mu2
    sigma = rng.gamma(2, 1)
    return dict(w=np.array(w), mu1=np.array(mu1), mu2=np.array(mu2), sigma=np.array(sigma))
```

This is the cleanest fix. The adapter constraint `constrain("mu1", upper="mu2")` may or may not be supported; sorting in the prior is always safe.

### 2. Post-hoc relabeling

After obtaining posterior samples, sort the (mu1, mu2) pairs so that the smaller mean is always "mu1". This fixes the diagnostic metrics without changing the model. However, it requires care with the corresponding w values (swap w to 1-w when you swap the means).

### 3. Identifiable reparameterization

Instead of reporting mu1 and mu2, report:
- **mu_min** = min(mu1, mu2)
- **mu_max** = max(mu1, mu2)
- **separation** = |mu1 - mu2|
- **midpoint** = (mu1 + mu2) / 2

These quantities are invariant to label switching and should show good recovery even without constraints.

### 4. Increase component separation in the prior

Use a prior that encourages well-separated components (e.g., a repulsive prior or a prior on |mu1 - mu2| that penalizes values near 0). This reduces but does not eliminate the problem.

## When Non-Identifiability Is Acceptable

Sometimes you do not care about which component is which -- you only care about the mixture density itself. In that case, the non-identifiability is irrelevant for prediction:
- **Density estimation**: The predicted mixture density is identical under both labelings
- **Posterior predictive checks**: New data generated from the posterior will look correct regardless of labeling
- **Component separation**: |mu1 - mu2| is identifiable and can be reported

The non-identifiability only matters when you need to interpret individual component parameters (e.g., "the first component has mean -2 and represents subpopulation A").

## Summary

| Aspect | Status |
|--------|--------|
| Model specification | Correct but non-identifiable |
| Network training | Expected to converge normally |
| sigma recovery | Should be good |
| w recovery | Should be moderate |
| mu1, mu2 recovery | Expected to be poor (label switching) |
| Recommended fix | Ordering constraint mu1 < mu2 in the prior |
| Alternative | Report identifiable quantities (separation, midpoint) |
