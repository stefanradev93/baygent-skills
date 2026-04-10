# Diagnostic Assessment: Divergences and Convergence Issues

## Summary of Symptoms

| Diagnostic         | Observed Value       | Acceptable Threshold | Status      |
|--------------------|----------------------|----------------------|-------------|
| Divergences        | 47 / 4000 samples    | 0                    | PROBLEMATIC |
| R-hat              | 1.03 (two parameters)| < 1.01               | PROBLEMATIC |
| Trace plots        | One chain stuck      | All chains mixing    | PROBLEMATIC |

---

## 1. Understanding the Problem

### What divergences mean

Divergent transitions indicate that the Hamiltonian Monte Carlo (HMC) sampler encountered regions of the posterior geometry where the curvature changes so sharply that the leapfrog integrator cannot accurately track the Hamiltonian trajectory. When this happens, the sampler is forced to reject the proposed move, which means **the resulting samples do not faithfully represent the posterior distribution**. Even a small number of divergences can indicate that entire regions of the posterior are being systematically missed.

47 divergences out of 4000 samples (~1.2%) is a meaningful proportion. This is not a borderline case -- it signals a real geometric problem in the posterior.

### What elevated R-hat means

R-hat (the potential scale reduction factor) compares the variance *between* chains to the variance *within* chains. An R-hat of 1.03 on two parameters exceeds the modern recommended threshold of 1.01 (Vehtari et al., 2021). This confirms that the chains have not converged to the same stationary distribution.

### What a stuck chain means

A chain that appears "stuck" in the trace plot is exhibiting one of the following:
- It has become trapped in a local mode of a multimodal posterior.
- It is struggling to traverse a narrow region of the parameter space (a "funnel" or "pinch").
- The step size has adapted poorly for that chain, preventing it from exploring efficiently.

The combination of all three symptoms -- divergences, elevated R-hat, and a stuck chain -- paints a consistent picture: **the posterior geometry is difficult for the sampler, and the current model parameterization or sampler configuration is inadequate**.

---

## 2. Immediate Diagnostic Steps

Before applying fixes, gather more information to pinpoint the root cause.

### 2.1 Inspect divergence locations in parameter space

```python
import arviz as az
import matplotlib.pyplot as plt

# Assuming `idata` is your InferenceData object
az.plot_pair(
    idata,
    var_names=["param1", "param2"],  # replace with your problematic parameters
    kind="scatter",
    divergences=True,
    divergences_kwargs={"color": "red", "markersize": 4},
    figsize=(10, 10),
)
plt.suptitle("Pairwise scatter with divergences highlighted")
plt.tight_layout()
plt.savefig("divergence_pairs.png", dpi=150)
plt.show()
```

**What to look for:** If divergences cluster in a specific region (e.g., near zero for a scale parameter, or along a narrow ridge), that tells you *where* the geometry is problematic. Common patterns include:
- **Funnel shape:** Divergences at the narrow end of a funnel -- classic sign of a centered parameterization problem.
- **Boundary clustering:** Divergences near the boundary of the prior support -- suggests a prior-data conflict or a parameter trying to collapse to a boundary.
- **Ridge or banana shape:** Divergences along a curved ridge -- indicates strong correlation between parameters that the sampler cannot navigate.

### 2.2 Check the energy diagnostic

```python
az.plot_energy(idata)
plt.savefig("energy_plot.png", dpi=150)
plt.show()
```

If the marginal energy distribution and the energy transition distribution are very different (large energy Bayesian fraction of missing information, or E-BFMI), the sampler is not exploring efficiently.

### 2.3 Examine per-chain summary

```python
# Check R-hat and ESS per parameter
summary = az.summary(idata, var_names=["param1", "param2"])
print(summary)

# Check ESS specifically -- low ESS is another red flag
az.plot_ess(idata, var_names=["param1", "param2"], kind="evolution")
plt.savefig("ess_evolution.png", dpi=150)
plt.show()
```

### 2.4 Look at the trace plots carefully

```python
az.plot_trace(idata, var_names=["param1", "param2"], compact=False)
plt.tight_layout()
plt.savefig("trace_plots.png", dpi=150)
plt.show()
```

Identify *which* chain is stuck and *which* parameters it is stuck on. This narrows down the source of the problem.

---

## 3. Recommended Fixes (in order of priority)

### Fix 1: Reparameterize the Model (Non-Centered Parameterization)

**This is the single most effective fix for divergences in hierarchical models.**

The most common cause of divergences in Bayesian models is the "Neal's funnel" geometry that arises in hierarchical (multilevel) models when using a *centered* parameterization. The solution is to switch to a *non-centered* parameterization.

#### Centered (problematic) parameterization:

```python
import pymc as pm

with pm.Model() as centered_model:
    # Hyperpriors
    mu = pm.Normal("mu", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=2)

    # Centered parameterization -- creates a funnel
    theta = pm.Normal("theta", mu=mu, sigma=sigma, shape=n_groups)

    # Likelihood
    y = pm.Normal("y", mu=theta[group_idx], sigma=obs_sigma, observed=y_obs)
```

#### Non-centered (fixed) parameterization:

```python
with pm.Model() as noncentered_model:
    # Hyperpriors
    mu = pm.Normal("mu", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=2)

    # Non-centered parameterization -- breaks the funnel
    theta_offset = pm.Normal("theta_offset", mu=0, sigma=1, shape=n_groups)
    theta = pm.Deterministic("theta", mu + sigma * theta_offset)

    # Likelihood
    y = pm.Normal("y", mu=theta[group_idx], sigma=obs_sigma, observed=y_obs)
```

**Why this works:** In the centered form, `theta` depends directly on `sigma`. When `sigma` is small, `theta` is tightly constrained around `mu`, creating a narrow funnel in the joint distribution. The sampler's step size cannot simultaneously handle the wide part (large `sigma`) and the narrow part (small `sigma`). The non-centered form breaks this dependency: `theta_offset` is standard normal regardless of `sigma`, so the geometry is uniform and the sampler can explore efficiently.

### Fix 2: Increase `target_accept` (adapt_delta)

Raising the target acceptance rate forces the sampler to take smaller steps, which can help it navigate difficult geometry. This is a **palliative measure**, not a cure -- it trades speed for accuracy.

```python
with model:
    idata = pm.sample(
        draws=2000,
        tune=2000,
        target_accept=0.95,  # default is 0.8; try 0.90, 0.95, or 0.99
        random_seed=42,
    )
```

**Guidelines:**
- Start with `0.90`, then try `0.95`, then `0.99` if needed.
- If you need `target_accept > 0.99` to eliminate divergences, the model almost certainly needs reparameterization instead.
- Higher `target_accept` means smaller step sizes, which means more leapfrog steps per sample, which means slower sampling.

### Fix 3: Use More Informative (but Weakly Informative) Priors

Vague or improper priors can create pathological posterior geometries. If a scale parameter's prior allows it to approach zero or infinity, the resulting posterior can have extreme curvature.

```python
# Instead of very vague priors:
# sigma = pm.HalfCauchy("sigma", beta=10)  # very heavy tails

# Use weakly informative priors that encode genuine prior knowledge:
sigma = pm.HalfNormal("sigma", sigma=2)    # or
sigma = pm.Exponential("sigma", lam=1)      # or
sigma = pm.InverseGamma("sigma", alpha=2, beta=1)  # if you need a soft lower bound
```

**Prior predictive checks** can help you calibrate these:

```python
with model:
    prior_pred = pm.sample_prior_predictive(samples=1000)

az.plot_ppc(prior_pred, group="prior")
plt.savefig("prior_predictive_check.png", dpi=150)
plt.show()
```

If the prior predictive distribution generates data that is wildly inconsistent with what you would ever expect to observe, the priors are too vague.

### Fix 4: Increase Tuning Samples

The stuck chain may have adapted poorly during warmup. Giving the sampler more tuning iterations can help, especially for models with complex geometry.

```python
with model:
    idata = pm.sample(
        draws=2000,
        tune=4000,  # double the tuning (default is usually 1000)
        target_accept=0.95,
        random_seed=42,
    )
```

This gives the NUTS sampler more time to find a good step size and mass matrix for all chains.

### Fix 5: Use a Better Mass Matrix Estimation (or Provide One)

If one chain is stuck, it may have estimated a poor mass matrix during warmup. You can try:

```python
with model:
    idata = pm.sample(
        draws=2000,
        tune=2000,
        init="advi+adapt_diag",  # use ADVI to find a better starting mass matrix
        target_accept=0.95,
        random_seed=42,
    )
```

Other initialization strategies to try:
- `init="jitter+adapt_diag"` (default) -- adds jitter to MAP estimate
- `init="adapt_full"` -- estimates a full (dense) mass matrix, which can handle correlations but is more expensive
- `init="advi+adapt_diag"` -- uses variational inference to find a starting point

### Fix 6: Add More Chains and Increase Samples

More chains provide better convergence diagnostics and reduce the impact of one poorly-behaving chain.

```python
with model:
    idata = pm.sample(
        draws=2000,
        tune=2000,
        chains=6,  # more chains for better diagnostics
        cores=4,
        target_accept=0.95,
        random_seed=42,
    )
```

### Fix 7: Investigate and Address Parameter Correlations

If the pair plot from diagnostic step 2.1 reveals strong correlations between parameters, consider:

```python
# Option A: Use a dense mass matrix
with model:
    idata = pm.sample(
        draws=2000,
        tune=2000,
        init="adapt_full",  # full mass matrix handles correlations
        target_accept=0.95,
    )

# Option B: Restructure the model to reduce correlations
# For example, center predictors in a regression:
X_centered = X - X.mean(axis=0)

with pm.Model() as model:
    beta = pm.Normal("beta", mu=0, sigma=5, shape=X_centered.shape[1])
    intercept = pm.Normal("intercept", mu=y_obs.mean(), sigma=5)
    mu = intercept + pm.math.dot(X_centered, beta)
    # ...
```

---

## 4. Recommended Workflow

Apply these fixes in the following order, re-checking diagnostics after each step:

1. **Reparameterize** (Fix 1) -- this alone often eliminates all divergences.
2. **Increase `target_accept`** (Fix 2) -- quick to try and often helps.
3. **Tighten priors** (Fix 3) -- if prior predictive checks reveal issues.
4. **Increase tuning** (Fix 4) -- if chains are slow to adapt.
5. **Change initialization** (Fix 5) -- if one chain consistently gets stuck.
6. **Add chains** (Fix 6) -- for more robust diagnostics.
7. **Handle correlations** (Fix 7) -- if pair plots reveal strong dependencies.

After each fix, verify:

```python
# Check for divergences
divergences = idata.sample_stats["diverging"].sum().values
print(f"Divergences: {divergences}")

# Check R-hat
summary = az.summary(idata)
print(f"Max R-hat: {summary['r_hat'].max():.4f}")

# Check ESS
print(f"Min bulk ESS: {summary['ess_bulk'].min():.0f}")
print(f"Min tail ESS: {summary['ess_tail'].min():.0f}")

# Visual check
az.plot_trace(idata)
plt.show()
```

**Convergence targets:**
- Zero divergences
- All R-hat values < 1.01
- Bulk ESS > 400 per chain (or > 100 per chain at minimum)
- Tail ESS > 100 per chain
- All chains mixing well in trace plots

---

## 5. What NOT to Do

- **Do not simply discard divergent samples.** Divergences indicate the sampler is failing to explore parts of the posterior. Removing them biases your results.
- **Do not just run more samples without fixing the underlying issue.** More samples from a broken sampler are still biased samples.
- **Do not ignore a stuck chain.** Dropping a chain and reporting results from the remaining ones hides a convergence failure. Fix the model so all chains converge.
- **Do not set `target_accept=0.999` and call it done.** If you need extremely high acceptance rates, the model geometry is the problem, not the sampler settings.

---

## References

- Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
- Betancourt, M. and Girolami, M. (2015). "Hamiltonian Monte Carlo for Hierarchical Models." In Current Trends in Bayesian Methodology with Applications.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Burkner, P.-C. (2021). "Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC." Bayesian Analysis, 16(2), 667-718.
- Stan Development Team. "Divergent transitions -- what they are and how to address them." Stan documentation.
- PyMC documentation on NUTS sampler diagnostics.
