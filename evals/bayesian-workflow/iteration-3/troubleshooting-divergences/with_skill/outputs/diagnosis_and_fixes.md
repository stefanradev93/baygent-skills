# Diagnosing and Fixing Divergences, High R-hat, and a Stuck Chain

## Executive summary

Your model is showing three interrelated convergence failures: 47 divergent transitions out of 4000 samples, R-hat of 1.03 on two parameters, and one chain stuck in a different region. These symptoms **all point to a posterior geometry problem** -- most likely a funnel or ridge that the sampler cannot navigate efficiently. Results from this model are **unreliable and must not be interpreted** until these issues are resolved. Below is a systematic diagnosis, concrete code fixes in order of priority, and verification steps to confirm the fixes work.

---

## 1. Diagnosis: What is happening and why

### 1.1 The three symptoms are connected

| Symptom | What it means | Severity |
|---------|--------------|----------|
| 47 divergences (out of 4000 draws) | The sampler hit regions of extreme curvature it could not navigate. Even 10+ divergences can bias results. | High -- results may be biased |
| R-hat = 1.03 on two parameters | Chains have not converged to the same distribution. The threshold is R-hat <= 1.01; yours exceeds it. | High -- parameter estimates are unreliable |
| One chain stuck | That chain found a local mode or got trapped in a problematic region (e.g., the neck of a funnel). | High -- confirms the other two symptoms |

These three issues rarely appear in isolation. The stuck chain is almost certainly **causing** many of the divergences (the sampler tries and fails to leave the problematic region) and **causing** the elevated R-hat (one chain exploring a different region inflates between-chain variance).

### 1.2 Most likely root causes

In decreasing order of probability:

1. **Funnel geometry in a hierarchical model** -- If you have group-level parameters with a prior on their standard deviation, the classic "Neal's funnel" makes the posterior narrow when the group SD is near zero and wide when it is large. The centered parameterization cannot navigate this.

2. **Flat or too-diffuse priors on scale parameters** -- Priors like `HalfCauchy`, `HalfFlat`, or very wide `HalfNormal` on standard deviations allow the sampler to wander into the near-zero region, creating the funnel.

3. **Highly correlated parameters** -- Strong posterior correlations (e.g., between an intercept and a slope when predictors are not standardized) create narrow ridges that are hard to explore.

4. **Multimodal posterior** -- The stuck chain may have found a separate mode. This can happen with mixture models, poorly identified models, or redundant parameterizations.

5. **Unstandardized predictors** -- If continuous predictors live on very different scales, shared priors become inappropriate and the posterior geometry becomes elongated, slowing mixing.

---

## 2. Immediate diagnostic steps (run these first)

Before applying fixes, gather information to identify which root cause applies to your model.

### 2.1 Full diagnostic summary

```python
import arviz as az
import numpy as np

# 1. Summary table -- check R-hat and ESS for ALL parameters
summary = az.summary(idata, round_to=3)
print(summary)

# 2. Identify the problematic parameters
high_rhat = summary[summary["r_hat"] > 1.01]
print("Parameters with R-hat > 1.01:")
print(high_rhat)

num_chains = idata.posterior.sizes["chain"]
low_ess = summary[summary["ess_bulk"] < 100 * num_chains]
print("Parameters with low ESS bulk:")
print(low_ess)

# 3. Count divergences
n_div = idata.sample_stats["diverging"].sum().item()
print(f"Total divergences: {n_div}")
```

### 2.2 Visualize where divergences occur

```python
# Trace and rank plots -- look for the stuck chain
az.plot_trace(idata, kind="rank_vlines")

# Pair plot with divergences highlighted
# Replace "param1" and "param2" with the two parameters that have R-hat = 1.03
az.plot_pair(
    idata,
    var_names=["param1", "param2"],
    divergences=True,
)
```

**What to look for in the pair plot:**
- If divergences cluster in a "funnel" shape (narrow at one end, wide at the other), you have a **funnel problem** -- go to Fix 1 and Fix 2.
- If divergences cluster along a narrow diagonal ridge, you have a **correlation problem** -- go to Fix 3.
- If divergences scatter without pattern, try increasing `target_accept` first (Fix 0).

### 2.3 Energy diagnostic

```python
az.plot_energy(idata)

# If the marginal energy and energy transition distributions are far apart,
# the sampler is not exploring the full posterior.
```

### 2.4 Check per-chain behavior

```python
# Per-chain divergence counts -- confirms whether the stuck chain is responsible
for chain in range(idata.posterior.sizes["chain"]):
    chain_divs = idata.sample_stats["diverging"].sel(chain=chain).sum().item()
    print(f"Chain {chain}: {chain_divs} divergences")
```

If most divergences come from the stuck chain, the geometry problem is localized and a reparameterization will likely fix it.

---

## 3. Fixes (apply in order)

### Fix 0: Increase `target_accept` (quick, sometimes sufficient)

This is the least invasive change. It makes the sampler take smaller steps, allowing it to navigate tighter curvature.

```python
import pymc as pm
import numpy as np

RANDOM_SEED = sum(map(ord, "my-analysis-name"))
rng = np.random.default_rng(RANDOM_SEED)

with model:
    idata = pm.sample(
        nuts_sampler="nutpie",
        target_accept=0.95,  # default is ~0.8; try 0.95 first, then 0.99
        random_seed=rng,
    )
    # Save immediately after sampling -- late crashes can destroy valid results
    idata.to_netcdf("model_output.nc")
```

**When this alone works:** mild geometry issues, small number of divergences (< 20), R-hat close to 1.01.

**When this is not enough:** funnel geometry, stuck chains, R-hat > 1.05. In your case (47 divergences + stuck chain), this alone is unlikely to fully resolve the problem, but apply it alongside the other fixes.

### Fix 1: Reparameterize -- non-centered parameterization

This is the single most effective fix for hierarchical model divergences. It eliminates the funnel by decoupling the group-level parameters from the group-level standard deviation.

```python
# ---- BEFORE (centered -- causes funnel) ----
with pm.Model(coords=coords) as model_centered:
    mu_global = pm.Normal("mu_global", mu=0, sigma=10)
    # Flat prior on SD creates funnel near zero
    sigma_group = pm.HalfCauchy("sigma_group", beta=5)

    # Group means drawn directly from the population distribution
    mu_group = pm.Normal("mu_group", mu=mu_global, sigma=sigma_group, dims="group")

    sigma_obs = pm.Gamma("sigma_obs", alpha=2, beta=2)
    pm.Normal("y", mu=mu_group[group_idx], sigma=sigma_obs, observed=y_data, dims="obs")
```

```python
# ---- AFTER (non-centered -- eliminates funnel) ----
with pm.Model(coords=coords) as model_noncentered:
    mu_global = pm.Normal("mu_global", mu=0, sigma=10)
    # Better prior: Gamma avoids the near-zero region that creates the funnel
    sigma_group = pm.Gamma("sigma_group", alpha=2, beta=2)

    # Offset drawn from standard normal (independent of sigma_group)
    mu_raw = pm.Normal("mu_raw", mu=0, sigma=1, dims="group")
    # Deterministic transform recovers the group means
    mu_group = pm.Deterministic(
        "mu_group",
        mu_global + mu_raw * sigma_group,
        dims="group",
    )

    sigma_obs = pm.Gamma("sigma_obs", alpha=2, beta=2)
    pm.Normal("y", mu=mu_group[group_idx], sigma=sigma_obs, observed=y_data, dims="obs")

    idata = pm.sample(
        nuts_sampler="nutpie",
        target_accept=0.95,
        random_seed=rng,
    )
    idata.to_netcdf("model_output_noncentered.nc")
```

**Why this works:** In the centered parameterization, `mu_group` depends directly on `sigma_group`. When `sigma_group` is near zero, all `mu_group` values must collapse to `mu_global`, creating a narrow funnel the sampler cannot navigate. The non-centered version draws `mu_raw` from a fixed standard normal -- no dependence on `sigma_group` during sampling.

**Rule of thumb from the skill reference:** Start with non-centered. Switch to centered only if non-centered shows poor ESS AND groups have substantial data (50+ observations each).

### Fix 2: Use better priors on scale parameters

Flat or heavy-tailed priors on standard deviations (`HalfCauchy`, `HalfFlat`, `HalfNormal(sigma=100)`) allow the sampler to explore the near-zero region, which is the source of the funnel.

```python
# ---- BAD: flat-ish priors on scale parameters ----
# sigma_group = pm.HalfCauchy("sigma_group", beta=5)   # heavy tail, allows near-zero
# sigma_group = pm.HalfFlat("sigma_group")              # improper prior
# sigma_group = pm.HalfNormal("sigma_group", sigma=100) # effectively flat

# ---- GOOD: priors that avoid the near-zero region ----
# Gamma(2, beta) has zero density at zero and a mode at 1/beta:
sigma_group = pm.Gamma("sigma_group", alpha=2, beta=2)  # mode at 0.5, mean at 1.0

# Exponential is another good choice:
sigma_group = pm.Exponential("sigma_group", lam=1)       # mean at 1.0, mode at 0

# For observation-level SD when you have a rough sense of scale:
sigma_obs = pm.Gamma("sigma_obs", alpha=2, beta=1)       # mode at 1.0, mean at 2.0
```

**Key insight from the skill reference:** If there is no group-level variation to detect, you do not need the hierarchy. A `sigma_group` posterior that piles up near zero is telling you the data does not support grouping -- consider removing that hierarchical level entirely.

### Fix 3: Standardize predictors

If your model includes continuous predictors on different scales (e.g., income in dollars and age in years), the posterior will have elongated ridges that slow mixing and cause correlations between parameters.

```python
import pandas as pd

# Standardize continuous predictors before modeling
predictors_to_standardize = ["age", "income", "tenure"]

# Store the means and SDs for back-transformation later
scaler_params = {}
for col in predictors_to_standardize:
    mean_val = df[col].mean()
    std_val = df[col].std()
    scaler_params[col] = {"mean": mean_val, "std": std_val}
    df[f"{col}_std"] = (df[col] - mean_val) / std_val

# Now use standardized predictors in the model
# This makes Normal(0, 2.5) an appropriate shared prior for all coefficients
with pm.Model(coords=coords) as model:
    # Weakly informative on standardized scale
    beta = pm.Normal("beta", mu=0, sigma=2.5, dims="predictor")
    # ...
```

After fitting, back-transform coefficients for interpretation:

```python
# Back-transform a coefficient from standardized to raw scale
# beta_raw = beta_std / std_x
# intercept_raw = intercept_std - sum(beta_std * mean_x / std_x)
for i, col in enumerate(predictors_to_standardize):
    raw_coef = (
        idata.posterior["beta"].sel(predictor=col)
        / scaler_params[col]["std"]
    )
    print(f"{col} (raw scale): {raw_coef.mean().item():.3f}")
```

### Fix 4: Check for multimodality

If the stuck chain found a separate mode, reparameterization alone may not help.

```python
# Check if the stuck chain occupies a different region
for param in ["param1", "param2"]:  # the two parameters with R-hat = 1.03
    for chain in range(idata.posterior.sizes["chain"]):
        chain_mean = idata.posterior[param].sel(chain=chain).mean().item()
        chain_std = idata.posterior[param].sel(chain=chain).std().item()
        print(f"{param} - Chain {chain}: mean={chain_mean:.3f}, std={chain_std:.3f}")
```

If one chain has a dramatically different mean, consider:

1. **Check model identifiability** -- Are there redundant parameters? (e.g., two intercepts that are not jointly constrained)
2. **Add sum-to-zero constraints** for categorical effects:
   ```python
   # Use ZeroSumNormal for group effects to remove the ambiguity
   # between the global intercept and group-level intercepts
   group_effect = pm.ZeroSumNormal("group_effect", sigma=2, dims="group")
   ```
3. **Run more tuning steps** to give chains more time to find the main mode:
   ```python
   idata = pm.sample(
       nuts_sampler="nutpie",
       tune=2000,       # default is 1000; give more warmup
       draws=2000,
       target_accept=0.95,
       random_seed=rng,
   )
   ```

### Fix 5: Marginalize discrete parameters (if applicable)

If your model contains discrete latent variables (e.g., mixture component indicators), NUTS cannot sample them. PyMC handles some cases automatically, but if not:

```python
# Instead of a discrete mixture indicator, use a marginalized mixture:
weights = pm.Dirichlet("weights", a=np.ones(K))
components = [
    pm.Normal.dist(mu=mu[k], sigma=sigma[k])
    for k in range(K)
]
pm.Mixture("y", w=weights, comp_dists=components, observed=y_data, dims="obs")
```

---

## 4. Recommended fix sequence for your specific case

Given your symptoms (47 divergences, R-hat = 1.03, one stuck chain), apply these steps in order:

```
Step 1: Run the diagnostics from Section 2 to identify which parameters and
        geometry are causing the problem.

Step 2: Apply Fix 1 (non-centered parameterization) + Fix 2 (better priors
        on scale parameters) + Fix 0 (target_accept=0.95).
        These three together resolve ~90% of divergence cases.

Step 3: Apply Fix 3 (standardize predictors) if you have continuous covariates
        on different scales.

Step 4: Re-run the model and check diagnostics again.

Step 5: If problems persist, increase target_accept to 0.99 and increase
        tune to 2000.

Step 6: If problems STILL persist, investigate multimodality (Fix 4) and
        model identifiability.
```

---

## 5. Verification: How to confirm the fix worked

After applying fixes, run the full diagnostic checklist. **Do not interpret results until every check passes.**

```python
import pymc as pm
import arviz as az
import numpy as np

RANDOM_SEED = sum(map(ord, "my-analysis-name"))
rng = np.random.default_rng(RANDOM_SEED)

# ---- After re-fitting the fixed model ----

# 1. Summary table
summary = az.summary(idata, round_to=3)
print(summary)

# 2. R-hat: ALL parameters must be <= 1.01
rhat_ok = (summary["r_hat"] <= 1.01).all()
print(f"R-hat OK (all <= 1.01): {rhat_ok}")

# 3. ESS: must be >= 100 * number of chains for both bulk and tail
num_chains = idata.posterior.sizes["chain"]
ess_bulk_ok = (summary["ess_bulk"] >= 100 * num_chains).all()
ess_tail_ok = (summary["ess_tail"] >= 100 * num_chains).all()
print(f"ESS bulk OK (>= {100 * num_chains}): {ess_bulk_ok}")
print(f"ESS tail OK (>= {100 * num_chains}): {ess_tail_ok}")

# 4. Divergences: must be zero
n_div = idata.sample_stats["diverging"].sum().item()
print(f"Divergences: {n_div} (target: 0)")

# 5. Visual check: rank plots should look uniform, no stuck chains
az.plot_trace(idata, kind="rank_vlines")

# 6. Energy diagnostic: distributions should overlap
az.plot_energy(idata)

# 7. Pair plot: no funnel patterns, no clustered divergences
az.plot_pair(
    idata,
    var_names=["param1", "param2"],
    divergences=True,
)
```

### Passing criteria

| Diagnostic | Pass criterion |
|-----------|---------------|
| R-hat | All parameters <= 1.01 |
| ESS bulk | All parameters >= 100 * number of chains |
| ESS tail | All parameters >= 100 * number of chains |
| Divergences | 0 |
| Rank plots | Uniform (no spikes, no gaps) |
| Trace plots | All chains overlapping, no stuck chains |
| Energy plot | Marginal and transition distributions overlap |

### After convergence passes, continue with model criticism

Convergence only tells you the sampler worked -- it says nothing about whether the model is appropriate for the data. Once all checks above pass, proceed with:

```python
# Posterior predictive check
with model:
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=rng))

az.plot_ppc(idata)

# Calibration check (mandatory)
import arviz_plots as azp
azp.plot_ppc_pit(idata)

# LOO-CV for predictive accuracy
# Note: nutpie does not auto-store log-likelihood; compute it explicitly
with model:
    pm.compute_log_likelihood(idata, model=model)

loo = az.loo(idata, pointwise=True)
print(loo)
az.plot_khat(loo)

# Save the final InferenceData
idata.to_netcdf("model_output_fixed.nc")
```

---

## 6. Quick-reference decision tree

```
47 divergences + R-hat 1.03 + stuck chain
|
|-- Is this a hierarchical model?
|   |
|   |-- YES --> Apply non-centered parameterization (Fix 1)
|   |           + Gamma/Exponential prior on group SD (Fix 2)
|   |           + target_accept=0.95 (Fix 0)
|   |
|   |-- NO --> Are predictors unstandardized?
|       |
|       |-- YES --> Standardize predictors (Fix 3)
|       |           + target_accept=0.95 (Fix 0)
|       |
|       |-- NO --> Check pair plot for geometry
|           |
|           |-- Funnel shape --> Reparameterize (Fix 1 adapted)
|           |-- Ridge shape --> Reduce correlations, standardize
|           |-- Scattered --> Increase target_accept to 0.99 (Fix 0)
|           |-- Separate modes --> Check identifiability (Fix 4)
|
|-- After fix: re-run all diagnostics (Section 5)
|   |
|   |-- All pass --> Proceed to model criticism
|   |-- Still failing --> Escalate: increase tune=2000,
|       target_accept=0.99, or revisit model specification
```

---

## 7. What NOT to do

- **Do not ignore the divergences and interpret results anyway.** Even 10+ divergences can systematically bias posterior estimates. With 47 divergences and a stuck chain, the bias could be severe.
- **Do not just add more draws.** More samples from a broken sampler gives you more broken samples. Fix the geometry first.
- **Do not drop the stuck chain and use the remaining chains.** This hides the problem rather than fixing it. The stuck chain is giving you valuable diagnostic information.
- **Do not use flat priors "to be safe."** Flat priors on scale parameters are a leading cause of funnel divergences. Weakly informative priors (Gamma, Exponential) are both statistically sound and computationally helpful.

---

## Appendix: Key thresholds reference

| Metric | Healthy | Concerning | Unacceptable |
|--------|---------|------------|-------------|
| Divergences | 0 | 1-9 | 10+ (your case: 47) |
| R-hat | <= 1.01 | 1.01-1.05 | > 1.05 (your case: 1.03) |
| ESS bulk | >= 100 * chains | 100-100 * chains | < 100 |
| ESS tail | >= 100 * chains | 100-100 * chains | < 100 |
| Pareto k (LOO) | < 0.5 | 0.5-0.7 | > 0.7 |
