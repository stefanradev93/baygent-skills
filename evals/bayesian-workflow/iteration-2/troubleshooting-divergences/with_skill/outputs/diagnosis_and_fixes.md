# Diagnosis and Fixes: Divergences, High R-hat, and a Stuck Chain

## Symptom Summary

| Diagnostic | Observed Value | Threshold | Status |
|---|---|---|---|
| Divergences | 47 / 4000 samples | 0 (even 10+ can bias results) | FAIL |
| R-hat | 1.03 on two parameters | <= 1.01 | FAIL |
| Trace plots | One chain appears stuck | All chains should overlap and mix well | FAIL |

**Verdict: Do NOT interpret these results.** All three diagnostic failures point to the same underlying problem -- the sampler is struggling with the posterior geometry. The results are unreliable and must not be used for inference until these issues are resolved.

---

## Diagnosis

The three symptoms you describe are strongly correlated and almost certainly share a common root cause:

1. **47 divergences** means the NUTS sampler repeatedly encountered regions of high curvature in the posterior that it could not navigate. This is not a minor warning -- even a handful of divergences (10+) can systematically bias posterior estimates by under-exploring difficult regions of the parameter space.

2. **R-hat of 1.03 on two parameters** means those two parameters have not converged. R-hat measures between-chain vs. within-chain variance; a value above 1.01 indicates the chains are exploring different regions and have not agreed on a common stationary distribution. The R-hat threshold is 1.01 (not 1.05 or 1.1 as sometimes loosely quoted).

3. **One chain stuck** is the visual confirmation: that chain is trapped in a local mode or a region of the posterior that the other chains have moved past. This directly explains the high R-hat (chains disagree) and likely contributes to the divergences (the geometry near that stuck region is pathological).

### Most Likely Root Causes

These symptoms together point to one (or more) of the following:

- **Funnel geometry in a hierarchical model**: If your model has group-level parameters with a scale (standard deviation) parameter, a centered parameterization creates a "funnel" -- when the group SD is near zero, the group means must also collapse to zero, creating extreme curvature. This is the single most common cause of exactly this symptom pattern.

- **Poorly chosen priors on scale parameters**: Flat or vague priors on standard deviation parameters (e.g., `HalfCauchy`, `HalfFlat`) put substantial mass near zero, exacerbating funnels. Priors like `Gamma(2, ...)` or `Exponential(...)` that avoid the near-zero region work much better.

- **Multimodal posterior**: The stuck chain may have found a secondary mode. This can happen with weakly identified models or models with discrete symmetries (e.g., label switching in mixture models).

- **Insufficient warmup**: Less likely as the primary cause if other chains converged, but the stuck chain may have needed more tuning steps to escape its initial region.

---

## Recommended Fixes (In Priority Order)

Follow this sequence. Each step addresses a progressively deeper issue. After each fix, re-run sampling and re-check all diagnostics before moving on.

### Fix 1: Increase `target_accept`

This is the cheapest fix to try first. It makes the sampler take smaller steps, which helps it navigate high-curvature regions.

```python
idata = pm.sample(nuts_sampler="nutpie", target_accept=0.95, random_seed=rng)
```

If divergences persist, try up to 0.99:

```python
idata = pm.sample(nuts_sampler="nutpie", target_accept=0.99, random_seed=rng)
```

**Why this helps**: Higher `target_accept` forces the sampler to use a smaller step size, so it can follow tight curves in the posterior without overshooting. The trade-off is slower sampling, but correctness always trumps speed.

**When this is NOT enough**: If the geometry is fundamentally pathological (funnels), no step size will fully fix it. Move to Fix 2.

### Fix 2: Reparameterize (Non-Centered Parameterization)

If your model has hierarchical structure (group-level parameters drawn from a population distribution), switch from centered to non-centered parameterization. This is the most impactful fix for funnel-related divergences.

```python
# BEFORE (centered -- causes funnels):
mu_group = pm.Normal("mu_group", mu=mu_global, sigma=sigma_global, dims="group")

# AFTER (non-centered -- eliminates the funnel):
mu_raw = pm.Normal("mu_raw", mu=0, sigma=1, dims="group")
mu_group = pm.Deterministic("mu_group", mu_global + mu_raw * sigma_global, dims="group")
```

**Why this helps**: The non-centered form decouples `mu_raw` from `sigma_global`. In the centered form, when `sigma_global` is near zero, the sampler must navigate an impossibly narrow funnel. The non-centered form makes the geometry smooth and uniform regardless of `sigma_global`'s value.

**Rule of thumb**: Start with non-centered. Only switch to centered if non-centered shows poor ESS AND groups have substantial data (50+ observations each).

### Fix 3: Use Stronger Priors on Scale Parameters

If your group-level standard deviation has a flat or weakly informative prior, tighten it to avoid the near-zero region:

```python
# AVOID (puts mass near zero, creates funnels):
sigma_global = pm.HalfCauchy("sigma_global", beta=5)
sigma_global = pm.HalfFlat("sigma_global")

# PREFER (avoids the near-zero region):
sigma_global = pm.Gamma("sigma_global", alpha=2, beta=2)
# or
sigma_global = pm.Exponential("sigma_global", lam=1)
```

**Why this helps**: If there is no group-level variation to detect, you do not need the hierarchy. A prior that avoids zero prevents the sampler from wasting time exploring a region that is both geometrically pathological and scientifically uninteresting.

### Fix 4: Increase Warmup (Tuning Steps)

The stuck chain may need more warmup to find the typical set:

```python
idata = pm.sample(nuts_sampler="nutpie", tune=2000, random_seed=rng)
```

This is worth trying alongside Fix 1, but it rarely solves the problem on its own if the geometry is fundamentally difficult.

### Fix 5: Investigate Multimodality

If the fixes above do not resolve the stuck chain, the posterior may be multimodal. Use pair plots to investigate:

```python
# Visualize where divergences cluster -- focus on the two problematic parameters
az.plot_pair(idata, var_names=["param1", "param2"], divergences=True)
```

If you see distinct clusters, consider:
- Whether the model is identified (can the likelihood distinguish between modes?)
- Adding constraints or informative priors to break the symmetry
- Whether a mixture model with label switching is the issue (if so, impose an ordering constraint)

---

## Verification Checklist

After applying fixes, run the full diagnostic checklist before interpreting any results:

```python
# 1. Summary table
summary = az.summary(idata, round_to=3)
print(summary)

# 2. R-hat -- must be <= 1.01 for ALL parameters
rhat_ok = (summary["r_hat"] <= 1.01).all()
print(f"R-hat OK: {rhat_ok}")

# 3. ESS (bulk and tail) -- must be >= 100 * number of chains
num_chains = idata.posterior.sizes["chain"]
ess_bulk_ok = (summary["ess_bulk"] >= 100 * num_chains).all()
ess_tail_ok = (summary["ess_tail"] >= 100 * num_chains).all()
print(f"ESS bulk OK: {ess_bulk_ok}, ESS tail OK: {ess_tail_ok}")

# 4. Divergences -- must be 0
n_div = idata.sample_stats["diverging"].sum().item()
print(f"Divergences: {n_div}")

# 5. Visual check -- rank plots should look uniform, all chains overlapping
az.plot_trace(idata, kind="rank_vlines")

# 6. Energy diagnostic -- marginal energy and transition distributions should overlap
az.plot_energy(idata)
```

**All checks must pass before you proceed to model criticism or interpretation.**

---

## Key Takeaways

1. **Do not interpret the current results.** 47 divergences, R-hat of 1.03, and a stuck chain all indicate unreliable inference.
2. **Start with `target_accept=0.95`** (up to 0.99). This is the easiest fix and sometimes sufficient.
3. **Reparameterize to non-centered** if you have hierarchical structure. This is the most common and most impactful fix for this symptom pattern.
4. **Use `Gamma` or `Exponential` priors** on scale parameters to avoid the near-zero funnel region.
5. **Re-run all diagnostics** after every change. Only proceed when R-hat <= 1.01, ESS >= 100 * number of chains, and divergences = 0.
