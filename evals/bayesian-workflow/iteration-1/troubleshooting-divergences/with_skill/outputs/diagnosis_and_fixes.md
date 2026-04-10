# Diagnostic Assessment: 47 Divergences, R-hat 1.03, Stuck Chain

## Summary of Symptoms

| Diagnostic | Observed Value | Threshold | Status |
|---|---|---|---|
| Divergences | 47 / 4000 samples | 0 | FAIL |
| R-hat (max) | 1.03 on two parameters | <= 1.01 | FAIL |
| Trace plots | One chain stuck in a different region | Chains should overlap | FAIL |

**Verdict: Do NOT interpret these results.** All three symptoms point to the same underlying problem -- the sampler is unable to explore the posterior geometry reliably. The stuck chain and the high R-hat confirm that at least one chain has not converged to the same stationary distribution as the others. The 47 divergences (well above the "even 10+ can bias results" guideline) indicate the posterior has regions of high curvature the sampler cannot navigate.

---

## Diagnosis

### What is happening

The combination of divergences + a stuck chain + elevated R-hat on specific parameters is a classic signature of one of two problems (or both):

1. **A "funnel" geometry in a hierarchical model.** When a group-level standard deviation parameter can approach zero, the posterior forms a funnel shape. Chains that enter the narrow neck of the funnel produce divergences; chains that stay in the wide part of the funnel get stuck there. This is the single most common cause of this exact symptom profile.

2. **Multimodality or a ridge in the posterior.** If the posterior has two separated modes (or a narrow ridge connecting regions), one chain may find one mode and get stuck while the others explore the main mode. The divergences occur when the sampler tries (and fails) to traverse the ridge between modes.

### Identifying the problematic parameters

The two parameters with R-hat of 1.03 are the prime suspects. Before applying any fix, run these diagnostics to confirm where the problem lies:

```python
import arviz as az
import matplotlib.pyplot as plt

# 1. Summary table -- identify which parameters have R-hat > 1.01
summary = az.summary(idata, round_to=3)
problematic = summary[summary["r_hat"] > 1.01]
print("Parameters with R-hat > 1.01:")
print(problematic)

# 2. Rank plots -- the stuck chain will show as a non-uniform distribution
az.plot_trace(idata, var_names=problematic.index.tolist(), kind="rank_vlines")
plt.tight_layout()
plt.savefig("rank_plots_problematic.png", dpi=150)

# 3. Pair plot with divergences highlighted
# Focus on the two problematic parameters and any scale/SD parameters
az.plot_pair(
    idata,
    var_names=problematic.index.tolist(),
    divergences=True,
    divergences_kwargs={"color": "red", "marker": "x", "alpha": 0.5},
)
plt.tight_layout()
plt.savefig("pair_plot_divergences.png", dpi=150)

# 4. Energy diagnostic -- check for exploration problems
az.plot_energy(idata)
plt.savefig("energy_plot.png", dpi=150)

# 5. Count divergences per chain to confirm the stuck chain
for chain in range(idata.posterior.dims["chain"]):
    n_div = idata.sample_stats["diverging"].sel(chain=chain).sum().item()
    print(f"Chain {chain}: {n_div} divergences")
```

The pair plot is the most informative diagnostic here. If you see divergent samples (red markers) clustering near low values of a standard deviation parameter, that confirms the funnel geometry hypothesis.

---

## Recommended Fixes (in order of priority)

Apply these fixes sequentially. After each fix, re-run sampling and check diagnostics before moving to the next.

### Fix 1: Reparameterize to non-centered form

This is the highest-priority fix because it directly addresses the funnel geometry. If your model has any hierarchical structure (group-level parameters drawn from a population distribution), switch from centered to non-centered parameterization.

**Before (centered -- causes funnel divergences):**

```python
import pymc as pm

with pm.Model(coords=coords) as model_centered:
    group_idx = pm.Data("group_idx", group_id, dims="obs")
    y = pm.Data("y", observed_y, dims="obs")

    # Population-level
    mu_global = pm.Normal("mu_global", mu=0, sigma=10)
    sigma_global = pm.Gamma("sigma_global", alpha=2, beta=2)

    # Group-level -- CENTERED: drawn directly from population distribution
    mu_group = pm.Normal(
        "mu_group", mu=mu_global, sigma=sigma_global, dims="group"
    )

    sigma_obs = pm.Gamma("sigma_obs", alpha=2, beta=2)
    pm.Normal("likelihood", mu=mu_group[group_idx], sigma=sigma_obs,
              observed=y, dims="obs")
```

**After (non-centered -- eliminates the funnel):**

```python
import pymc as pm

with pm.Model(coords=coords) as model_noncentered:
    group_idx = pm.Data("group_idx", group_id, dims="obs")
    y = pm.Data("y", observed_y, dims="obs")

    # Population-level
    mu_global = pm.Normal("mu_global", mu=0, sigma=10)
    sigma_global = pm.Gamma("sigma_global", alpha=2, beta=2)

    # Group-level -- NON-CENTERED: standard normal offset, then scale
    mu_raw = pm.Normal("mu_raw", mu=0, sigma=1, dims="group")
    mu_group = pm.Deterministic(
        "mu_group", mu_global + mu_raw * sigma_global, dims="group"
    )

    sigma_obs = pm.Gamma("sigma_obs", alpha=2, beta=2)
    pm.Normal("likelihood", mu=mu_group[group_idx], sigma=sigma_obs,
              observed=y, dims="obs")
```

**Why this works:** The centered parameterization creates a statistical dependency between `mu_group` and `sigma_global`. When `sigma_global` is small, `mu_group` is tightly constrained -- creating the narrow neck of the funnel. The non-centered form breaks this dependency by having the sampler explore `mu_raw` (a standard normal) independently of `sigma_global`, then reconstructing `mu_group` deterministically.

**Rule of thumb:** Start with non-centered. Only switch back to centered if non-centered shows poor ESS AND groups have substantial data (50+ observations each).

---

### Fix 2: Increase `target_accept`

If reparameterization alone does not fully eliminate divergences, increase the target acceptance rate. This makes the sampler take smaller steps, allowing it to navigate tighter curvature at the cost of slower sampling.

```python
with model_noncentered:
    idata = pm.sample(
        nuts_sampler="nutpie",
        target_accept=0.95,  # default is 0.8; try 0.95 first, then 0.99
    )
```

Try this progression:
- `target_accept=0.95` -- usually sufficient after reparameterization
- `target_accept=0.99` -- for stubborn cases; sampling will be slower
- If 0.99 still has divergences, the problem is the model, not the sampler settings

---

### Fix 3: Strengthen priors on scale parameters

If the pair plot shows divergences clustering near zero for a standard deviation parameter, a prior that avoids near-zero values can help stabilize sampling. This is scientifically defensible: if there is truly no group-level variation, you should not be modeling it hierarchically.

```python
# Instead of a prior that allows near-zero values:
# sigma_global = pm.HalfNormal("sigma_global", sigma=5)  # has mass near 0

# Use a Gamma that stays away from zero:
sigma_global = pm.Gamma("sigma_global", alpha=2, beta=2)
# This has mode at 0.5 and very little mass near 0
```

Alternatively, if you have domain knowledge about the plausible range of group-level variation:

```python
# Weakly informative, stays away from 0
sigma_global = pm.Gamma("sigma_global", alpha=3, beta=1)
# mode at 2, mean at 3 -- adjust alpha/beta to match your domain
```

---

### Fix 4: Increase tuning steps and draws

The stuck chain may need more warmup to escape its initial region. This alone is unlikely to fix the fundamental geometry problem, but combined with the above fixes, it ensures the sampler has adequate adaptation time.

```python
with model_noncentered:
    idata = pm.sample(
        nuts_sampler="nutpie",
        tune=2000,          # double the default tuning (default is 1000)
        draws=2000,         # more draws for better ESS
        target_accept=0.95,
        chains=4,
    )
```

---

### Fix 5 (if applicable): Marginalize discrete parameters

If your model contains discrete latent variables (mixture indicators, change-point locations), NUTS cannot sample them directly, which can cause divergences and stuck chains. Marginalize them out when possible:

```python
# Instead of a discrete mixture indicator:
# z = pm.Categorical("z", p=[0.3, 0.7])

# Use a marginalized mixture:
pm.NormalMixture("obs", w=[0.3, 0.7], mu=[mu1, mu2], sigma=[s1, s2],
                 observed=data, dims="obs")
```

---

## Post-Fix Verification Checklist

After applying fixes, run the full diagnostic checklist before interpreting results:

```python
import pymc as pm
import arviz as az

# After re-sampling with the fixed model:
summary = az.summary(idata, round_to=3)
print(summary)

# 1. R-hat: all parameters must be <= 1.01
rhat_ok = (summary["r_hat"] <= 1.01).all()
print(f"R-hat OK (all <= 1.01): {rhat_ok}")

# 2. ESS: must be >= 100 * number of chains for both bulk and tail
num_chains = idata.posterior.dims["chain"]
ess_bulk_ok = (summary["ess_bulk"] >= 100 * num_chains).all()
ess_tail_ok = (summary["ess_tail"] >= 100 * num_chains).all()
print(f"ESS bulk OK: {ess_bulk_ok}, ESS tail OK: {ess_tail_ok}")

# 3. Divergences: must be 0
n_div = idata.sample_stats["diverging"].sum().item()
print(f"Divergences: {n_div} (target: 0)")

# 4. Visual checks
az.plot_trace(idata, kind="rank_vlines")
az.plot_energy(idata)

# 5. If hierarchical: check the funnel plot
az.plot_pair(
    idata,
    var_names=["sigma_global", "mu_group"],
    divergences=True,
)
```

**All three conditions must pass before interpreting results:**
- R-hat <= 1.01 on all parameters
- ESS bulk and tail >= 100 * number of chains
- Zero divergences

---

## Decision Tree Summary

```
47 divergences + R-hat 1.03 + stuck chain
│
├── Is the model hierarchical?
│   ├── YES → Apply Fix 1: Non-centered parameterization
│   │         Then re-check. Still divergences?
│   │         ├── YES → Apply Fix 2: target_accept=0.95
│   │         │         Still divergences?
│   │         │         ├── YES → Apply Fix 3: Stronger priors on SD params
│   │         │         │         + Fix 4: More tuning
│   │         │         └── NO → Run full verification checklist
│   │         └── NO → Run full verification checklist
│   │
│   └── NO → Check pair plots for problematic geometry
│           ├── Correlated parameters → Reparameterize to reduce correlation
│           ├── Multimodal → Consider mixture model or different parameterization
│           └── Ridge/constraint → Check for near-redundant parameters
│
└── After all fixes: Run posterior predictive checks
    to confirm the model reproduces the data
```

---

## Key Takeaways

1. **Do not interpret the current results.** With 47 divergences and R-hat of 1.03, the posterior summaries, credible intervals, and predictions are unreliable.

2. **Non-centered reparameterization is the most likely fix.** The symptom profile (divergences + stuck chain + elevated R-hat on specific parameters) strongly suggests a funnel geometry in a hierarchical model.

3. **Increasing `target_accept` is a complement, not a substitute, for reparameterization.** It helps the sampler navigate residual curvature but cannot fix fundamental geometry problems.

4. **Always re-run the full diagnostic checklist after applying fixes.** Do not assume the fix worked -- verify it.
