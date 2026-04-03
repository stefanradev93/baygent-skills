# Structural Causal Model: Job Training Program — Mediation Analysis

## Problem Statement

We want to decompose the **total causal effect** of job training on annual earnings into:

1. **Direct Effect (NDE)**: Training → Earnings (skills channel, bypassing confidence/interviews)
2. **Indirect Effect (NIE)**: Training → Confidence → Interview Performance → Earnings

The causal graph (DAG) is:

```
Training ──────────────────────────────▶ Earnings
    │                                       ▲
    └──▶ Confidence ──▶ Interview Perf ─────┘
```

This is a **sequential mediation** problem: two mediators in a chain (confidence → interview_performance), with a direct path from treatment to outcome.

---

## Approach: Structural Equation Modeling (SEM) with PyMC

We use a Bayesian SEM approach where each node in the DAG gets its own regression equation. This is sometimes called the **"potential outcomes + structural equations"** approach to mediation.

### Why Bayesian SEM?
- Propagates uncertainty through all equations simultaneously
- Gives full posterior distributions for direct and indirect effects
- Handles the chained mediation (two mediators) naturally

### Identification Assumptions
For this decomposition to be causally valid, we need:
1. **No unmeasured confounders** between treatment and mediators
2. **No unmeasured confounders** between treatment and outcome
3. **No unmeasured confounders** between mediators and outcome
4. **No treatment-mediator interaction** in its effect on outcome (or we explicitly model it)
5. **Sequential ignorability**: given training assignment, the mediators are "as good as randomly assigned"

These are strong assumptions. In an observational study, they would require careful justification (ideally with a randomized training assignment, which we assume here).

---

## Code

```python
import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATA
# ─────────────────────────────────────────────
# True structural parameters (what we want to recover):
#   - Direct effect of training on earnings: 3000
#   - Training → confidence: 1.5
#   - Confidence → interview perf: 0.8
#   - Interview perf → earnings: 2500
#   - So indirect effect = 1.5 * 0.8 * 2500 = 3000

RANDOM_SEED = sum(map(ord, "structural-mediation"))
rng = np.random.default_rng(RANDOM_SEED)

N = 1000

# Treatment assignment (assume RCT or as-if random)
training = rng.binomial(1, 0.5, size=N)

# Mediator 1: Confidence score
# training increases confidence by 1.5 units on average
confidence_score = (
    5.0                          # baseline
    + 1.5 * training             # treatment effect
    + rng.normal(0, 1.0, size=N) # noise
)

# Mediator 2: Interview performance
# confidence drives interview performance (0.8 per unit)
interview_performance = (
    3.0                                  # baseline
    + 0.8 * confidence_score             # confidence effect
    + rng.normal(0, 1.0, size=N)         # noise
)

# Outcome: Annual earnings
# Direct path (skills): 3000 per training unit
# Indirect path already captured through interview_performance
annual_earnings = (
    40000.0                               # baseline earnings
    + 3000.0 * training                   # direct skills effect
    + 2500.0 * interview_performance      # interview → earnings
    + rng.normal(0, 2000.0, size=N)       # noise
)

data = pd.DataFrame({
    "training": training,
    "confidence_score": confidence_score,
    "interview_performance": interview_performance,
    "annual_earnings": annual_earnings,
})

print("Data summary:")
print(data.describe().round(2))
print(f"\nCorr(training, earnings): {data['training'].corr(data['annual_earnings']):.3f}")

# ─────────────────────────────────────────────
# 2. STANDARDIZE CONTINUOUS VARIABLES
# ─────────────────────────────────────────────
# Standardizing makes priors easier to set and
# coefficients more interpretable on a common scale.

conf_mean, conf_std = data["confidence_score"].mean(), data["confidence_score"].std()
inter_mean, inter_std = data["interview_performance"].mean(), data["interview_performance"].std()
earn_mean, earn_std = data["annual_earnings"].mean(), data["annual_earnings"].std()

conf_z   = (data["confidence_score"] - conf_mean) / conf_std
inter_z  = (data["interview_performance"] - inter_mean) / inter_std
earn_z   = (data["annual_earnings"] - earn_mean) / earn_std
treat    = data["training"].values

# ─────────────────────────────────────────────
# 3. STRUCTURAL CAUSAL MODEL (BAYESIAN SEM)
# ─────────────────────────────────────────────
# We write one regression per endogenous node in the DAG:
#
#   M1 = a0 + a1*T + e_M1          (confidence equation)
#   M2 = b0 + b1*M1 + e_M2         (interview equation — M1 is the only parent)
#   Y  = c0 + c1*T + c2*M2 + e_Y   (earnings equation — direct + through M2)
#
# Note: T does NOT appear in M2's equation because the DAG says
# training only affects interview performance *through* confidence.
# If you believe training could also directly boost interview prep,
# add a b2*T term to M2's equation.

with pm.Model() as sem_model:

    # ── Equation 1: Confidence ~ Training ──────────────────────
    a0 = pm.Normal("a0", 0, 2)          # intercept
    a1 = pm.Normal("a1", 0, 2)          # training → confidence
    sigma_conf = pm.HalfNormal("sigma_conf", 1)

    mu_conf = a0 + a1 * treat
    conf_obs = pm.Normal("conf_obs", mu=mu_conf, sigma=sigma_conf, observed=conf_z)

    # ── Equation 2: Interview ~ Confidence ─────────────────────
    b0 = pm.Normal("b0", 0, 2)          # intercept
    b1 = pm.Normal("b1", 0, 2)          # confidence → interview
    sigma_inter = pm.HalfNormal("sigma_inter", 1)

    mu_inter = b0 + b1 * conf_z         # using observed conf (single-equation approach)
    inter_obs = pm.Normal("inter_obs", mu=mu_inter, sigma=sigma_inter, observed=inter_z)

    # ── Equation 3: Earnings ~ Training + Interview ─────────────
    c0 = pm.Normal("c0", 0, 2)          # intercept
    c1 = pm.Normal("c1", 0, 2)          # training → earnings (direct/skills effect)
    c2 = pm.Normal("c2", 0, 2)          # interview → earnings
    sigma_earn = pm.HalfNormal("sigma_earn", 1)

    mu_earn = c0 + c1 * treat + c2 * inter_z
    earn_obs = pm.Normal("earn_obs", mu=mu_earn, sigma=sigma_earn, observed=earn_z)

    # ── Derived quantities: effect decomposition ────────────────
    # All on the standardized scale first, then convert to dollars.
    #
    # Indirect effect (standardized):
    #   Training → Confidence → Interview → Earnings
    #   = a1 (conf_std units) → b1 (inter_std units) → c2 (earn_std units)
    #   = a1 * (conf_std / inter_std) ... but because we're using observed
    #   mediators with z-scores, the chain is:
    #   NIE_z = a1 * b1 * c2   (all in SD units of their respective outcomes)
    #
    # To convert to dollars:
    #   NIE_$ = NIE_z * earn_std  (in dollars, per unit increase in training)
    # Similarly for direct:
    #   NDE_$ = c1 * earn_std
    #
    # BUT: b1 links conf_z → inter_z, so the indirect path is:
    #   training → conf_z: coefficient a1 (i.e., a1 SDs of conf per treatment)
    #   conf_z → inter_z:  coefficient b1 (i.e., b1 SDs of inter per SD of conf)
    #   inter_z → earn_z:  coefficient c2
    #   NIE_z = a1 * b1 * c2  (this is the change in earn_z per treatment unit)

    NIE_z = pm.Deterministic("NIE_z", a1 * b1 * c2)
    NDE_z = pm.Deterministic("NDE_z", c1)
    ATE_z = pm.Deterministic("ATE_z", NDE_z + NIE_z)

    # Convert to original dollar scale
    NIE_dollars = pm.Deterministic("NIE_dollars", NIE_z * earn_std)
    NDE_dollars = pm.Deterministic("NDE_dollars", NDE_z * earn_std)
    ATE_dollars = pm.Deterministic("ATE_dollars", ATE_z * earn_std)

    # Proportion mediated (indirect / total)
    prop_mediated = pm.Deterministic("prop_mediated", NIE_z / ATE_z)

    # ── Sample ──────────────────────────────────────────────────
    idata = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        random_seed=RANDOM_SEED,
        target_accept=0.9,
        progressbar=True,
    )

# ─────────────────────────────────────────────
# 4. DIAGNOSTICS
# ─────────────────────────────────────────────
print("\n=== MCMC Diagnostics ===")
summary = az.summary(
    idata,
    var_names=["a1", "b1", "c1", "c2",
               "NIE_dollars", "NDE_dollars", "ATE_dollars", "prop_mediated"],
    round_to=2,
)
print(summary)

# Check R-hat and ESS
rhat_max = summary["r_hat"].max()
ess_min  = summary["ess_bulk"].min()
print(f"\nMax R-hat: {rhat_max:.3f} (want < 1.01)")
print(f"Min ESS bulk: {ess_min:.0f} (want > 400)")

# ─────────────────────────────────────────────
# 5. RESULTS & INTERPRETATION
# ─────────────────────────────────────────────
post = idata.posterior

nie = post["NIE_dollars"].values.flatten()
nde = post["NDE_dollars"].values.flatten()
ate = post["ATE_dollars"].values.flatten()
pm_  = post["prop_mediated"].values.flatten()

def hdi_summary(samples, name, true_val=None):
    mean = samples.mean()
    hdi  = az.hdi(samples, hdi_prob=0.94)
    line = f"{name}: ${mean:,.0f}  [94% HDI: ${hdi[0]:,.0f} — ${hdi[1]:,.0f}]"
    if true_val is not None:
        line += f"  (true: ${true_val:,.0f})"
    return line

# True indirect effect in dollars = 1.5 * 0.8 * 2500 = 3000
# True direct effect = 3000
# True total = 6000
print("\n=== EFFECT DECOMPOSITION ===")
print(hdi_summary(nde, "Direct Effect (NDE) — skills channel",  true_val=3000))
print(hdi_summary(nie, "Indirect Effect (NIE) — confidence→interview channel", true_val=3000))
print(hdi_summary(ate, "Total Effect (ATE)",                    true_val=6000))
print(f"Proportion mediated: {pm_.mean():.1%}  [94% HDI: {az.hdi(pm_, hdi_prob=0.94)[0]:.1%} — {az.hdi(pm_, hdi_prob=0.94)[1]:.1%}]  (true: 50%)")

# ─────────────────────────────────────────────
# 6. PLOT
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Effect Decomposition: Job Training → Annual Earnings", fontsize=13)

for ax, samples, label, true_val, color in zip(
    axes,
    [nde, nie, ate],
    ["Direct Effect\n(skills channel)", "Indirect Effect\n(confidence→interview)", "Total Effect"],
    [3000, 3000, 6000],
    ["steelblue", "coral", "seagreen"],
):
    ax.hist(samples, bins=50, color=color, alpha=0.7, density=True)
    ax.axvline(samples.mean(), color="black", lw=2, label=f"Posterior mean: ${samples.mean():,.0f}")
    ax.axvline(true_val, color="red", lw=2, linestyle="--", label=f"True value: ${true_val:,.0f}")
    hdi = az.hdi(samples, hdi_prob=0.94)
    ax.axvspan(hdi[0], hdi[1], alpha=0.15, color=color, label="94% HDI")
    ax.set_xlabel("Annual earnings ($)")
    ax.set_ylabel("Posterior density")
    ax.set_title(label)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/causal-inference-workspace/iteration-1/structural-mediation/without_skill/outputs/effect_decomposition.png",
    dpi=150,
    bbox_inches="tight",
)
print("\nPlot saved.")
```

---

## Interpretation Guide

### Structural Equations

| Equation | Meaning |
|---|---|
| `Confidence = a0 + a1*Training + e` | How much does training boost confidence? (`a1`) |
| `Interview = b0 + b1*Confidence + e` | How much does confidence improve interviews? (`b1`) |
| `Earnings = c0 + c1*Training + c2*Interview + e` | Direct skills effect (`c1`) and interview-to-earnings conversion (`c2`) |

### Effect Decomposition

- **Natural Direct Effect (NDE)** = `c1 * earn_std` in dollars
  This is the earnings increase from training holding the confidence/interview channel fixed — the "pure skills" channel.

- **Natural Indirect Effect (NIE)** = `a1 * b1 * c2 * earn_std` in dollars
  This is the earnings increase *through* the confidence → interview pathway.

- **Average Total Effect (ATE)** = NDE + NIE

- **Proportion Mediated** = NIE / ATE — how much of the training benefit flows through the confidence channel vs direct skills.

### What to do if assumptions fail

| Assumption | Diagnostic / Fix |
|---|---|
| Unmeasured confounding (training ↔ outcome) | Sensitivity analysis (e-value), IV if available |
| Unmeasured confounding (mediator ↔ outcome) | Especially tricky — use expert knowledge or partial identification bounds |
| Treatment-mediator interaction | Add `training * confidence_z` term to earnings equation; interpret NDE/NIE using marginal effects |
| Non-linearity | Replace linear equations with splines or GP priors |

### Key Caveats

1. **Single-equation mediator approach**: We condition on *observed* mediator values rather than jointly modeling them. For a fully joint model, you'd want to use `pm.MvNormal` or explicitly model counterfactual distributions. The approach above recovers the same point estimates but may slightly under-represent uncertainty in the indirect effect.

2. **Identification of NIE requires no treatment-mediator interaction** in the outcome equation (or explicit modeling of it). If training *directly* boosts interview performance (not just through confidence), the model as written would misattribute that to the indirect channel.

3. **Randomized training** is assumed here. With observational data, you'd need to condition on confounders in every equation (or use a front-door / instrumental variable strategy).

---

## Expected Results (with the synthetic data above)

| Effect | True Value | What You Should Recover |
|---|---|---|
| Direct (NDE) | $3,000 | ~$3,000 ± uncertainty |
| Indirect (NIE) | $3,000 (= 1.5 × 0.8 × 2,500) | ~$3,000 ± uncertainty |
| Total (ATE) | $6,000 | ~$6,000 ± uncertainty |
| Proportion mediated | 50% | ~50% |

The model should recover these values well because the synthetic data was generated from the exact structural equations the model assumes — a useful sanity check before applying to real data.
