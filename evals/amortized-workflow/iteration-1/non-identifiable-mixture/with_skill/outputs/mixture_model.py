"""
Amortized Bayesian inference for a two-component Gaussian mixture model.

Generative model:
    w ~ Beta(2, 2)                   (mixing weight, bounded (0, 1))
    mu1 ~ Normal(0, 5)               (mean of component 1)
    mu2 ~ Normal(0, 5)               (mean of component 2)
    sigma ~ Gamma(2, 1)              (shared std, bounded > 0)
    For each observation i = 1..N:
        z_i ~ Bernoulli(w)
        x_i | z_i ~ Normal(mu_{z_i+1}, sigma)

This model is intentionally non-identifiable when w ~ 0.5: swapping
(w, mu1, mu2) -> (1-w, mu2, mu1) yields the same likelihood. The
diagnostics should reveal this as poor recovery / high NRMSE for mu1
and mu2, while w and sigma may recover well.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

import bayesflow as bf

# Add the skill scripts to the path
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..", "..",
        "baygent-skills", "amortized-workflow",
    ),
)
from scripts.check_diagnostics import check_diagnostics
from scripts.inspect_training import inspect_history

# ── Reproducibility ──────────────────────────────────────────────
RANDOM_SEED = sum(map(ord, "mixture-label-switching-v1"))
rng = np.random.default_rng(RANDOM_SEED)

# Output directory (same folder as this script)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==================================================================
# 1. Define prior + observation model (online training)
# ==================================================================

def prior():
    """Sample mixture parameters from the prior."""
    w = rng.beta(2, 2)               # mixing weight in (0, 1)
    mu1 = rng.normal(0, 5)           # mean of component 1
    mu2 = rng.normal(0, 5)           # mean of component 2
    sigma = rng.gamma(2, 1)          # shared std (> 0)
    return dict(w=w, mu1=mu1, mu2=mu2, sigma=sigma)


def observation_model(w, mu1, mu2, sigma, N):
    """Generate N i.i.d. draws from the two-component Gaussian mixture."""
    z = rng.binomial(1, w, size=N)
    x = np.where(
        z == 1,
        rng.normal(mu1, sigma, size=N),
        rng.normal(mu2, sigma, size=N),
    )
    return dict(x=x)


def meta_fn():
    """Vary the number of observations between batches."""
    return dict(N=rng.integers(30, 200))


simulator = bf.make_simulator([prior, observation_model], meta_fn=meta_fn)


# ==================================================================
# 2. Simulation sanity check
# ==================================================================

print("=== Simulation sanity check ===")
test_sim = simulator.sample(5)
for key, val in test_sim.items():
    shape = val.shape if hasattr(val, "shape") else type(val)
    print(f"  {key}: {shape}")
print()


# ==================================================================
# 3. Choose architecture (SMALL config from model-sizes.md)
# ==================================================================

# 4 parameters -> summary_dim = 2 * 4 = 8
summary_net = bf.networks.SetTransformer(
    summary_dim=8,
    embed_dims=(32, 32),
    num_heads=(2, 2),
    mlp_depths=(1, 1),
    mlp_widths=(64, 64),
)

# Small inference network
inference_net = bf.networks.FlowMatching(
    subnet_kwargs={"widths": (128, 128), "time_embedding_dim": 16}
)


# ==================================================================
# 4. Build the adapter
# ==================================================================

adapter = (
    bf.Adapter()
    .broadcast("N", to="x")
    .as_set(["x"])                                # (N,) -> (N, 1) for SetTransformer
    .constrain("w", lower=0, upper=1)             # mixing weight in (0, 1)
    .constrain("sigma", lower=0)                  # std must be positive
    .sqrt("N")                                    # feature-engineer sample size
    .convert_dtype("float64", "float32")
    .concatenate(["w", "mu1", "mu2", "sigma"], into="inference_variables")
    .concatenate(["x"], into="summary_variables")
    .rename("N", "inference_conditions")
)


# ==================================================================
# 5. Create workflow
# ==================================================================

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath=os.path.join(OUT_DIR, "checkpoints"),
    checkpoint_name="mixture_model",
)


# ==================================================================
# 6. Train the amortizer
# ==================================================================

print("=== Training ===")
history = workflow.fit_online(
    epochs=150,
    batch_size=32,
    num_batches_per_epoch=100,
    validation_data=300,
    verbose=2,
)

# ── Mandatory: save history and inspect convergence ──────────────
history_path = os.path.join(OUT_DIR, "history.json")
with open(history_path, "w") as f:
    json.dump(history.history, f)

training_report = inspect_history(history.history)
print("\n=== Training Report ===")
print(json.dumps(training_report, indent=2))

if not training_report["overall"]["ok"]:
    print("\nTRAINING ISSUES -- address before continuing:")
    for issue in training_report["overall"]["issues"]:
        print(f"  - {issue}")


# ==================================================================
# 7. In-silico diagnostics on held-out simulations
# ==================================================================

print("\n=== Diagnostics ===")

# MUST use workflow.simulate() -- NEVER loop over simulator() manually
test_data = workflow.simulate(300)

# plot_default_diagnostics ALWAYS returns a dict[str, Figure]
figures = workflow.plot_default_diagnostics(test_data=test_data)
for name, fig in figures.items():
    fig_path = os.path.join(OUT_DIR, f"diagnostics_{name}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: diagnostics_{name}.png")

metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)

# ── Mandatory: save diagnostics and check house thresholds ───────
metrics_path = os.path.join(OUT_DIR, "metrics.csv")
metrics.to_csv(metrics_path)
print("\n=== Diagnostics Table ===")
print(metrics.to_string())
print()

diag_report = check_diagnostics(metrics)
diag_report_path = os.path.join(OUT_DIR, "diagnostics_report.json")
with open(diag_report_path, "w") as f:
    json.dump(diag_report, f, indent=2)

print("=== Diagnostics Report ===")
print(json.dumps(diag_report, indent=2))


# ==================================================================
# 8. Interpret diagnostics per-parameter
# ==================================================================

print("\n=== Per-parameter Interpretation ===")
for param, info in diag_report["parameters"].items():
    print(f"\n  {param}:")
    for v in info["verdicts"]:
        print(f"    {v}")

decision = diag_report["overall"]["decision"]
print(f"\n  Overall decision: {decision}")
print(f"  Recommendation: {diag_report['overall']['recommendation']}")


# ==================================================================
# 9. Go / no-go gate
# ==================================================================

if decision == "STOP":
    print(
        "\n*** STOPPING: Diagnostics indicate non-identifiability or poor recovery. ***\n"
        "This is EXPECTED for a label-switching mixture model.\n"
        "See analysis_notes.md for interpretation and remediation strategies.\n"
        "Do NOT proceed to real-data inference with this model."
    )
    # We intentionally do NOT raise here so the script completes and
    # saves all artifacts. But we clearly flag the problem.
elif decision == "WARN":
    print(
        "\n*** WARNING: Some parameters have marginal diagnostics. ***\n"
        "Likely mu1 and mu2 showing weak recovery due to label switching.\n"
        "See analysis_notes.md for interpretation."
    )
else:
    print(
        "\n*** GO: All diagnostics within acceptable bounds. ***\n"
        "This would be surprising for the unconstrained mixture --\n"
        "check whether w was far from 0.5 in most test simulations."
    )


# ==================================================================
# 10. Summary
# ==================================================================

print("\n=== Simulation Budget ===")
print(f"  Training: {150} epochs x {100} batches x {32} sims/batch = {150 * 100 * 32:,} sims")
print(f"  Validation: 300 sims (auto-simulated)")
print(f"  Diagnostics: 300 held-out sims")
print(f"\n  Architecture: SetTransformer (Small) + FlowMatching (Small)")
print(f"  Training mode: online")
print(f"  All outputs saved to: {OUT_DIR}")
