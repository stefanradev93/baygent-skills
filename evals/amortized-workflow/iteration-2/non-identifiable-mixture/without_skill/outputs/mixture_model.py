"""
Amortized Bayesian inference for a two-component Gaussian mixture model.

Generative model:
    w     ~ Beta(2, 2)             (mixing weight, bounded (0, 1))
    mu1   ~ Normal(0, 5)           (mean of component 1)
    mu2   ~ Normal(0, 5)           (mean of component 2)
    sigma ~ Gamma(2, 1)            (shared std, bounded > 0)
    For each observation i = 1..N:
        z_i ~ Bernoulli(w)
        x_i | z_i ~ Normal(mu_{z_i+1}, sigma)

This model is intentionally non-identifiable when w ~ 0.5: swapping
(w, mu1, mu2) -> (1-w, mu2, mu1) yields the same likelihood (label
switching). We train a BayesFlow amortized estimator and use diagnostics
to expose which parameters the network cannot reliably recover.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import json

import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 0. Reproducibility and paths
# ---------------------------------------------------------------------------

RANDOM_SEED = sum(map(ord, "mixture-label-switching-v1"))
rng = np.random.default_rng(RANDOM_SEED)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. Define prior + observation model
# ---------------------------------------------------------------------------


def prior(rng=None):
    """Sample mixture parameters from the prior."""
    rng = rng or np.random.default_rng()
    w = rng.beta(2, 2)
    mu1 = rng.normal(0, 5)
    mu2 = rng.normal(0, 5)
    sigma = rng.gamma(2, 1)
    return dict(
        w=np.array(w),
        mu1=np.array(mu1),
        mu2=np.array(mu2),
        sigma=np.array(sigma),
    )


def observation_model(w, mu1, mu2, sigma, rng=None):
    """Generate 80 i.i.d. draws from the two-component Gaussian mixture."""
    rng = rng or np.random.default_rng()
    N = 80
    z = rng.binomial(1, w, size=N)
    x = np.where(
        z == 1,
        rng.normal(mu1, sigma, size=N),
        rng.normal(mu2, sigma, size=N),
    )
    return dict(x=x)


simulator = bf.make_simulator([prior, observation_model])

# ---------------------------------------------------------------------------
# 2. Simulation sanity check
# ---------------------------------------------------------------------------

print("=== Simulation sanity check ===")
test_sims = simulator.sample(5, rng=rng)
for key, val in test_sims.items():
    print(f"  {key}: shape={np.shape(val)}, dtype={np.asarray(val).dtype}")
for i in range(3):
    w_i = float(test_sims["w"][i])
    mu1_i = float(test_sims["mu1"][i])
    mu2_i = float(test_sims["mu2"][i])
    sigma_i = float(test_sims["sigma"][i])
    x_i = test_sims["x"][i]
    print(
        f"  sim {i}: w={w_i:.2f}, mu1={mu1_i:.2f}, mu2={mu2_i:.2f}, "
        f"sigma={sigma_i:.2f}, x_mean={np.mean(x_i):.2f}, x_std={np.std(x_i):.2f}"
    )
print()

# ---------------------------------------------------------------------------
# 3. Architecture -- 4 parameters => summary_dim = 8
# ---------------------------------------------------------------------------

summary_net = bf.networks.SetTransformer(
    summary_dim=8,
    embed_dims=(32, 32),
    num_heads=(2, 2),
    mlp_depths=(1, 1),
    mlp_widths=(64, 64),
)

inference_net = bf.networks.FlowMatching(
    subnet_kwargs={"widths": (128, 128), "time_embedding_dim": 16},
)

# ---------------------------------------------------------------------------
# 4. Adapter
# ---------------------------------------------------------------------------

adapter = (
    bf.Adapter()
    .as_set(["x"])                            # (80,) -> (80, 1) for SetTransformer
    .constrain("w", lower=0, upper=1)         # mixing weight in (0, 1)
    .constrain("sigma", lower=0)              # std must be positive
    .convert_dtype("float64", "float32")
    .concatenate(["x"], into="summary_variables")
    .concatenate(["w", "mu1", "mu2", "sigma"], into="inference_variables")
)

# ---------------------------------------------------------------------------
# 5. Workflow
# ---------------------------------------------------------------------------

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath=os.path.join(OUT_DIR, "checkpoints"),
    checkpoint_name="mixture_model",
)

# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------

print("=== Training ===")
history = workflow.fit_online(
    epochs=120,
    batch_size=32,
    num_batches_per_epoch=100,
    validation_data=300,
    verbose=2,
)

# Save training history
history_path = os.path.join(OUT_DIR, "history.json")
with open(history_path, "w") as f:
    json.dump(history.history, f)
print(f"\nTraining history saved to {history_path}")

# Quick convergence check: compare first vs last 10 epochs
train_loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])
if len(train_loss) >= 20:
    early = np.mean(train_loss[:10])
    late = np.mean(train_loss[-10:])
    print(f"  Train loss: first 10 epochs avg = {early:.4f}, last 10 epochs avg = {late:.4f}")
    if late >= early:
        print("  WARNING: Training loss did not decrease -- check learning rate or architecture.")
if len(val_loss) >= 20:
    early_val = np.mean(val_loss[:10])
    late_val = np.mean(val_loss[-10:])
    print(f"  Val loss:   first 10 epochs avg = {early_val:.4f}, last 10 epochs avg = {late_val:.4f}")
    if late_val > early_val * 1.1:
        print("  WARNING: Possible overfitting -- val loss increased significantly.")

# ---------------------------------------------------------------------------
# 7. In-silico diagnostics
# ---------------------------------------------------------------------------

print("\n=== In-silico diagnostics ===")
test_data = workflow.simulate(300)

# Visual diagnostics (recovery plots, calibration, etc.)
figures = workflow.plot_default_diagnostics(test_data=test_data)
for name, fig in figures.items():
    fig_path = os.path.join(OUT_DIR, f"diagnostics_{name}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: diagnostics_{name}.png")

# Numerical diagnostics (NRMSE, R2, calibration error, etc.)
metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)
metrics_path = os.path.join(OUT_DIR, "metrics.csv")
metrics.to_csv(metrics_path)
print(f"\nDiagnostics table:\n{metrics}\n")

# ---------------------------------------------------------------------------
# 8. Interpret diagnostics per-parameter
# ---------------------------------------------------------------------------

print("=== Per-parameter interpretation ===")
print(
    "We expect the following pattern for an unconstrained mixture model:\n"
    "  - w:     Moderate to good recovery (NRMSE < 0.5, R2 > 0.5)\n"
    "  - mu1:   POOR recovery (high NRMSE, low R2) due to label switching\n"
    "  - mu2:   POOR recovery (high NRMSE, low R2) due to label switching\n"
    "  - sigma: Good recovery (NRMSE < 0.3, R2 > 0.7) -- shared, not affected\n"
    "\n"
    "If mu1 and mu2 show R2 near 0 or negative, this confirms the network\n"
    "cannot distinguish which component is 'first' vs 'second'. The posterior\n"
    "is bimodal for these parameters, and the point-estimate metrics collapse.\n"
)

# Parse metrics to flag problematic parameters
param_names = ["w", "mu1", "mu2", "sigma"]
print("Parameter-level assessment:")
for param in param_names:
    row = metrics.loc[metrics.index.str.contains(param, case=False)]
    if row.empty:
        print(f"  {param}: not found in metrics table")
        continue
    row = row.iloc[0]
    # Try to read common diagnostic columns
    r2 = row.get("R2", row.get("r2", None))
    nrmse = row.get("NRMSE", row.get("nrmse", None))

    status_parts = []
    if r2 is not None:
        if r2 < 0.3:
            status_parts.append(f"R2={r2:.3f} [POOR]")
        elif r2 < 0.7:
            status_parts.append(f"R2={r2:.3f} [MARGINAL]")
        else:
            status_parts.append(f"R2={r2:.3f} [GOOD]")

    if nrmse is not None:
        if nrmse > 0.8:
            status_parts.append(f"NRMSE={nrmse:.3f} [POOR]")
        elif nrmse > 0.5:
            status_parts.append(f"NRMSE={nrmse:.3f} [MARGINAL]")
        else:
            status_parts.append(f"NRMSE={nrmse:.3f} [GOOD]")

    if status_parts:
        print(f"  {param}: {', '.join(status_parts)}")
    else:
        print(f"  {param}: {row.to_dict()}")

# ---------------------------------------------------------------------------
# 9. Go / no-go gate
# ---------------------------------------------------------------------------

print("\n=== Go / no-go decision ===")

# Check if mu1 or mu2 have poor recovery
has_poor_recovery = False
for param in ["mu1", "mu2"]:
    row = metrics.loc[metrics.index.str.contains(param, case=False)]
    if not row.empty:
        r2 = row.iloc[0].get("R2", row.iloc[0].get("r2", None))
        if r2 is not None and r2 < 0.3:
            has_poor_recovery = True

if has_poor_recovery:
    print(
        "STOP: mu1 and/or mu2 show poor recovery (R2 < 0.3).\n"
        "This confirms label switching non-identifiability.\n"
        "\n"
        "Remediation options (see analysis_notes.md for details):\n"
        "  1. Impose an ordering constraint: mu1 < mu2\n"
        "  2. Use a post-hoc relabeling algorithm on posterior samples\n"
        "  3. Report results in terms of identifiable quantities\n"
        "     (e.g., sorted means, or component separation |mu1 - mu2|)\n"
        "\n"
        "Do NOT proceed to real-data inference with unconstrained mu1, mu2."
    )
else:
    print(
        "All parameters show acceptable recovery.\n"
        "This could happen if w is consistently far from 0.5 in the prior,\n"
        "making label switching less severe. Check the w prior."
    )

# ---------------------------------------------------------------------------
# 10. Inference on a synthetic test dataset
# ---------------------------------------------------------------------------

print("\n=== Inference on test dataset ===")

# Ground truth
true_w = 0.5   # maximally ambiguous
true_mu1 = -2.0
true_mu2 = 3.0
true_sigma = 1.0

# Generate test data
z_test = rng.binomial(1, true_w, size=80)
x_obs = np.where(
    z_test == 1,
    rng.normal(true_mu1, true_sigma, size=80),
    rng.normal(true_mu2, true_sigma, size=80),
)

print(f"Ground truth: w={true_w}, mu1={true_mu1}, mu2={true_mu2}, sigma={true_sigma}")
print(f"Observed: mean={np.mean(x_obs):.3f}, std={np.std(x_obs):.3f}")

# Amortized posterior samples
real_data = {"x": x_obs}
samples = workflow.sample(conditions=real_data, num_samples=2000)

for param in ["w", "mu1", "mu2", "sigma"]:
    s = samples[param][0, :, 0]  # (1, 2000, 1) -> (2000,)
    print(
        f"  {param}: mean={np.mean(s):.3f}, std={np.std(s):.3f}, "
        f"95% CI=[{np.percentile(s, 2.5):.3f}, {np.percentile(s, 97.5):.3f}]"
    )

# ---------------------------------------------------------------------------
# 11. Visualize posterior for mu1 and mu2 to show label switching
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# mu1 vs mu2 joint posterior -- should show bimodal "X" pattern
ax = axes[0, 0]
mu1_s = samples["mu1"][0, :, 0]
mu2_s = samples["mu2"][0, :, 0]
ax.scatter(mu1_s, mu2_s, alpha=0.15, s=5, c="steelblue")
ax.axvline(true_mu1, color="red", linestyle="--", label="true mu1")
ax.axhline(true_mu2, color="red", linestyle="--", label="true mu2")
ax.axvline(true_mu2, color="orange", linestyle=":", alpha=0.7, label="swapped mu1")
ax.axhline(true_mu1, color="orange", linestyle=":", alpha=0.7, label="swapped mu2")
ax.set_xlabel("mu1 posterior")
ax.set_ylabel("mu2 posterior")
ax.set_title("Joint posterior: mu1 vs mu2\n(bimodal = label switching)")
ax.legend(fontsize=8)

# Marginal mu1
ax = axes[0, 1]
ax.hist(mu1_s, bins=50, alpha=0.6, density=True, color="steelblue")
ax.axvline(true_mu1, color="red", linewidth=2, label=f"true ({true_mu1})")
ax.axvline(true_mu2, color="orange", linewidth=2, linestyle=":", label=f"swapped ({true_mu2})")
ax.set_title("Marginal posterior: mu1")
ax.legend()

# Marginal mu2
ax = axes[1, 0]
ax.hist(mu2_s, bins=50, alpha=0.6, density=True, color="steelblue")
ax.axvline(true_mu2, color="red", linewidth=2, label=f"true ({true_mu2})")
ax.axvline(true_mu1, color="orange", linewidth=2, linestyle=":", label=f"swapped ({true_mu1})")
ax.set_title("Marginal posterior: mu2")
ax.legend()

# w and sigma marginals
ax = axes[1, 1]
w_s = samples["w"][0, :, 0]
sigma_s = samples["sigma"][0, :, 0]
ax.hist(w_s, bins=40, alpha=0.5, density=True, label="w", color="green")
ax.axvline(true_w, color="green", linewidth=2, linestyle="--")
ax.hist(sigma_s, bins=40, alpha=0.5, density=True, label="sigma", color="purple")
ax.axvline(true_sigma, color="purple", linewidth=2, linestyle="--")
ax.set_title("Marginal posteriors: w and sigma")
ax.legend()

plt.suptitle("Posterior diagnostics for non-identifiable mixture", fontsize=14)
plt.tight_layout()
posterior_path = os.path.join(OUT_DIR, "posterior_label_switching.png")
fig.savefig(posterior_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPosterior visualization saved to {posterior_path}")

# ---------------------------------------------------------------------------
# 12. Summary
# ---------------------------------------------------------------------------

print("\n=== Simulation Budget ===")
total_train = 120 * 100 * 32
print(f"  Training: 120 epochs x 100 batches x 32 sims/batch = {total_train:,} sims")
print(f"  Validation: 300 sims")
print(f"  Diagnostics: 300 held-out sims")
print(f"  Architecture: SetTransformer (Small) + FlowMatching (Small)")
print(f"  All outputs saved to: {OUT_DIR}")
print("\n=== Done ===")
