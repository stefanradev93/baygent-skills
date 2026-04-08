"""
Amortized Bayesian inference for linear regression with varying N.
y = alpha + beta * x + noise(sigma), N in [10, 200)
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bayesflow as bf

RANDOM_SEED = sum(map(ord, "regression-varying-n-v1"))
rng = np.random.default_rng(RANDOM_SEED)

def prior():
    alpha = np.random.normal(0.0, 5.0)
    beta = np.random.normal(0.0, 5.0)
    sigma = np.random.gamma(shape=2.0, scale=1.0)
    return dict(alpha=alpha, beta=beta, sigma=sigma)

def observation_model(alpha, beta, sigma, N):
    x = np.random.normal(0.0, 1.0, size=N)
    y = np.random.normal(alpha + beta * x, sigma, size=N)
    return dict(x=x, y=y)

def meta_fn():
    return dict(N=np.random.randint(10, 200))

simulator = bf.make_simulator([prior, observation_model], meta_fn=meta_fn)

summary_net = bf.networks.SetTransformer(
    summary_dim=6,
    embed_dims=(32, 32),
    num_heads=(2, 2),
    mlp_depths=(1, 1),
    mlp_widths=(64, 64),
)

inference_net = bf.networks.FlowMatching(
    subnet_kwargs={"widths": (128, 128, 128), "time_embedding_dim": 16}
)

adapter = (
    bf.Adapter()
    .broadcast("N", to="x")
    .as_set(["x", "y"])
    .constrain("sigma", lower=0)
    .sqrt("N")
    .convert_dtype("float64", "float32")
    .concatenate(["alpha", "beta", "sigma"], into="inference_variables")
    .concatenate(["x", "y"], into="summary_variables")
    .rename("N", "inference_conditions")
)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath="checkpoints",
    checkpoint_name="regression_varying_n",
)

history = workflow.fit_online(
    epochs=300, batch_size=32, num_batches_per_epoch=100,
    validation_data=300, verbose=2,
)

with open("history.json", "w") as f:
    json.dump(history.history, f)

sys.path.insert(0, ".")
from amortized_workflow.scripts.inspect_training import inspect_history
training_report = inspect_history(history.history)
print(json.dumps(training_report, indent=2))

test_data = workflow.simulate(300)

figures = workflow.plot_default_diagnostics(test_data=test_data)
for name, fig in figures.items():
    fig.savefig(f"diagnostics_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)
print(metrics.to_string())
metrics.to_csv("metrics.csv")

from amortized_workflow.scripts.check_diagnostics import check_diagnostics
diag_report = check_diagnostics(metrics)
print(json.dumps(diag_report, indent=2))

if diag_report["overall"]["decision"] == "STOP":
    raise RuntimeError(diag_report["overall"]["recommendation"])

rng_obs = np.random.default_rng(42)
N_obs = 80
x_obs = rng_obs.normal(0.0, 1.0, size=N_obs)
true_alpha, true_beta, true_sigma = 1.5, -0.8, 0.5
y_obs = rng_obs.normal(true_alpha + true_beta * x_obs, true_sigma)

real_data = {"x": x_obs, "y": y_obs, "N": np.array(N_obs)}
samples = workflow.sample(conditions=real_data, num_samples=1000)

alpha_samples = samples["alpha"]
beta_samples = samples["beta"]
sigma_samples = samples["sigma"]

print(f"alpha: {alpha_samples.mean():.3f} (true: {true_alpha})")
print(f"beta:  {beta_samples.mean():.3f} (true: {true_beta})")
print(f"sigma: {sigma_samples.mean():.3f} (true: {true_sigma})")

n_ppc = 50
for s in range(n_ppc):
    theta_s = {
        "alpha": float(alpha_samples[0, s]) if alpha_samples.ndim == 2 else float(alpha_samples[0, s, 0]),
        "beta": float(beta_samples[0, s]) if beta_samples.ndim == 2 else float(beta_samples[0, s, 0]),
        "sigma": float(sigma_samples[0, s]) if sigma_samples.ndim == 2 else float(sigma_samples[0, s, 0]),
    }
    rep = observation_model(**theta_s, N=N_obs)
