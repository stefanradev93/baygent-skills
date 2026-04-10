"""
Gaussian location-scale amortized inference with BayesFlow.

Model: x_i ~ Normal(mu, sigma), i = 1, ..., N=50
Parameters: mu (unconstrained), sigma (positive)
Training: online
"""

import json
import sys
import os

import bayesflow as bf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = sum(map(ord, "gaussian-location-scale-v1"))
rng = np.random.default_rng(RANDOM_SEED)

N_OBS = 50

def prior():
    mu = np.random.normal(0.0, 5.0)
    sigma = np.random.gamma(shape=2.0, scale=1.0)
    return {"mu": mu, "sigma": sigma}

def observation_model(mu, sigma):
    x = np.random.normal(loc=mu, scale=sigma, size=N_OBS)
    return {"x": x}

simulator = bf.make_simulator([prior, observation_model])

_batch = simulator.sample(4)
print("Simulator sanity check — keys:", list(_batch.keys()))

summary_net = bf.networks.SetTransformer(
    summary_dim=4,
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
    .as_set(["x"])
    .constrain("sigma", lower=0)
    .convert_dtype("float64", "float32")
    .concatenate(["x"], into="summary_variables")
    .concatenate(["mu", "sigma"], into="inference_variables")
)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath="checkpoints",
    checkpoint_name="gaussian_location_scale",
)

history = workflow.fit_online(
    epochs=200,
    batch_size=64,
    num_batches_per_epoch=100,
    validation_data=300,
    verbose=2,
)

with open("history.json", "w") as f:
    json.dump(history.history, f)

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "amortized-workflow", "scripts"))

from inspect_training import inspect_history
from check_diagnostics import check_diagnostics

training_report = inspect_history(history.history)
print(json.dumps(training_report, indent=2))

if not training_report["overall"]["ok"]:
    for issue in training_report["overall"]["issues"]:
        print(f"  - {issue}")

test_data = workflow.simulate(500)

figures = workflow.plot_default_diagnostics(test_data=test_data)
for name, fig in figures.items():
    fig.savefig(f"diagnostics_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)
print(metrics.to_string())
metrics.to_csv("metrics.csv")

diag_report = check_diagnostics(metrics)
print(json.dumps(diag_report, indent=2))

if diag_report["overall"]["decision"] == "STOP":
    raise RuntimeError(diag_report["overall"]["recommendation"])

test_sim = workflow.simulate(1)
x_obs = test_sim["x"]

samples = workflow.sample(conditions={"x": x_obs}, num_samples=2000)

mu_samples = np.squeeze(samples["mu"])
sigma_samples = np.squeeze(samples["sigma"])
true_mu = float(np.squeeze(test_sim["mu"]))
true_sigma = float(np.squeeze(test_sim["sigma"]))

print(f"True mu: {true_mu:.3f}, Post. mu: {mu_samples.mean():.3f}")
print(f"True sigma: {true_sigma:.3f}, Post. sigma: {sigma_samples.mean():.3f}")

n_ppc = 50
for s in range(n_ppc):
    mu_s = float(mu_samples[s])
    sigma_s = float(sigma_samples[s])
    x_rep = observation_model(mu=mu_s, sigma=sigma_s)["x"]
