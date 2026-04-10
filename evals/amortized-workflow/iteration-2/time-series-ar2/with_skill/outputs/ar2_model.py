"""
AR(2) amortized inference with BayesFlow.
Parameters: phi1, phi2, sigma. Time series length T=100.
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bayesflow as bf

RANDOM_SEED = sum(map(ord, "ar2-amortized-inference-v1"))
rng = np.random.default_rng(RANDOM_SEED)

T = 100

def prior():
    while True:
        phi1 = rng.uniform(-1.5, 1.5)
        phi2 = rng.uniform(-1.0, 1.0)
        if abs(phi2) < 1.0 and (phi2 + phi1) < 1.0 and (phi2 - phi1) < 1.0:
            break
    sigma = rng.gamma(shape=2.0, scale=0.5)
    return {"phi1": np.array([phi1]), "phi2": np.array([phi2]), "sigma": np.array([sigma])}

def observation_model(phi1, phi2, sigma):
    phi1, phi2, sigma = float(phi1), float(phi2), float(sigma)
    y = np.zeros(T)
    y[0] = rng.normal(0, sigma / max(1e-6, np.sqrt(1 - phi1**2 - phi2**2 + 1e-8)))
    y[1] = rng.normal(phi1 * y[0], sigma)
    for t in range(2, T):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + rng.normal(0, sigma)
    return {"time_series": y.reshape(T, 1).astype(np.float64)}

simulator = bf.make_simulator([prior, observation_model])

summary_net = bf.networks.TimeSeriesTransformer(
    embed_dims=(32, 32), num_heads=(2, 2),
    mlp_depths=(1, 1), mlp_widths=(64, 64),
    time_embed_dim=4, time_axis=1, summary_dim=6,
)

inference_net = bf.networks.FlowMatching(
    subnet_kwargs={"widths": (128, 128, 128), "time_embedding_dim": 16}
)

adapter = (
    bf.Adapter()
    .as_time_series(["time_series"])
    .constrain("sigma", lower=0)
    .convert_dtype("float64", "float32")
    .concatenate(["phi1", "phi2", "sigma"], into="inference_variables")
    .concatenate(["time_series"], into="summary_variables")
)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath="checkpoints_ar2",
    checkpoint_name="ar2_model",
)

history = workflow.fit_online(
    epochs=200, batch_size=32, num_batches_per_epoch=100,
    validation_data=300, verbose=2,
)

with open("history_ar2.json", "w") as f:
    json.dump(history.history, f)

sys.path.insert(0, "scripts")
from inspect_training import inspect_history
training_report = inspect_history(history.history)
print(json.dumps(training_report, indent=2))

test_data = workflow.simulate(300)

figures = workflow.plot_default_diagnostics(test_data=test_data)
for name, fig in figures.items():
    fig.savefig(f"diagnostics_ar2_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)
print(metrics)
metrics.to_csv("metrics_ar2.csv")

from check_diagnostics import check_diagnostics
diag_report = check_diagnostics(metrics)
print(json.dumps(diag_report, indent=2))

if diag_report["overall"]["decision"] == "STOP":
    raise RuntimeError(diag_report["overall"]["recommendation"])

held_out_prior = prior()
held_out_obs = observation_model(**{k: held_out_prior[k] for k in ["phi1", "phi2", "sigma"]})

samples = workflow.sample(
    conditions={"time_series": held_out_obs["time_series"][np.newaxis]},
    num_samples=1000,
)

phi1_samples = samples["phi1"]
phi2_samples = samples["phi2"]
sigma_samples = samples["sigma"]

print(f"phi1  mean: {phi1_samples[0, :, 0].mean():.3f}")
print(f"phi2  mean: {phi2_samples[0, :, 0].mean():.3f}")
print(f"sigma mean: {sigma_samples[0, :, 0].mean():.3f}")

n_ppc = 50
for s in range(n_ppc):
    theta_s = {
        "phi1": phi1_samples[0, s, 0],
        "phi2": phi2_samples[0, s, 0],
        "sigma": sigma_samples[0, s, 0],
    }
    rep = observation_model(**theta_s)
