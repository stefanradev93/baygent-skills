"""
Amortized Bayesian inference for a 4-parameter SIR compartmental model.

Parameters: beta (>0), gamma (>0), i0 (0,1), p (0,1)
Observations: daily case counts for 30 days.
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bayesflow as bf

RANDOM_SEED = sum(map(ord, "sir-amortized-inference-v1"))
rng = np.random.default_rng(RANDOM_SEED)

N_POP = 10_000
T_DAYS = 30

def prior():
    beta = rng.gamma(shape=2.0, scale=0.25)
    gamma = rng.gamma(shape=2.0, scale=0.1)
    i0 = rng.beta(a=1.5, b=20.0)
    p = rng.beta(a=2.0, b=3.0)
    return {
        "beta": np.atleast_1d(beta).astype(np.float64),
        "gamma": np.atleast_1d(gamma).astype(np.float64),
        "i0": np.atleast_1d(i0).astype(np.float64),
        "p": np.atleast_1d(p).astype(np.float64),
    }

def sir_observation_model(beta, gamma, i0, p):
    beta, gamma, i0, p_rep = float(beta), float(gamma), float(i0), float(p)
    S = (1.0 - i0) * N_POP
    I = i0 * N_POP
    R = 0.0
    daily_cases = []
    for _ in range(T_DAYS):
        new_infections = beta * S * I / N_POP
        new_recoveries = gamma * I
        new_infections = min(new_infections, S)
        new_recoveries = min(new_recoveries, I)
        S = max(S - new_infections, 0.0)
        I = max(I + new_infections - new_recoveries, 0.0)
        R = R + new_recoveries
        reported = rng.poisson(lam=max(p_rep * new_infections, 0.0))
        daily_cases.append(float(reported))
    return {"daily_cases": np.array(daily_cases, dtype=np.float64).reshape(T_DAYS, 1)}

simulator = bf.make_simulator([prior, sir_observation_model])

summary_net = bf.networks.TimeSeriesTransformer(
    summary_dim=8,
    embed_dims=(32, 32),
    num_heads=(2, 2),
    mlp_depths=(1, 1),
    mlp_widths=(64, 64),
    time_embed_dim=4,
    time_axis=1,
)

inference_net = bf.networks.FlowMatching(
    subnet_kwargs={"widths": (128, 128, 128), "time_embedding_dim": 16}
)

adapter = (
    bf.Adapter()
    .constrain("beta", lower=0)
    .constrain("gamma", lower=0)
    .constrain("i0", lower=0, upper=1)
    .constrain("p", lower=0, upper=1)
    .convert_dtype("float64", "float32")
    .concatenate(["daily_cases"], into="summary_variables")
    .concatenate(["beta", "gamma", "i0", "p"], into="inference_variables")
)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath="checkpoints/sir",
    checkpoint_name="sir_model",
)

history = workflow.fit_online(
    epochs=300, batch_size=32, num_batches_per_epoch=100,
    validation_data=300, verbose=2,
)

with open("history.json", "w") as f:
    json.dump(history.history, f)

sys.path.insert(0, "amortized-workflow/scripts")
from inspect_training import inspect_history
training_report = inspect_history(history.history)
print(json.dumps(training_report, indent=2))

test_data = workflow.simulate(500)

figures = workflow.plot_default_diagnostics(test_data=test_data)
for name, fig in figures.items():
    fig.savefig(f"diagnostics_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)
print(metrics.to_string())
metrics.to_csv("metrics.csv")

from check_diagnostics import check_diagnostics
diag_report = check_diagnostics(metrics)
print(json.dumps(diag_report, indent=2))

if diag_report["overall"]["decision"] == "STOP":
    raise RuntimeError(diag_report["overall"]["recommendation"])

x_obs_real = test_data["daily_cases"][[0]]
samples = workflow.sample(conditions={"daily_cases": x_obs_real}, num_samples=1000)

for param in ["beta", "gamma", "i0", "p"]:
    draws = samples[param][0, :, 0]
    print(f"  {param:5s}  mean={draws.mean():.4f}  std={draws.std():.4f}")

n_ppc = 50
for s in range(n_ppc):
    theta_s = {
        "beta": float(samples["beta"][0, s, 0]),
        "gamma": float(samples["gamma"][0, s, 0]),
        "i0": float(samples["i0"][0, s, 0]),
        "p": float(samples["p"][0, s, 0]),
    }
    x_rep = sir_observation_model(**theta_s)
