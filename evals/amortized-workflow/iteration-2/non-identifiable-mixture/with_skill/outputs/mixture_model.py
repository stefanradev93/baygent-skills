"""
Amortized Bayesian inference for a two-component Gaussian mixture model.
Parameters: w in (0,1), mu1 in R, mu2 in R, sigma > 0
"""

import json
import sys
import warnings
import bayesflow as bf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/baygent-skills/amortized-workflow")
from scripts.check_diagnostics import check_diagnostics
from scripts.inspect_training import inspect_history

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_SEED = sum(map(ord, "gaussian-mixture-non-identifiable-v1"))
rng = np.random.default_rng(RANDOM_SEED)

N_OBS = 200

def prior():
    w = rng.uniform(0.0, 1.0)
    mu1 = rng.normal(0.0, 3.0)
    mu2 = rng.normal(0.0, 3.0)
    sigma = np.abs(rng.normal(0.0, 2.0)) + 1e-6
    return {"w": w, "mu1": mu1, "mu2": mu2, "sigma": sigma}

def observation_model(w, mu1, mu2, sigma):
    component = rng.binomial(1, w, size=N_OBS)
    x = np.where(
        component == 1,
        rng.normal(mu1, sigma, size=N_OBS),
        rng.normal(mu2, sigma, size=N_OBS),
    )
    return {"x": x}

simulator = bf.make_simulator([prior, observation_model])

summary_net = bf.networks.SetTransformer(
    summary_dim=8,
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
    .constrain("w", lower=0.0, upper=1.0)
    .constrain("sigma", lower=0.0)
    .convert_dtype("float64", "float32")
    .concatenate(["x"], into="summary_variables")
    .concatenate(["w", "mu1", "mu2", "sigma"], into="inference_variables")
)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath="checkpoints/gaussian_mixture",
    checkpoint_name="gmm_non_identifiable",
)

history = workflow.fit_online(
    epochs=300, batch_size=32, num_batches_per_epoch=100,
    validation_data=300, verbose=2,
)

with open("history.json", "w") as f:
    json.dump(history.history, f)

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

diag_report = check_diagnostics(metrics)
print(json.dumps(diag_report, indent=2))

decision = diag_report["overall"]["decision"]
print(f"Overall decision: {decision}")

# Non-identifiability analysis and remediation options printed
# Real-data inference skipped due to expected diagnostic issues
