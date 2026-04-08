"""
Offline simulation bank — amortized Bayesian inference with BayesFlow.
20,000 parameter-data pairs, proprietary simulator unavailable.
"""

import json
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import bayesflow as bf

SKILL_SCRIPTS = Path(__file__).parent / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from inspect_training import inspect_history
from check_diagnostics import check_diagnostics

RANDOM_SEED = sum(map(ord, "offline-simulation-bank-v1"))
rng = np.random.default_rng(RANDOM_SEED)

SIMULATION_BANK_DIR = Path(os.environ.get("SIM_BANK_DIR", "./simulation_bank"))
OUTPUT_DIR = Path("./offline_sim_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

def load_simulation_bank(bank_dir):
    npz_files = sorted(Path(bank_dir).glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {bank_dir}")
    all_params, all_obs = [], []
    for fpath in npz_files:
        data = np.load(fpath)
        all_params.append(data["parameters"])
        all_obs.append(data["observables"])
    return {"parameters": np.stack(all_params, axis=0), "observables": np.stack(all_obs, axis=0)}

def split_dataset(data, train_frac, val_frac, test_frac, seed):
    n_total = data["parameters"].shape[0]
    idx = np.random.default_rng(seed).permutation(n_total)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    def subset(d, indices):
        return {k: v[indices] for k, v in d.items()}
    return subset(data, idx[:n_train]), subset(data, idx[n_train:n_train+n_val]), subset(data, idx[n_train+n_val:])

full_data = load_simulation_bank(SIMULATION_BANK_DIR)
train_data, val_data, test_data = split_dataset(full_data, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION, RANDOM_SEED)

adapter = (
    bf.Adapter()
    .as_set(["observables"])
    .convert_dtype("float64", "float32")
    .concatenate(["observables"], into="summary_variables")
    .concatenate(["parameters"], into="inference_variables")
)

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

workflow = bf.BasicWorkflow(
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath=str(OUTPUT_DIR / "checkpoints"),
    checkpoint_name="offline_sim_bank",
)

history = workflow.fit_offline(
    data=train_data, epochs=200, batch_size=64,
    validation_data=val_data, verbose=2,
)

with open(OUTPUT_DIR / "history.json", "w") as f:
    json.dump(history.history, f)

training_report = inspect_history(history.history)
print(json.dumps(training_report, indent=2))

metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)
print(metrics.to_string())
metrics.to_csv(OUTPUT_DIR / "metrics.csv")

figures = workflow.plot_default_diagnostics(test_data=test_data)
for name, fig in figures.items():
    fig.savefig(OUTPUT_DIR / f"diagnostics_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

diag_report = check_diagnostics(metrics)
print(json.dumps(diag_report, indent=2))

if diag_report["overall"]["decision"] == "STOP":
    raise RuntimeError(diag_report["overall"]["recommendation"])

print("LIMITATION: PPCs unavailable — simulator is proprietary and inaccessible.")
