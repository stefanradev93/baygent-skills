"""
Bayesian Denoising with BayesFlow — Fashion MNIST
Clean images as inference targets, blurry images as conditions.
DiffusionModel with UNet subnet, offline training.
"""

import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

RANDOM_SEED = sum(map(ord, "bayesian-denoising-fashionmnist"))
rng = np.random.default_rng(RANDOM_SEED)

import keras
(x_train_raw, _), (x_test_raw, _) = keras.datasets.fashion_mnist.load_data()

x_train_raw = x_train_raw[..., np.newaxis]
x_test_raw = x_test_raw[..., np.newaxis]

x_train_clean = x_train_raw.astype(np.float64) / 255.0
x_test_clean = x_test_raw.astype(np.float64) / 255.0

BLUR_SIGMA = 2.0

def apply_gaussian_blur(images, sigma=BLUR_SIGMA):
    blurred = np.empty_like(images)
    for i in range(images.shape[0]):
        for c in range(images.shape[-1]):
            blurred[i, :, :, c] = gaussian_filter(images[i, :, :, c], sigma=sigma)
    return blurred

x_train_blurry = apply_gaussian_blur(x_train_clean)
x_test_blurry = apply_gaussian_blur(x_test_clean)

N_TRAIN = 5_000
N_VAL = 500

idx = rng.permutation(len(x_train_clean))
train_idx = idx[:N_TRAIN]
val_idx = idx[N_TRAIN:N_TRAIN + N_VAL]

train_data = {
    "clean_image": x_train_clean[train_idx],
    "blurry_image": x_train_blurry[train_idx],
}
val_data = {
    "clean_image": x_train_clean[val_idx],
    "blurry_image": x_train_blurry[val_idx],
}

N_TEST = 200
test_data = {
    "clean_image": x_test_clean[:N_TEST],
    "blurry_image": x_test_blurry[:N_TEST],
}

import bayesflow as bf

adapter = (
    bf.Adapter()
    .convert_dtype("float64", "float32")
    .rename("clean_image", "inference_variables")
    .rename("blurry_image", "inference_conditions")
)

diffusion = bf.networks.DiffusionModel(
    subnet=bf.networks.UNet,
    prediction_type="velocity",
    noise_schedule="cosine",
)

workflow = bf.BasicWorkflow(
    inference_network=diffusion,
    adapter=adapter,
    checkpoint_filepath="checkpoints",
    checkpoint_name="bayesian_denoising",
    initial_learning_rate=1e-4,
)

history = workflow.fit_offline(
    data=train_data, epochs=30, batch_size=32,
    validation_data=val_data, verbose=2,
)

with open("history.json", "w") as f:
    json.dump(history.history, f)

try:
    _skill_root = os.path.join(os.path.dirname(__file__), "..", "amortized-workflow")
    sys.path.insert(0, _skill_root)
    from scripts.inspect_training import inspect_history
    training_report = inspect_history(history.history)
    print(json.dumps(training_report, indent=2))
except ImportError:
    print("inspect_training not available — checking inline")
    losses = history.history.get("loss", [])
    val_losses = history.history.get("val_loss", [])
    print(f"Final train loss: {losses[-1]:.4f} | Final val loss: {val_losses[-1]:.4f}")

N_EXAMPLES = 5
N_SAMPLES = 4

conditions = {"blurry_image": test_data["blurry_image"][:N_EXAMPLES].astype(np.float64)}
posterior_samples = workflow.sample(conditions=conditions, num_samples=N_SAMPLES)
generated = posterior_samples["clean_image"]

n_cols = 2 + N_SAMPLES
fig = plt.figure(figsize=(2.5 * n_cols, 2.5 * N_EXAMPLES))
gs = gridspec.GridSpec(N_EXAMPLES, n_cols, figure=fig, hspace=0.05, wspace=0.05)

for row in range(N_EXAMPLES):
    ax = fig.add_subplot(gs[row, 0])
    ax.imshow(test_data["blurry_image"][row, :, :, 0], cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax = fig.add_subplot(gs[row, 1])
    ax.imshow(test_data["clean_image"][row, :, :, 0], cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    for s in range(N_SAMPLES):
        ax = fig.add_subplot(gs[row, 2 + s])
        ax.imshow(np.clip(generated[row, s, :, :, 0], 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

plt.savefig("posterior_sample_grid.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved posterior_sample_grid.png")
