# Analysis Notes: Linear Regression with Variable N

## Model

y = alpha + beta * x + noise(sigma), with N observations per dataset.

## Design Choices

### Variable sample size handling

- **meta_fn** draws N ~ Uniform{10, 200} per simulated dataset. This is passed to `bf.make_simulator` as `meta_fn=meta`, so N varies *between* batches (not within).
- **`.broadcast("N", to="x")`** replicates the scalar N along the batch dimension so it can be concatenated into tensors downstream.
- **`.sqrt("N")`** applies a monotone transform before routing to the inference network -- sqrt(N) grows more smoothly and is a better conditioning signal than raw N.
- **N routed as `inference_conditions`**, so the inference network directly sees sample-size information alongside the learned summary from the SetTransformer. This dual-path setup (summary_variables + inference_conditions) lets the network adapt its posterior width to the amount of data.

### Architecture

- **SetTransformer (Small)** for (x, y) pairs: these are exchangeable regression observations, routed through `summary_variables`. summary_dim = 6 (2x the 3 parameters).
- **FlowMatching (Small)** as the inference network.
- Started with Small config per the skill's hard rule. Scale up only if diagnostics show poor recovery.

### Adapter

- `.as_set(["x", "y"])` reshapes 1D arrays to (N, 1) for the SetTransformer.
- `.constrain("sigma", lower=0)` applies softplus bijection so the network predicts in unconstrained space.
- `.convert_dtype("float64", "float32")` prevents dtype mismatch with JAX backend.
- Concatenation order: `[alpha, beta, sigma]` into `inference_variables`; `[x, y]` into `summary_variables`; `N` renamed to `inference_conditions`.

### Training

- 100 epochs, batch_size=32, 100 batches/epoch = 320,000 simulated datasets total.
- `validation_data=300` for overfitting detection.
- Training history saved to JSON and inspected with `inspect_training.py`.

### Diagnostics

- 300 held-out simulations via `workflow.simulate(300)`.
- Both `plot_default_diagnostics` and `compute_default_diagnostics` used.
- Metrics checked against house thresholds (ECE < 0.10, NRMSE < 0.15, contraction > 0.80) via `check_diagnostics.py`.

### Priors

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| alpha | Normal(0, 5) | Weakly informative intercept |
| beta | Normal(0, 3) | Weakly informative slope |
| sigma | Gamma(2, 1) | Positive, mode at 1, mean at 2 |

### Limitations

- The network is trained for x ~ Normal(0, 1). Real data with very different covariate distributions may require retraining or prior adjustment.
- The prior on sigma (Gamma(2,1)) puts most mass on moderate noise levels. Very low or very high noise regimes are under-represented.
- 100 epochs may be insufficient for the full N range -- check diagnostics and extend if needed.
