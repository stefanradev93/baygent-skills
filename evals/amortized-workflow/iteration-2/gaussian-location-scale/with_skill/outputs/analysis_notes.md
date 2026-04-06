# Analysis Notes: Gaussian Location-Scale Model

## Inferential targets

- **mu** (real-valued): location parameter
- **sigma** (positive): scale parameter
- **Observations**: 50 i.i.d. draws from Normal(mu, sigma)

## Design choices

### Prior specification

- `mu ~ Normal(0, 5)` -- weakly informative, allows a wide range of locations.
- `sigma ~ |Normal(0, 2)|` (half-normal) -- weakly informative, concentrates mass on moderate scales while allowing larger values.

### Data routing

50 i.i.d. observations are **exchangeable** (permutation-invariant), so they are routed through `summary_variables` with a `SetTransformer`. They are NOT treated as a flat vector in `inference_conditions`. The adapter uses `.as_set(["x"])` to reshape `(50,)` to `(50, 1)`.

### Adapter constraints

`sigma` has support on `(0, inf)`. The adapter applies `.constrain("sigma", lower=0)` so the inference network works in unconstrained space and the softplus bijection guarantees valid posterior samples.

### Architecture

- **Summary network**: `SetTransformer` -- Small config (embed_dims=(32,32), num_heads=(2,2), mlp_depths=(1,1), mlp_widths=(64,64)). `summary_dim=4` (2x the 2 parameters being inferred).
- **Inference network**: `FlowMatching` -- Small subnet config (widths=(128,128), time_embedding_dim=16).
- Rationale: 2 parameters with a simple likelihood; Small is appropriate. Scale up only if diagnostics show poor recovery or calibration.

### Training regime

- **Mode**: Online (simulator is fast)
- **Budget**: 100 epochs x 100 batches/epoch x 32 simulations/batch = 320,000 simulated datasets
- **Validation**: 300 auto-simulated validation sets per epoch

### Diagnostics pipeline

1. Training history saved as JSON, inspected with `inspect_training.py` for NaN, overfitting, under-training.
2. 300 held-out simulations generated via `workflow.simulate(300)`.
3. Visual diagnostics saved via `workflow.plot_default_diagnostics()`.
4. Numerical diagnostics checked against house thresholds via `check_diagnostics.py` (ECE, NRMSE, contraction).
5. Go/no-go gate before proceeding to real-data inference.

### Posterior predictive checks

PPCs reuse `observation_model()` directly -- the simulator function is never re-implemented. 50 posterior draws are passed through the forward model and compared to the observed data via sample mean and sample standard deviation as test quantities.

## Limitations

- The prior is fixed; no sensitivity analysis across alternative priors is performed.
- The model assumes Gaussian data -- no robustness to outliers or heavy tails.
- 100 training epochs may be insufficient for harder problems; this is adequate for a 2-parameter Gaussian.
