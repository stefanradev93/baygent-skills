# Analysis Notes: Gaussian Location-Scale Amortized Inference

## Model

- **Parameters**: mu (location), sigma (scale)
- **Prior**: mu ~ Normal(0, 5), sigma ~ HalfNormal(2)
- **Likelihood**: x_i ~ Normal(mu, sigma), i = 1..50, i.i.d.

## Design Choices

### Architecture

- **Summary network**: SetTransformer with summary_dim=4. The set transformer respects exchangeability of i.i.d. observations (permutation invariance). summary_dim=4 gives 2x the number of parameters — enough capacity for a 2-parameter model without overfitting.
- **Inference network**: FlowMatching with a 2-layer subnet (128 units each). Flow matching is the default and generally best-performing density estimator in BayesFlow 2.x. The subnet is deliberately small given the low dimensionality.

### Adapter

- `as_set(["x"])`: Reshapes the 50-dimensional observation vector into a (50, 1) set for the SetTransformer input.
- `constrain("sigma", lower=0)`: Applies a softplus/log transform so the network operates in unconstrained space for sigma, then back-transforms at inference time.
- `convert_dtype("float64", "float32")`: NumPy defaults to float64, but JAX/neural networks prefer float32.
- `concatenate`: Groups variables into the two BayesFlow slots — `summary_variables` (what the summary network sees) and `inference_variables` (what the flow learns to generate).

### Training

- 100 epochs, 32 batch size, 100 batches per epoch = 320,000 total simulations during training.
- 300 validation simulations per epoch for monitoring overfitting.
- Online training (simulations generated on-the-fly) — appropriate here since the simulator is cheap.

### Diagnostics

- BayesFlow's built-in `plot_default_diagnostics` produces recovery plots and calibration histograms (simulation-based calibration / SBC).
- `compute_default_diagnostics` returns numerical metrics (R-squared for recovery, calibration error, etc.).
- 300 test simulations for diagnostics — sufficient for a 2-parameter model.

### Inference

- 2000 posterior samples drawn via a single forward pass through the trained network.
- Ground truth test case: mu=2.5, sigma=1.3 — deliberately not centered at the prior to verify the network generalizes.
- Posterior predictive checks with 200 replications to verify the inferred posterior is consistent with the observed data.

## Limitations

- No automated threshold checks on diagnostic metrics (the with_skill version uses dedicated helper scripts for this).
- Training hyperparameters (epochs, batch size, network size) were chosen by judgment rather than systematic tuning.
- No early stopping — for this simple model 100 epochs is unlikely to overfit, but it would be good practice for more complex models.
- The prior is fairly diffuse (sigma of 5 for mu, 2 for sigma) — if real data lives in a narrow range, a tighter prior could improve amortization efficiency.
