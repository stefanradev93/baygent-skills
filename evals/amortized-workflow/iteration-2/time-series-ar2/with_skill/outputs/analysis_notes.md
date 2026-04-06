# AR(2) Amortized Inference -- Design Notes

## Architecture choices

- **Summary network**: `TimeSeriesTransformer` (Small config). The data is a temporally ordered sequence of length 100 -- not exchangeable -- so a SetTransformer would destroy the ordering information that is critical for identifying AR coefficients. The TimeSeriesTransformer uses internal Time2Vec positional embeddings (`time_axis=1`) to encode temporal position.
- **Inference network**: `FlowMatching` with Small subnet (widths 128x128, time_embedding_dim 16).
- **summary_dim = 6**: heuristic of 2x the number of parameters (3 params).
- **Started with Small config** per skill rules. Scale up only if diagnostics show poor recovery/calibration.

## Adapter design

- `.as_time_series(["y"])` reshapes `(T,)` to `(T, 1)` so the TimeSeriesTransformer receives a proper 3D tensor `(batch, T, 1)`.
- `.constrain("sigma", lower=0)` applies a softplus bijection so the network predicts in unconstrained space but sigma is always positive after back-transform.
- `phi1` and `phi2` are left unconstrained -- the prior is `Uniform(-1, 1)` but the parameters have no hard physical boundary (non-stationary AR processes are valid), so no `.constrain()` needed.
- Data routed through `summary_variables` (not `inference_conditions`) since this is structured temporal data.

## Prior specification

- `phi1, phi2 ~ Uniform(-1, 1)` -- deliberately wide, covers both stationary and non-stationary regimes. The network learns to handle both.
- `sigma ~ Gamma(2, 1)` -- weakly informative, ensures positivity, peaks near 1.

## Training regime

- Online training with `fit_online` (simulator is fast -- pure NumPy AR loop).
- 50 epochs, batch_size=32, 100 batches/epoch = 160,000 total simulations.
- `validation_data=300` for overfitting detection (mandatory per skill rules).
- Training history saved to JSON and inspected via `inspect_history()`.

## Diagnostics

- 300 held-out simulations via `workflow.simulate(300)`.
- Numerical metrics checked against house thresholds via `check_diagnostics()`.
- Visual diagnostics saved as PNGs.
- Go/no-go gate enforced: script raises `RuntimeError` on STOP.

## Posterior predictive checks

- 50 posterior draws passed through the *same* `observation_model` function (no re-implementation).
- Two PPC summaries: (1) raw time series overlay, (2) autocorrelation at lags 1 and 2.
- Lag-1 and lag-2 ACF are the most directly informative test quantities for AR(2) -- they are sufficient statistics for the AR coefficients under Gaussian noise.

## Limitations

- Prior does not enforce stationarity (roots inside unit circle). This is intentional for generality but means the network sees some explosive trajectories during training.
- Sequence length is fixed at T=100. For variable-length inference, would need to either pad/truncate or pass T as `inference_conditions`.
- Gaussian noise assumption is baked in. Model misspecification (heavy tails, heteroskedasticity) would need simulator changes.
