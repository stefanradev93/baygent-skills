# Analysis Notes: Compartmental SIR Model with BayesFlow

## Design Choices

### Parameter Constraints

All four parameters have bounded support. The adapter handles this via `.constrain()` with softplus/sigmoid bijections so the inference network operates in unconstrained space:

| Parameter | Support | Adapter call |
|-----------|---------|-------------|
| beta (infection rate) | (0, inf) | `.constrain("beta", lower=0)` |
| gamma (recovery rate) | (0, inf) | `.constrain("gamma", lower=0)` |
| i0 (initial infected) | (0, 1) | `.constrain("i0", lower=0, upper=1)` |
| p (reporting prob.) | (0, 1) | `.constrain("p", lower=0, upper=1)` |

Constraints are applied **before** concatenation into `inference_variables`, as required by the adapter ordering rules.

### Summary Network: TimeSeriesTransformer (not SetTransformer)

Daily case counts over 30 days are an **ordered sequence** -- day order matters for epidemic dynamics. A SetTransformer would destroy temporal structure. The TimeSeriesTransformer (Small config) with `time_axis=1` and internal Time2Vec embedding captures this correctly.

`summary_dim = 8` (2x the 4 parameters being inferred), following the skill heuristic.

### Architecture Size: Small

Started with Small config per the skill's hard rule. 4 parameters with a 30-step time series is a modest problem. Scale up to Base only if diagnostics show poor recovery or calibration.

### Data Routing

- `daily_cases` -> `.as_time_series()` -> `summary_variables` (through TimeSeriesTransformer)
- All 4 parameters -> `.constrain()` -> `inference_variables`
- No `inference_conditions` needed -- no fixed-length metadata beyond what the summary network sees

### Prior Specification

- `beta ~ Gamma(2, 0.3)`: weakly informative, centers around 0.6 with moderate spread
- `gamma ~ Gamma(2, 0.15)`: centers around 0.3, consistent with recovery periods of a few days
- `i0 ~ Beta(2, 50)`: small initial infected fraction (~4%), realistic for outbreak onset
- `p ~ Beta(5, 2)`: reporting probability centered around 0.7, reflecting imperfect surveillance

### Training Configuration

- Online training (simulator is fast -- discrete-time SIR)
- 50 epochs, 32 batch size, 100 batches/epoch = 160,000 simulations
- `validation_data=300` for overfitting detection
- JAX backend via `KERAS_BACKEND=jax`

### Observation Model

Discrete-time SIR with Poisson-distributed reported cases. The Poisson noise and reporting probability `p` create a realistic observation process. Population size fixed at 10,000.

## Diagnostics Gate

The script enforces the full diagnostics pipeline:

1. Training history saved as JSON and inspected via `inspect_training.py`
2. 300 held-out simulations generated via `workflow.simulate(300)`
3. `compute_default_diagnostics` and `plot_default_diagnostics` both called
4. Metrics saved to CSV and checked via `check_diagnostics.py`
5. Hard stop if any parameter exceeds ECE > 0.10 or NRMSE > 0.15

## Limitations

- Discrete-time SIR is a simplification; continuous-time ODE or stochastic models may be more realistic
- Fixed population size (not inferred)
- No age structure, spatial heterogeneity, or time-varying transmission
- Prior ranges should be validated against domain knowledge for the specific disease
- 50 training epochs is a starting point; may need more if diagnostics show under-training
