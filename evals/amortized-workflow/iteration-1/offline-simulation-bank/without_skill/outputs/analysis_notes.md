# Analysis Notes: Offline Simulation Bank Training

## Task Summary

Train a BayesFlow amortized posterior estimator from 20,000 pre-generated
parameter-data pairs (`.npz` files). The original simulator is proprietary
and unavailable, so all training is offline.

## Design Choices

### Data Split

- **Train**: 18,000 (90%)
- **Validation**: 1,000 (5%) -- for monitoring loss during training
- **Test**: 1,000 (5%) -- held out for post-training diagnostics

Shuffled with a fixed random seed for reproducibility.

### Architecture

- **Summary network**: `SetTransformer` with `summary_dim=6` (2x the 3 parameters).
  The 50 i.i.d. observations are exchangeable, so a set-equivariant architecture
  is the natural choice. Used medium-sized config (64-dim embeddings, 4 heads).
- **Inference network**: `FlowMatching` with a 2-layer subnet (256 units each).
  Flow matching is the current default in BayesFlow 2.x for continuous posteriors.

### Adapter Pipeline

1. `.as_set(["observables"])` -- reshapes `(50,)` to `(50, 1)` for the SetTransformer
2. `.convert_dtype("float64", "float32")` -- match Keras/JAX expectations
3. `.concatenate(["observables"], into="summary_variables")` -- route to summary net
4. `.concatenate(["parameters"], into="inference_variables")` -- route to inference net

### Training

- 150 epochs, batch size 64
- `fit_offline` since we have a fixed dataset (no on-the-fly simulation)
- Validation data passed for loss tracking and early stopping awareness

### Diagnostics

- BayesFlow built-in `plot_default_diagnostics` and `compute_default_diagnostics`
  on the held-out test set
- These produce calibration checks (simulation-based calibration),
  posterior z-scores vs. contraction, and recovery plots

## Known Limitations

1. **No posterior predictive checks**: The simulator is unavailable, so we cannot
   generate replicated data from posterior draws. If it becomes available, PPCs
   should be run immediately.
2. **No parameter constraints enforced**: Without knowing the generative model,
   we cannot apply `.constrain()` in the adapter (e.g., positivity constraints).
   If any parameters are known to be bounded, adding constraints would improve
   training.
3. **Fixed simulation budget**: With 20,000 sims and no simulator, we cannot
   generate more data if diagnostics are poor. The only recourse is architecture
   or hyperparameter tuning.
4. **Prior coverage unknown**: We trust that the simulation bank covers the
   relevant prior range. If the prior was too narrow or the bank is biased,
   the estimator may extrapolate poorly on real data.
