# Analysis Notes: Offline Simulation Bank Training

## Problem setup

- 20,000 pre-generated parameter-data pairs as .npz files
- Each file: 3 parameters, 50 i.i.d. observable measurements
- Simulator is proprietary and unavailable -- no online training, no PPCs

## Design choices

### Training mode: `fit_offline`

All 20,000 simulations fit in memory. `fit_offline` is the correct choice per the skill: offline training does not require a callable simulator, only pre-simulated pairs from the correct generative process.

### Data split: train / validation / test

- **Train**: 19,000 (95%)
- **Validation**: 500 (2.5%) -- passed as `validation_data=` to `fit_offline` for overfitting detection (mandatory per skill)
- **Test**: 500 (2.5%) -- held out for diagnostics only (never seen during training)

Using training data for diagnostics would give over-optimistic results (skill hard rule).

### Architecture: SetTransformer (Small)

The 50 i.i.d. measurements are exchangeable set data. Per the skill's conditioning logic, they MUST be routed through `summary_variables` with a `SetTransformer` -- never flattened into `inference_conditions`.

Started with the Small config per the skill's mandatory rule. Scale up only if diagnostics show poor recovery/calibration.

- `summary_dim = 6` (2x the 3 parameters, per the heuristic)
- `embed_dims = (32, 32)`, `num_heads = (2, 2)`, `mlp_depths = (1, 1)`, `mlp_widths = (64, 64)`

### Inference network: FlowMatching (Small)

Default choice per the skill. Small subnet config: `widths=(128, 128)`, `time_embedding_dim=16`.

### Adapter: explicit `bf.Adapter()`

Used an explicit adapter chain (not the naming shorthand) to handle:
1. `.as_set(["observables"])` -- converts (50,) to (50, 1) for the SetTransformer
2. `.convert_dtype("float64", "float32")` -- NumPy defaults to float64, JAX expects float32
3. `.concatenate` into the reserved keys

No `.constrain()` applied since we have no information about parameter bounds from the proprietary simulator. If bounds are known, add `.constrain("parameters", lower=..., upper=...)` before `.convert_dtype`.

## Limitations

1. **No PPCs possible.** The simulator is unavailable, so we cannot re-simulate from posterior draws. This is the most significant limitation -- good in-silico diagnostics do not guarantee the simulator explains real data.
2. **No parameter constraints.** Without knowledge of the parameter support, we cannot apply `.constrain()`. If parameters have bounded support, the network may predict out-of-support values.
3. **Prior unknown.** We trust that the simulation bank was generated from a reasonable prior. If the prior was too narrow, the amortizer may not generalize to real data outside that range.
4. **Fixed observation count.** The model is trained on exactly 50 observations per simulation. Inference on datasets with a different number of observations would require retraining or architectural changes.

## Diagnostic gate

The script enforces the skill's house thresholds:
- ECE > 0.10 or NRMSE > 0.15 for any parameter: STOP (do not proceed)
- ECE > 0.05 or NRMSE > 0.10: WARN (proceed only with user acceptance)
- Contraction > 0.99 with ECE > 0.05: STOP (overconfident and miscalibrated)

## Next steps if diagnostics pass

1. Run inference on real observations via `workflow.sample(conditions=..., num_samples=1000)`
2. If the simulator becomes available again, run PPCs immediately
3. If diagnostics show poor recovery, scale up to Base config and retrain
