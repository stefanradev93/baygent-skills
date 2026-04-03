# Analysis Notes: SIR Compartmental Model with BayesFlow

## Design Choices

### Prior specification
- **beta ~ Gamma(2, 0.3)**: Weakly informative, centered around 0.6 with moderate spread. Covers realistic R0 values when combined with gamma.
- **gamma ~ Gamma(2, 0.15)**: Centered around 0.3, corresponding to ~3-day recovery. Realistic for many infectious diseases.
- **i0 ~ Beta(2, 50)**: Strongly concentrated near zero (mean ~0.04). Epidemics typically start from a small seed.
- **p ~ Beta(5, 2)**: Mildly informative, favoring moderate-to-high reporting (mean ~0.71).

### Parameter constraints
Four parameters with mixed constraint types:
- **beta, gamma > 0**: `adapter.constrain(lower=0)` applies a log-transform, mapping the positive real line to unconstrained space.
- **i0, p in (0, 1)**: `adapter.constrain(lower=0, upper=1)` applies a logit-transform, mapping the unit interval to unconstrained space.

The adapter handles forward (constrain -> unconstrained for training) and inverse (unconstrained -> constrained for inference) transforms automatically. This is critical -- without these transforms, the flow matching network would need to learn hard boundaries, leading to boundary artifacts and poor calibration.

### Network architecture
- **TimeSeriesTransformer** for the summary network, not SetTransformer. Daily case counts have temporal ordering -- day 5 vs day 25 matters. A set-based architecture would discard this information.
- **summary_dim=8**: Rule of thumb is 2x the number of parameters (4 params -> 8 summary dimensions).
- **FlowMatching** as the inference network. Standard choice for continuous posterior approximation in BayesFlow 2.x.

### Training configuration
- 50 epochs, 32 batch size, 100 batches/epoch = 160k total simulations during training.
- 300 validation simulations per epoch for monitoring overfitting.
- Online training (simulations generated on-the-fly) rather than offline, since the simulator is fast.

### Diagnostics
- 300 held-out test simulations for calibration and recovery checks.
- Explicit constraint verification: checking that all posterior samples respect parameter bounds.
- Posterior predictive checks using the same `observation_model` function (no re-implementation).

## Potential Issues
- **Identifiability**: beta and gamma can trade off against each other (what matters for dynamics is roughly beta/gamma = R0). The joint posterior plot helps diagnose this.
- **i0 near boundary**: Very small i0 values (near 0) can be hard to recover because the logit transform stretches that region. If calibration is poor for i0, consider reparameterizing as log(i0).
- **Stiff dynamics**: Some parameter combinations produce very fast epidemics that burn out in a few days, leaving mostly zeros. This can create a multimodal posterior landscape.
- **50 epochs may be insufficient** for this 4-parameter problem with nonlinear dynamics. If validation loss hasn't plateaued, increase epochs.
