# Analysis Notes — Regression with Varying N

## Variable N
meta_fn returns N ~ Uniform(10, 200). bf.make_simulator(..., meta_fn=meta_fn) injects N into observation_model.

## Adapter
- .broadcast("N", to="x") handles batch dimension
- .as_set(["x", "y"]) marks observations as exchangeable
- .sqrt("N") scales N before routing as inference_conditions
- .constrain("sigma", lower=0)

## Architecture
SetTransformer (Small) for exchangeable (x,y) pairs. summary_dim=6 (2 × 3 params). FlowMatching Small.

## PPCs
Reuses observation_model() directly with posterior draws.
