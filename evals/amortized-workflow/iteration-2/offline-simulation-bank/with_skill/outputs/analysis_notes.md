# Analysis Notes — Offline Simulation Bank

## Training mode
fit_offline() — no simulator available. 80/10/10 train/val/test split.

## Architecture
SetTransformer (Small) for 50 i.i.d. observations. summary_dim=8. FlowMatching Small.

## Adapter
Explicit bf.Adapter() with .as_set, .convert_dtype, .concatenate routing.

## Limitation
PPCs not possible — proprietary simulator unavailable. Inference quality assessed via in-silico diagnostics only.
