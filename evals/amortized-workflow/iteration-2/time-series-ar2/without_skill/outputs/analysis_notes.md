# Analysis Notes (Without Skill)

## Architecture
TimeSeriesTransformer (not SetTransformer — correct for ordered data). Ad hoc sizes (embed_dim=64, num_heads=4, num_layers=2).

## Issues
- Uses fabricated bf.Approximator class and approximator.constrain() method
- Uses fabricated bf.diagnostics.run_sbc() function
- No reference to model-sizes.md
- No structured diagnostics gate (manual coverage check only)
