# Analysis Notes (Without Skill)

## Variable-N handling
meta_fn draws N ~ Uniform(10, 200). sqrt(N) routed as inference_conditions.

## Missing patterns
- No .broadcast() for N
- PPC reimplements the forward model inline instead of calling observation_model()
- No reference to model-sizes.md

## Adapter
Concatenates x,y into summary_variables then applies .as_set. Uses .apply for sqrt(N).
