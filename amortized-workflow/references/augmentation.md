# Augmentations

All `fit_*` methods in BayesFlow accept an `augmentations=` argument:

```python
workflow.fit_offline(..., augmentations=augment)
```

Use augmentations for training-only transformations. They are applied to each batch during fitting, but not during inference.

Supported patterns for augmentations:

- `None` - do nothing
- `Callable` - apply one function to the whole batch
- `Mapping` - apply functions to specific batch keys
- `Sequence` - apply several functions in order

A minimal example is adding tiny Gaussian noise to `x` during training:

```python
import numpy as np

def augment(batch):
    batch = dict(batch)
    batch["x"] += 0.01 * np.random.normal(size=batch["x"].shape)
    return batch

workflow.fit_offline(..., augmentations=augment)
```

If a transformation must also be applied at inference time, it should usually be part of the simulator, adapter, or general preprocessing pipeline instead. The rule of thumb: **augmentations mutate values, adapters mutate structure.** Never restructure (rename, split, merge, delete) keys in an augmentation.