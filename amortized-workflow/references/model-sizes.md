# Model Sizes for Different Architectures

**ALWAYS start with the Small configuration.** Scale up to Base or Large ONLY if in-silico diagnostics show poor recovery or calibration after sufficient training. Oversized networks waste compute, train slower, and can hurt calibration on simple problems.

**SetTransformer is reserved for set-based data.** Never use this architecture when dealing with time series data. If time indices (continuous or discrete) are part of
the simulator output, always indicate the `time_axis` when initializing a `TimeSeriesTransformer` which is important for the internal `Time2Vec` embedding.

## `summary_dim` heuristic

All summary networks accept a `summary_dim` argument that controls the dimensionality of the learned summary statistics vector. As a starting heuristic, **set `summary_dim` to 2× the number of parameters being inferred.** For example, if you are estimating 5 parameters, start with `summary_dim=10`. Scale up if diagnostics show poor recovery; scale down if the problem is very low-dimensional (1–2 parameters).

## SetTransformer
| Model Size | embed_dims             | num_heads             | mlp_depths           | mlp_widths              |
|------------|------------------------|-----------------------|----------------------|-------------------------|
| Small      | (32, 32)               | (2, 2)                | (1, 1)               | (64, 64)                |
| Base       | (64, 64, 64)           | (4, 4, 4)             | (2, 2, 2)            | (128, 128, 128)         |
| Large      | (128, 128, 128, 128)   | (8, 8, 8, 8)          | (2, 2, 2, 2)         | (256, 256, 256, 256)    |

## FusionTransformer
| Model Size | embed_dims             | num_heads             | mlp_depths           | mlp_widths              | template_dim |
|------------|------------------------|-----------------------|----------------------|-------------------------|--------------|
| Small      | (32, 32)               | (2, 2)                | (1, 1)               | (64, 64)                | 64           |
| Base       | (64, 64, 64)           | (4, 4, 4)             | (2, 2, 2)            | (128, 128, 128)         | 128          |
| Large      | (128, 128, 128, 128)   | (8, 8, 8, 8)          | (2, 2, 2, 2)         | (256, 256, 256, 256)    | 256          |

## TimeSeriesTransformer
| Model Size | embed_dims             | num_heads             | mlp_depths           | mlp_widths              | time_embed_dim | time_axis |
|------------|------------------------|-----------------------|----------------------|-------------------------|----------------|-----------|
| Small      | (32, 32)               | (2, 2)                | (1, 1)               | (64, 64)                | 4              | 1         |
| Base       | (64, 64, 64)           | (4, 4, 4)             | (2, 2, 2)            | (128, 128, 128)         | 8              | 2         |
| Large      | (128, 128, 128, 128)   | (8, 8, 8, 8)          | (2, 2, 2, 2)         | (256, 256, 256, 256)    | 16             | 3         |

## ConvolutionalNetwork
| Model Size | widths                  | blocks_per_stage           |
|------------|------------------------|----------------------------|
| Small      | (16, 32)               | 2                     |
| Base       | (32, 64, 128)          | 3                  |
| Large      | (32, 64, 128, 256)     | 4               |

## TimeSeriesNetwork
| Model Size | filters                | kernel_sizes           | strides           | recurrent_dim |
|------------|-----------------------|------------------------|-------------------|---------------|
| Small      | (16, 32)              | (3, 3)                 | (1, 1)            | 64            |
| Base       | (16, 32, 64)          | (3, 3, 3)              | (1, 1, 1)         | 128           |
| Large      | (16, 32, 64, 128)     | (3, 3, 3, 3)           | (1, 1, 1, 1)      | 256           |

## Inference networks (subnet sizing)

All three inference networks (`FlowMatching`, `DiffusionModel`, `StableConsistencyModel`) use a `TimeMLP` subnet by default. Control its capacity via `subnet_kwargs=`:

```python
bf.networks.FlowMatching(subnet_kwargs={"widths": (128, 128), "time_embedding_dim": 16})
```

| Model Size | widths                 | time_embedding_dim |
|------------|------------------------|--------------------|
| Small      | (128, 128)             | 16                 |
| Base       | (256, 256)             | 32                 |
| Large      | (512, 512, 512)        | 64                 |

**Base is the BayesFlow default.** Start with Small for problems with fewer than ~5 parameters and simple summary statistics. Scale to Large only if diagnostics show poor calibration after sufficient training with Base.