# Model Sizes for Different Architectures

**ALWAYS start with the Small configuration.** Scale up to Base or Large ONLY if in-silico diagnostics show poor recovery or calibration after sufficient training. Oversized networks waste compute, train slower, and can hurt calibration on simple problems.

**SetTransformer is reserved for set-based data.** Never use this architecture when dealing with time series data. If time indices (continuous or discrete) are part of
the simulator output, always indicate the `time_axis` when initializing a `TimeSeriesTransformer` which is important for the internal `Time2Vec` embedding.

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