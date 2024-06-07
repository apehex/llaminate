import tensorflow as tf

import mlable.utils

# CACHING #####################################################################

def create_cache(batch_dim: int, cache_dim: int, head_dim: int, num_layers: int, num_heads: int=None) -> list:
    return [mlable.utils.create_cache(batch_dim=batch_dim, cache_dim=cache_dim, head_dim=head_dim, num_heads=num_heads) for _ in range(num_layers)]
