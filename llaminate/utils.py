import tensorflow as tf

import mlable.utils

# CACHING #####################################################################

def create_cache(batch_dim: int, cache_dim: int, head_dim: int, num_layers: int, num_heads: int=None) -> list:
    return [mlable.utils.create_cache(batch_dim=batch_dim, cache_dim=cache_dim, head_dim=head_dim, num_heads=num_heads) for _ in range(num_layers)]

# MASKING #####################################################################

def compute_attention_masks(time_step: int, seq_len: int, input_mask: tf.Tensor) -> tf.Tensor:
    """Computes causal attention mask."""
    bsz = tf.shape(input_mask)[0]
    batch_time_step = tf.fill([bsz, 1], time_step)
    causal_padding = tf.greater(
        tf.expand_dims(tf.range(seq_len), 0), batch_time_step)
    causal_padding = tf.cast(causal_padding, tf.bool)
    causal_padding = tf.logical_and(
        causal_padding, tf.expand_dims(tf.cast(input_mask, tf.bool), axis=-1))
    attention_mask = tf.expand_dims(causal_padding, axis=1)
    attention_mask = tf.squeeze(attention_mask, axis=1)
    return tf.logical_not(attention_mask)
