"""Building blocks of llaminate."""

import keras
import tensorflow as tf

import mlable.tensorflow.layers as _mtl

# FEED FORWARD ################################################################

@keras.saving.register_keras_serializable(package='blocks')
class FeedForwardBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float=1.,
        **kwargs
    ) -> None:
        super(FeedForwardBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'dim': dim,
            'hidden_dim': hidden_dim,
            'multiple_of': multiple_of,
            'ffn_dim_multiplier': ffn_dim_multiplier,}
        # shape
        __hidden_dim = int(2 * hidden_dim * ffn_dim_multiplier / 3)
        __hidden_dim = multiple_of * ((__hidden_dim + multiple_of - 1) // multiple_of)
        # layers
        self._w1 = tf.keras.layers.Dense(units=__hidden_dim, activation='silu', use_bias=False, kernel_initializer='glorot_uniform')
        self._w2 = tf.keras.layers.Dense(units=dim, activation='linear', use_bias=False, kernel_initializer='glorot_uniform')
        self._w3 = tf.keras.layers.Dense(units=__hidden_dim, activation='linear', use_bias=False, kernel_initializer='glorot_uniform')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._w2(self._w1(inputs) * self._w3(inputs))

    def get_config(self) -> dict:
        __parent_config = super(FeedForwardBlock, self).get_config()
        return {**__parent_config, **self._config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# TRANSFORMER #################################################################

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        epsilon: float,
        multiple_of: int,
        ffn_dim_multiplier: float=1.,
        **kwargs
    ) -> None:
        super(TransformerBlock, self).__init__(name=str(layer_id), **kwargs)
        # config
        self._config = {
            'layer_id': layer_id,
            'dim': dim,
            'n_heads': n_heads,
            'epsilon': epsilon,
            'multiple_of': multiple_of,
            'ffn_dim_multiplier': ffn_dim_multiplier,}
        # layers
        self._attention_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, rms_scaling=True)
        self._attention = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=dim // n_heads, value_dim=dim // n_heads, use_bias=False, kernel_initializer='glorot_uniform')
        self._ffn_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, rms_scaling=True)
        self._ffn = FeedForwardBlock(dim=dim, hidden_dim=4 * dim, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        __t = self._attention_norm(inputs)
        __t = inputs + self._attention(query=__t, value=__t, key=__t, use_causal_mask=True)
        return __t + self._ffn(self._ffn_norm(__t))

    def get_config(self) -> dict:
        __parent_config = super(FeedForwardBlock, self).get_config()
        return {**__parent_config, **self._config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
