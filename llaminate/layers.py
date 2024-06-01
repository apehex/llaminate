"""Building blocks of llaminate."""

import keras
import tensorflow as tf

import mlable.layers.embedding

# FEED FORWARD ################################################################

@keras.saving.register_keras_serializable(package='blocks')
class FeedForwardBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        **kwargs
    ) -> None:
        super(FeedForwardBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,}
        # layers
        self._gelu = tf.keras.layers.Dense(units=self._config['hidden_dim'], activation='gelu', use_bias=False, kernel_initializer='zeros', name='gate')
        self._linear = tf.keras.layers.Dense(units=self._config['hidden_dim'], activation='linear', use_bias=False, kernel_initializer='zeros', name='linear')
        self._output = tf.keras.layers.Dense(units=self._config['input_dim'], activation='linear', use_bias=False, kernel_initializer='zeros', name='output')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # gating mechanism
        return self._output(self._gelu(inputs) * self._linear(inputs))

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
