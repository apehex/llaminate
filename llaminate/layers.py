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

# DECODER #####################################################################

EPSILON = 1e-5

@keras.saving.register_keras_serializable(package='blocks')
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        embed_dim: int,
        head_dim: int,
        hidden_dim: int,
        **kwargs
    ) -> None:
        # init
        super(DecoderBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'num_heads': num_heads,
            'num_kv_heads': num_kv_heads,
            'embed_dim': embed_dim,
            'head_dim': head_dim,
            'hidden_dim': hidden_dim,}
        # layers
        self._attention_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=EPSILON, rms_scaling=True, gamma_initializer='ones') # RMS
        self._position = mlable.layers.embedding.RotaryPositionalEmbedding(head_dim=head_dim)
        self._attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_dim, value_dim=head_dim, use_bias=False, kernel_initializer='glorot_uniform')
        self._ffn_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=EPSILON, rms_scaling=True, gamma_initializer='ones') # RMS
        self._ffn = FeedForwardBlock(input_dim=embed_dim, hidden_dim=hidden_dim)

    def call(self, inputs: tf.Tensor, positions: tf.Tensor, cache: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        # residual
        __x = inputs
        # normalize
        __y = self._attention_norm(__x)
        # position embedding
        __y = self._position(inputs=__y, positions=positions)
        # attention
        __y = self._attention(key=__y, query=__y, value=__y, use_causal_mask=True)
        # residual
        __x = __y + __x
        # normalize
        __y = self._ffn_norm(__x)
        # augment
        __y = self._ffn(__y)
        # residual
        return __y + __x

    def get_config(self) -> dict:
        __parent_config = super(DecoderBlock, self).get_config()
        return {**__parent_config, **self._config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
