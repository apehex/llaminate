"""Building blocks of llaminate."""

import keras
import tensorflow as tf

import mlable.layers.embedding
import mlable.layers.transformer

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
        __config = super(FeedForwardBlock, self).get_config()
        __config.update(self._config)
        return __config

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
        embed_dim: int,
        head_dim: int,
        hidden_dim: int,
        epsilon: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(DecoderBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'num_heads': num_heads,
            'embed_dim': embed_dim,
            'head_dim': head_dim,
            'hidden_dim': hidden_dim,
            'epsilon': epsilon,}
        # layers
        self._attention_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, rms_scaling=True, gamma_initializer='ones') # RMS
        self._position = mlable.layers.embedding.RotaryPositionalEmbedding(sequence_axis=1, feature_axis=-1)
        self._attention = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=num_heads, key_dim=head_dim, value_dim=head_dim, use_bias=False, kernel_initializer='glorot_uniform')
        self._ffn_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, rms_scaling=True, gamma_initializer='ones') # RMS
        self._ffn = FeedForwardBlock(input_dim=embed_dim, hidden_dim=hidden_dim)

    def call(self, inputs: tf.Tensor, cache: tf.Tensor, mask: tf.Tensor=None, position: int=0) -> tf.Tensor:
        # residual
        __x = inputs
        # normalize
        __y = self._attention_norm(__x)
        # position embedding
        __yp = self._position(inputs=__y, offset=position)
        # attention
        __y, __cache = self._attention(key=__yp, query=__yp, value=__y, cache=cache, step=position, attention_mask=mask, use_causal_mask=True)
        # residual
        __x = __y + __x
        # normalize
        __y = self._ffn_norm(__x)
        # augment
        __y = self._ffn(__y)
        # residual
        return __y + __x, __cache

    def get_config(self) -> dict:
        __config = super(DecoderBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
