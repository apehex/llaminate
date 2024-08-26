"""Building blocks of llaminate."""

import keras
import tensorflow as tf

import mlable.blocks.transformer

# CONSTANTS ###################################################################

EPSILON = 1e-5

# WITH CACHE ##################################################################

@keras.saving.register_keras_serializable(package='blocks')
class CacheDecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        head_dim: int,
        hidden_dim: int,
        sequence_axis: int=1,
        epsilon: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(CacheDecoderBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'num_heads': num_heads,
            'embed_dim': embed_dim,
            'head_dim': head_dim,
            'hidden_dim': hidden_dim,
            'epsilon': epsilon,}
        # layers
        self._attention = mlable.blocks.transformer.CachedSelfAttentionBlock(num_heads=num_heads, head_dim=head_dim, sequence_axis=sequence_axis, epsilon=epsilon, center=False, scale=False)
        self._ffn = mlable.blocks.transformer.FeedForwardBlock(embed_dim=embed_dim, hidden_dim=hidden_dim, epsilon=epsilon, center=False, scale=False)

    def call(
        self,
        inputs: tf.Tensor,
        cache: tf.Tensor=None,
        position: int=0,
        attention_mask: tf.Tensor=None,
        training: bool=False,
    ) -> tf.Tensor:
        # self attention
        __x, __cache = self._attention(inputs=inputs, cache=cache, position=position, attention_mask=attention_mask, training=training, use_causal_mask=True)
        # residual + augmentation
        __x = inputs + __x + self._ffn(inputs + __x)
        # return values and cache
        return __x, __cache

    def get_config(self) -> dict:
        __config = super(CacheDecoderBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# WITHOUT CACHE ###############################################################

@keras.saving.register_keras_serializable(package='blocks')
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        head_dim: int,
        hidden_dim: int,
        sequence_axis: int=1,
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
        self._attention = mlable.blocks.transformer.SelfAttentionBlock(num_heads=num_heads, head_dim=head_dim, sequence_axis=sequence_axis, epsilon=epsilon, center=False, scale=False)
        self._ffn = mlable.blocks.transformer.FeedForwardBlock(embed_dim=embed_dim, hidden_dim=hidden_dim, epsilon=epsilon, center=False, scale=False)

    def call(
        self,
        inputs: tf.Tensor,
        attention_mask: tf.Tensor=None,
        training: bool=False,
    ) -> tf.Tensor:
        # residual + self attention
        __x = inputs + self._attention(inputs=inputs, attention_mask=attention_mask, training=training, use_causal_mask=True)
        # residual + augmentation
        return __x + self._ffn(__x)

    def get_config(self) -> dict:
        __config = super(DecoderBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
