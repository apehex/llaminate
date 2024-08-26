"""llaminate model."""

import functools

import keras
import tensorflow as tf

import llaminate.layers

# CONSTANTS ###################################################################

EPSILON = 1e-5

# WITH CACHE ##################################################################

@keras.saving.register_keras_serializable(package='models')
class CacheTransformer(tf.keras.models.Model):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        embed_dim: int,
        head_dim: int,
        hidden_dim: int,
        output_dim: int,
        epsilon: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(CacheTransformer, self).__init__(**kwargs)
        # config
        self._config = {
            'num_layers': num_layers,
            'num_heads': num_heads,
            'embed_dim': embed_dim,
            'head_dim': head_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'epsilon': epsilon,}
        # layers
        self._tail = tf.keras.layers.Dense(units=embed_dim, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='tail')
        self._blocks = [
            llaminate.layers.CacheDecoderBlock(
                num_heads=num_heads,
                embed_dim=embed_dim,
                head_dim=head_dim,
                hidden_dim=hidden_dim,
                sequence_axis=1,
                epsilon=epsilon,
                name='block-{}'.format(__i))
            for __i in range(num_layers)]
        self._head = tf.keras.layers.Dense(units=output_dim, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='head')

    def call(self, inputs: tuple, attention_mask: tf.Tensor=None, **kwargs) -> tf.Tensor:
        # embed
        __y = self._tail(inputs)
        # blocks
        __y = functools.reduce(lambda __x, __b: __b(inputs=__x, attention_mask=attention_mask, position=0, training=True, cache=None)[0], self._blocks, __y)
        # decompress
        return self._head(__y)

    def infer(
        self,
        inputs: tf.Tensor,
        attention_mask: tf.Tensor=None,
        cache: list=None,
        position: int=0,
        **kwargs,
    ) -> tuple:
        # init
        __cache = self._config['num_layers'] * [None] if cache is None else cache
        # embed
        __y = self._tail(inputs)
        # blocks
        for __i, __block in enumerate(self._blocks):
            __y, __cache[__i] = __block(inputs=__y, cache=__cache[__i], attention_mask=attention_mask, position=position, training=False)
        # decompress
        __y = self._head(__y)
        # used in inference only
        return (__y, __cache)

    def get_config(self) -> dict:
        __config = super(CacheTransformer, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# WITHOUT CACHE ###############################################################

@keras.saving.register_keras_serializable(package='models')
class Transformer(tf.keras.models.Model):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        embed_dim: int,
        head_dim: int,
        hidden_dim: int,
        output_dim: int,
        epsilon: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(Transformer, self).__init__(**kwargs)
        # config
        self._config = {
            'num_layers': num_layers,
            'num_heads': num_heads,
            'embed_dim': embed_dim,
            'head_dim': head_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'epsilon': epsilon,}
        # layers
        self._tail = tf.keras.layers.Dense(units=embed_dim, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='tail')
        self._blocks = [
            llaminate.layers.DecoderBlock(
                num_heads=num_heads,
                embed_dim=embed_dim,
                head_dim=head_dim,
                hidden_dim=hidden_dim,
                sequence_axis=1,
                epsilon=epsilon,
                name='block-{}'.format(__i))
            for __i in range(num_layers)]
        self._head = tf.keras.layers.Dense(units=output_dim, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='head')

    def call(self, inputs: tuple, attention_mask: tf.Tensor=None, **kwargs) -> tf.Tensor:
        # embed
        __y = self._tail(inputs)
        # blocks
        __y = functools.reduce(lambda __x, __b: __b(inputs=__x, attention_mask=attention_mask, **kwargs), self._blocks, __y)
        # decompress
        return self._head(__y)

    def get_config(self) -> dict:
        __config = super(Transformer, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
