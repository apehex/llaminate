"""llaminate model."""

import keras
import tensorflow as tf

import llaminate.layers

# TRANSFORMER #################################################################

EPSILON = 1e-5

@keras.saving.register_keras_serializable(package='models')
class Transformer(tf.keras.models.Model):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        cache_dim: int,
        embed_dim: int,
        head_dim: int,
        hidden_dim: int,
        epsilon: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(Transformer, self).__init__(**kwargs)
        # config
        self._config = {
            'num_layers': num_layers,
            'num_heads': num_heads,
            'cache_dim': cache_dim,
            'embed_dim': embed_dim,
            'head_dim': head_dim,
            'hidden_dim': hidden_dim,
            'epsilon': epsilon,}
        # layers
        self._encoder = None
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
        self._norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, beta_initializer='zeros', gamma_initializer='ones') # rms_scaling=True, 
        self._decoder = None

    def call(self, inputs: tf.Tensor, cache: list=None, mask: tf.Tensor=None, position: int=0) -> tf.Tensor:
        # init
        __cache = cache
        # byte embedding
        __y = self._encoder(inputs)
        # blocks
        for __i, __block in enumerate(self._blocks):
            __layer_cache = __cache[__i] if __cache is not None else None
            __layer_cache, __y = __block(inputs=__y, cache=__layer_cache, mask=mask, position=position)
            # update the cache
            if __cache is not None:
                __cache[__i] = __layer_cache
        # normalize
        __y = self._norm(__y)
        # decompress
        return __cache, self._decoder(__y)

    def set_tokenizer(self, encoder: tf.keras.models.Model, decoder: tf.keras.models.Model) -> None:
        # set the weights
        self._encoder = encoder
        self._decoder = decoder

    def get_config(self) -> dict:
        __config = super(Transformer, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
