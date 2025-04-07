"""llaminate model."""

import functools

import keras
import tensorflow as tf

import mlable.layers.embedding

import llaminate.layers

# CONSTANTS ###################################################################

EPSILON = 1e-5

# BASE TRANSFORMER #############################################################

@keras.saving.register_keras_serializable(package='models')
class Transformer(tf.keras.models.Model):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        input_dim: int,
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
            'input_dim': input_dim,
            'embed_dim': embed_dim,
            'head_dim': head_dim,
            'hidden_dim': hidden_dim,
            'epsilon': epsilon,}
        # the inputs is always UTF-32-BE bytes => 256
        self._embed = mlable.layers.embedding.TokunEmbedding(input_dim=256, output_dim=embed_dim // input_dim, name='embed')
        # blocks
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
        # 8 bits for each input byte
        self._head = tf.keras.layers.Dense(units=8 * input_dim, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='head')

    def call(self, inputs: tf.Tensor, logits: bool=True, **kwargs) -> tf.Tensor:
        # embed
        __outputs = self._embed(inputs)
        # blocks
        __outputs = functools.reduce(lambda __x, __b: __b(inputs=__x, use_causal_mask=True, **kwargs), self._blocks, __outputs)
        # decompress
        __outputs = self._head(__outputs)
        # scale
        return __outputs if logits else tf.nn.softmax(__outputs, axis=-1)

    def get_config(self) -> dict:
        __config = super(Transformer, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
