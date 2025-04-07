"""llaminate model."""

import functools

import keras
import tensorflow as tf

import mlable.blocks.transformer
import mlable.layers.embedding

import llaminate.layers

# CONSTANTS ###################################################################

EPSILON = 1e-5
DROPOUT = 0.0

# BASE TRANSFORMER #############################################################

@keras.saving.register_keras_serializable(package='models')
class Transformer(tf.keras.models.Model):
    def __init__(
        self,
        layer_num: int,
        head_num: int,
        token_dim: int,
        embed_dim: int,
        head_dim: int,
        hidden_dim: int,
        epsilon: float=EPSILON,
        dropout: float=DROPOUT,
        **kwargs
    ) -> None:
        # init
        super(Transformer, self).__init__(**kwargs)
        # config
        self._config = {
            'layer_num': layer_num,
            'head_num': head_num,
            'token_dim': token_dim,
            'embed_dim': embed_dim,
            'head_dim': head_dim,
            'hidden_dim': hidden_dim,
            'epsilon': epsilon,
            'dropout': dropout,}
        # the inputs is always UTF-32-BE bytes => 256
        self._embed = mlable.layers.embedding.TokunEmbedding(input_dim=256, output_dim=embed_dim // token_dim, name='embed')
        # blocks
        self._blocks = [
            mlable.blocks.transformer.ResidualDecoderBlock(
                head_num=head_num,
                key_dim=head_dim,
                value_dim=head_dim,
                hidden_dim=hidden_dim,
                attention_axes=[1],
                dropout_rate=dropout,
                epsilon=epsilon,
                use_bias=True,
                center=True,
                scale=True,
                name='block-{}'.format(__i))
            for __i in range(layer_num)]
        # 8 bits for each input byte
        self._head = tf.keras.layers.Dense(units=8 * token_dim, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='head')

    def call(self, inputs: tf.Tensor, logits: bool=True, **kwargs) -> tf.Tensor:
        # embed
        __outputs = self._embed(inputs)
        # blocks
        __outputs = functools.reduce(lambda __x, __b: __b(query=__x, key=__x, value=__x, use_causal_mask=True, **kwargs), self._blocks, __outputs)
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
