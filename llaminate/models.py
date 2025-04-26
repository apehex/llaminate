"""llaminate model."""

import functools

import keras
import tensorflow as tf

import mlable.blocks.transformer
import mlable.layers.embedding
import mlable.layers.shaping

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
        # layers
        self._group = None
        self._embed = None
        self._head = None
        self._split = None
        self._blocks = []

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
        # group the bytes token by token
        self._group = mlable.layers.shaping.Divide(axis=-1, factor=self._config['token_dim'], insert=True, right=True, name='group')
        # the inputs is always UTF-32-BE bytes => 256
        self._embed = mlable.layers.embedding.TokunEmbedding(input_dim=256, output_dim=self._config['embed_dim'] // self._config['token_dim'], name='embed')
        # blocks
        self._blocks = [
            mlable.blocks.transformer.ResidualDecoderBlock(
                head_num=self._config['head_num'],
                key_dim=self._config['head_dim'],
                value_dim=self._config['head_dim'],
                hidden_dim=self._config['hidden_dim'],
                attention_axes=[1],
                dropout_rate=self._config['dropout'],
                epsilon=self._config['epsilon'],
                use_bias=True,
                center=True,
                scale=True,
                name='block-{}'.format(__i))
            for __i in range(self._config['layer_num'])]
        # 8 bits for each input byte
        self._head = tf.keras.layers.Dense(units=8 * self._config['token_dim'], activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='head')
        # flatten the bytes
        self._split = mlable.layers.shaping.Divide(axis=-1, factor=self._config['token_dim'], insert=False, right=False, name='split')
        # build
        for __l in [self._group, self._embed] + self._blocks + [self._head, self._split]:
            __l.build(__shape)
            __shape = __l.compute_output_shape(__shape)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape) + (8,)

    def call(self, inputs: tf.Tensor, logits: bool=True, **kwargs) -> tf.Tensor:
        # group the bytes by token (B, S * T) => (B, S, T)
        __outputs = self._group(inputs)
        # embed the bytes (B, S, T) => (B, S, T * E) = (B, S, L)
        __outputs = self._embed(__outputs)
        # transform (B, S, T * E)
        __outputs = functools.reduce(lambda __x, __b: __b(query=__x, key=__x, value=__x, use_causal_mask=True, **kwargs), self._blocks, __outputs)
        # decompress (B, S, T * E) => (B, S, T * 8)
        __outputs = self._head(__outputs)
        # split the tokens into individual bytes (B, S, T * 8) => (B, S * T, 8)
        __outputs = self._split(__outputs)
        # scale
        return __outputs if logits else tf.nn.softmax(__outputs, axis=-1)

    def get_config(self) -> dict:
        __config = super(Transformer, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
