import functools

import tensorflow as tf

import mlable.utils
import tokun.pipeline

# DISTRIBUTE ##################################################################

_map = lambda __f: (lambda *__t: tuple(map(__f, __t)))

# INDIVIDUAL OPERATIONS #######################################################

def _combine(inputs: tf.Tensor, features: list, separator: str='\x1d') -> tf.Tensor:
    return tf.strings.join(inputs=[inputs[__f] for __f in features], separator=separator)

def _offset(inputs: tf.Tensor, ticks: int) -> tuple:
    return (tokun.pipeline.offset(data=inputs, ticks=ticks), inputs)

def _reshape(inputs: tf.Tensor, sample_dim: int, batch_dim: int=None) -> tf.Tensor:
    __shape = (batch_dim, 4 * sample_dim) if batch_dim else (4 * sample_dim,)
    return tf.reshape(inputs, shape=__shape)

def _embed(inputs: tf.Tensor, targets: tf.Tensor, embed_dim: int, encoder: callable=None) -> tuple:
    __x, __y = inputs, tf.one_hot(targets, depth=embed_dim, axis=-1)
    if encoder is not None:
        __x, __y = encoder(inputs), encoder(targets)
    return (__x, __y)

# PREPROCESS ##################################################################

def preprocess(inputs: tf.Tensor, token_dim: int, embed_dim: int, sample_dim: int, features: list, separator: str='\x1d', batch_dim: int=None, encoder: callable=None) -> tf.data.Dataset:
    # combine the features
    __outputs = _combine(inputs=inputs, features=features, separator=separator)
    # (input, target) where target is the next character for each input
    __outputs = _offset(inputs=__outputs, ticks=token_dim // 4)
    # encode => (4 * S,) int
    __encode = functools.partial(tokun.pipeline.encode, token_size=token_dim, sample_size=sample_dim)
    __outputs = tuple(map(__encode, __outputs))
    # reshape => (4 * S,) int
    __reshape = functools.partial(_reshape, batch_dim=batch_dim, sample_dim=sample_dim)
    __outputs = tuple(map(__reshape, __outputs))
    # one-hot encoding for the targets => (4 * S, E) int (bool)
    __embed = functools.partial(_embed, embed_dim=embed_dim, encoder=encoder)
    __outputs = __embed(*__outputs)
    # chain the operations
    return __outputs
