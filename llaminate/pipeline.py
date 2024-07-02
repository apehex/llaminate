import functools

import tensorflow as tf

import mlable.data
import tokun.pipeline

# DISTRIBUTE ##################################################################

_map = lambda __f: (lambda *__t: tuple(map(__f, __t)))

# INDIVIDUAL OPERATIONS #######################################################

def _combine(inputs: tf.Tensor, features: list=[], separator: str='\x1d') -> tf.Tensor:
    __inputs = [inputs[__f] for __f in features] if features else [inputs]
    return tf.strings.join(inputs=__inputs, separator=separator)

def _offset(inputs: tf.Tensor, ticks: int) -> tuple:
    return (tokun.pipeline.offset(data=inputs, ticks=ticks), inputs)

def _reshape(input: tf.Tensor, sample_dim: int, batch_dim: int=None) -> tf.Tensor:
    __shape = (batch_dim, 4 * sample_dim) if batch_dim else (4 * sample_dim,)
    return tf.reshape(inputs, shape=__shape)

def _embed(xy: tuple, embed_dim: int, encoder: callable=None) -> tuple:
    __x, __y = xy
    if encoder is not None:
        __x, __y = encoder(__x), encoder(__y)
    else:
        __x, __y = __x, tf.one_hot(__y, depth=embed_dim, axis=-1)
    return (__x, __y)

# PREPROCESS ##################################################################

def preprocess(dataset: tf.data.Dataset, token_dim: int, embed_dim: int, sample_dim: int, features: list=[], separator: str='\x1d', batch_dim: int=None, encoder: callable=None) -> tf.data.Dataset:
    # chain the operations
    __pipeline = [
        # combine the features
        (functools.partial(_combine, features=features, separator=separator), True),
        # (input, target) where target is the next character for each input
        (functools.partial(_offset, ticks=token_dim // 4), True),
        # encode => (4 * S,) int
        (_map(functools.partial(tokun.pipeline.encode, token_size=token_dim, sample_size=sample_dim)), True),
        # reshape => (4 * S,) int
        (_map(functools.partial(_reshape, batch_dim=batch_dim, sample_dim=sample_dim)), True),
        # one-hot encoding for the targets => (4 * S, E) int (bool)
        (functools.partial(_embed, embed_dim=embed_dim, encoder=encoder), True)]
    # split args
    __operations, __replace = zip(*__pipeline)
    # udpate dataset
    return mlable.data.process(dataset=dataset, pipeline=__operations, replace=__replace)
