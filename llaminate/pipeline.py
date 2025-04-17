import functools
import math

import tensorflow as tf

import mlable.maths.ops
import mlable.text

# MASK ########################################################################

def mask(data: tf.Tensor, encoding_dim: int, padding_value: int=0, padding_weight: float=0.0, data_weight: float=1.0, dtype: tf.DType=tf.float32) -> tf.Tensor:
    # byte level mask (B, S * T)
    __weights = tf.not_equal(data, padding_value)
    # token level mask, but expressed byte by byte
    __weights = mlable.maths.ops.reduce_any(data=__weights, group=encoding_dim, axis=-1, keepdims=True)
    # cast from bool to allow multiplications
    __weights = tf.cast(__weights, dtype=dtype)
    # rescale the weights
    return data_weight * __weights + padding_weight * (1. - __weights)

# PREPROCESS ##################################################################

def _parser_factory(token_dim: int, features: list, separator: str='\x1d', drop_dim: int=0, encoding_dim: int=4) -> callable:
    # number of characters per token
    __ticks = token_dim // max(1, encoding_dim - drop_dim)
    # wrapper
    def __parser(inputs) -> tuple:
        # fetch the relevant features
        __inputs = tf.strings.join(inputs=[inputs[__f] for __f in features], separator=separator)
        # (input, target) where target is the next token for each input
        return (mlable.text.offset(data=__inputs, ticks=__ticks), __inputs)
    # customized fn
    return __parser

def _encoder_factory(sample_dim: int, encoding: str='UTF-32-BE') -> callable:
    # text encoding (UTF-32-BE or UTF-8)
    __utf = functools.partial(mlable.text.encode, sample_dim=sample_dim, output_dtype=tf.uint8, output_encoding=encoding)
    # encode all
    def __encoder(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (__utf(inputs), __utf(targets))
    # customized fn
    return __encoder

def _formatter_factory(batch_dim: int, sample_dim: int, drop_dim: int=0, encoding_dim: int=4) -> callable:
    # sample dimension after trimming
    __dim = max(1, encoding_dim - drop_dim) * (sample_dim // encoding_dim)
    # remove the leading 0s in UTF-32-BE
    __trim = functools.partial(mlable.text.trim, count=drop_dim, outof=encoding_dim)
    # enforce types
    __cast_i = functools.partial(tf.cast, dtype=tf.int32)
    __cast_t = functools.partial(tf.cast, dtype=tf.float32)
    # enforce shapes
    __reshape = functools.partial(tf.reshape, shape=(batch_dim, __dim))
    # chain the operations
    def __formatter(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (__cast_i(__reshape(__trim(inputs))), __cast_t(__reshape(__trim(targets))))
    # customized fn
    return __formatter

def _embedder_factory() -> callable:
    # embed all
    def __embedder(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (inputs, mlable.maths.ops.expand_base(targets, base=2, depth=8))
    # customized fn
    return __embedder

def _masker_factory(encoding_dim: int, data_weight: float=1.0, padding_weight: float=0.0) -> callable:
    def __masker(inputs: tf.Tensor) -> tf.Tensor:
        return mask(data=inputs, encoding_dim=encoding_dim, data_weight=data_weight, padding_weight=padding_weight, padding_value=0, dtype=tf.float32)
    # customized fn
    return __masker

# > END-TO-END ################################################################

def _preprocess(inputs: tf.Tensor, parser: callable, encoder: callable, embedder: callable, masker: callable, formatter: callable) -> tuple:
    # fetch the relevant features
    __inputs, __targets = parser(inputs=inputs)
    # encode / tokenize
    __inputs, __targets = encoder(inputs=__inputs, targets=__targets)
    # enforce types + shapes
    __inputs, __targets = formatter(inputs=__inputs, targets=__targets)
    # embed in binary
    __inputs, __targets = embedder(inputs=__inputs, targets=__targets)
    # sequence mask to ignore padding during training
    __weights = masker(inputs=__inputs)
    # pack both sourcecode and bytecode into the model inputs
    return (__inputs, __targets, __weights)

def preprocess_factory(batch_dim: int, sample_dim: int, token_dim: int, features: list, separator: str='\x1d', encoding: str='UTF-32-BE', data_weight: float=1.0, padding_weight: float=0.0, drop_dim: int=0) -> callable:
    __encoding_dim = 4 if '32' in encoding else 1
    # custom fn
    __parser = _parser_factory(token_dim=token_dim, drop_dim=drop_dim, encoding_dim=__encoding_dim, features=features, separator=separator)
    __encoder = _encoder_factory(sample_dim=sample_dim, encoding=encoding)
    __formatter = _formatter_factory(batch_dim=batch_dim, sample_dim=sample_dim, drop_dim=drop_dim, encoding_dim=__encoding_dim)
    __embedder = _embedder_factory()
    __masker = _masker_factory(encoding_dim=__encoding_dim, data_weight=data_weight, padding_weight=padding_weight)
    # actual preprocessing function
    return functools.partial(_preprocess, parser=__parser, encoder=__encoder, embedder=__embedder, masker=__masker, formatter=__formatter)

# < ###########################################################################

def postprocess(logits: tf.Tensor, threshold: float=0.0, temp: float=1.0, topp: float=0.0, topk: int=0, seed: int=None, dtype: tf.DType=tf.int32, encoding: str='UTF-32-BE') -> tf.Tensor:
    # values encoded as binary arrays
    __bytes = mlable.sampling.binary(logits=logits, threshold=threshold, temp=temp, topp=topp, topk=topk, seed=seed, dtype=dtype)
    # decode the bytes into strings
    return mlable.text.postprocess(data=__bytes, encoding=encoding)
