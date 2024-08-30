import functools
import math

import tensorflow as tf

import mlable.ops
import tokun.pipeline
import tokun.model

# MASK ########################################################################

def mask(data: tf.Tensor, padding_value: int=0, padding_weight: float=0.0, data_weight: float=1.0, dtype: tf.dtypes.DType=tf.float32) -> tf.Tensor:
    # byte level mask
    __weights = tf.not_equal(data, padding_value)
    # instruction level mask, but expressed byte by byte
    __weights = mlable.ops.reduce_any(data=__weights, group=None, axis=-1, keepdims=False)
    # cast from bool to allow multiplications
    __weights = tf.cast(__weights, dtype=dtype)
    # rescale the weights
    return data_weight * __weights + padding_weight * (1. - __weights)

# BINARIZE ####################################################################

def binarize(data: tf.Tensor, input_dim: int) -> tf.Tensor:
    __depth = int(math.log(input_dim, 2))
    #  decompose in base 2
    __output = mlable.ops.expand_base(data, base=2, depth=__depth)
    # merge all the bits in a single sequence
    return mlable.ops.merge(__output, left_axis=-2, right_axis=-1, left=True)

# PREPROCESS ##################################################################

def _parser_factory(token_dim: int, features: list, separator: str='\x1d', output_dtype: tf.dtypes.DType=tf.int32) -> callable:
    # length of the encoding of each character (IE each character = 4 bytes = 1 codepoint in UTF-32-BE)
    __factor = 1 if output_dtype == tf.int32 else 4
    def __parser(inputs) -> tuple:
        # fetch the relevant features
        __inputs = tf.strings.join(inputs=[inputs[__f] for __f in features], separator=separator)
        # (input, target) where target is the next token for each input
        return (tokun.pipeline.offset(data=__inputs, ticks=token_dim // __factor), __inputs)
    # customized fn
    return __parser

def _encoder_factory(token_dim: int, sample_dim: int, output_dtype: tf.dtypes.DType=tf.int32) -> callable:
    # text encoding (UTF-32-BE)
    __utf32 = functools.partial(tokun.pipeline.encode, token_size=token_dim, sample_size=sample_dim, output_dtype=output_dtype)
    # encode all
    def __encoder(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (__utf32(inputs), __utf32(targets))
    # customized fn
    return __encoder

def _formatter_factory(batch_dim: int, sample_dim: int, token_dim: int, output_dtype: tf.dtypes.DType=tf.int32) -> callable:
    # length of each encoded value in bytes
    __factor = 4 if output_dtype == tf.int32 else 1
    # enforce types
    __cast_i = functools.partial(tf.cast, dtype=tf.int32)
    __cast_t = functools.partial(tf.cast, dtype=tf.float32)
    # enforce shapes
    __reshape = functools.partial(tf.reshape, shape=(batch_dim, sample_dim // (__factor * token_dim), token_dim))
    # chain the operations
    def __formatter(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (__cast_i(__reshape(inputs)), __cast_t(__reshape(targets)))
    # customized fn
    return __formatter

def _embedder_factory(input_dim: int) -> callable:
    # decompose the codepoints in base 2
    __binarize = functools.partial(binarize, input_dim=input_dim)
    # embed all
    def __embedder(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (inputs, __binarize(targets))
    # customized fn
    return __embedder

def _masker_factory(data_weight: float=1.0, padding_weight: float=0.0) -> callable:
    def __masker(inputs: tf.Tensor) -> tf.Tensor:
        return mask(data=inputs, padding_value=0, data_weight=data_weight, padding_weight=padding_weight, dtype=tf.float32)
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
    # embed with tokun
    __inputs, __targets = embedder(inputs=__inputs, targets=__targets)
    # sequence mask to ignore padding during training
    __weights = masker(inputs=__inputs)
    # pack both sourcecode and bytecode into the model inputs
    return (__inputs, __targets, __weights)

def preprocess_factory(batch_dim: int, sample_dim: int, token_dim: int, input_dim: int, features: list, separator: str='\x1d', data_weight: float=1.0, padding_weight: float=0.0, output_dtype: tf.dtypes.DType=tf.int32) -> callable:
    # custom fn
    __parser = _parser_factory(token_dim=token_dim, features=features, separator=separator, output_dtype=output_dtype)
    __encoder = _encoder_factory(sample_dim=sample_dim, token_dim=token_dim, output_dtype=output_dtype)
    __embedder = _embedder_factory(input_dim=input_dim)
    __formatter = _formatter_factory(batch_dim=batch_dim, sample_dim=sample_dim, token_dim=token_dim, output_dtype=output_dtype)
    __masker = _masker_factory(data_weight=data_weight, padding_weight=padding_weight)
    # actual preprocessing function
    return functools.partial(_preprocess, parser=__parser, encoder=__encoder, embedder=__embedder, masker=__masker, formatter=__formatter)

# < ###########################################################################

def postprocess(prediction: tf.Tensor, token_dim: int, input_dim: int=256, random: bool=False) -> tf.Tensor:
    # values encoded as binary arrays
    __depth = int(math.log(input_dim, 2))
    __output = mlable.sampling.binary(prediction=__output, depth=__depth, threshold=0.5, random=random)
    # merge the token and sequence axes
    __output = mlable.ops.merge(__output, left_axis=-2, right_axis=-1, left=True)
    # merge the bytes into codepoints
    if input_dim == 256:
        __output = codepoint(data=__output)
    # decode the UTF-32-BE codepoints
    return tokun.pipeline.decode(data=__output)
