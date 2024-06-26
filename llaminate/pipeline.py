import tensorflow as tf

# DISTRIBUTE ##################################################################

_map = lambda __f: (lambda *__t: tuple(map(__f, __t)))

# PREPROCESS ##################################################################

def preprocess(dataset: tf.data.Dataset, token_dim: int, embed_dim: int, sample_dim: int, features: list=[]) -> tf.data.Dataset:
    __pipeline = [
        # join the features
        ((lambda __x: tf.strings.join(inputs=[__x[__f] for __f in features], separator='\x1d') if features else __x), True),
        # (input, target) where target is the next character for each input
        ((lambda __x: (tokun.pipeline.offset(data=__x, ticks=16), __x)), True),
        # encode => (4 * S,) int
        (_map(functools.partial(tokun.pipeline.encode, token_size=token_dim, sample_size=sample_dim)), True),
        # reshape => (4 * S,) int
        (_map(functools.partial(tf.reshape, shape=(4 * sample_dim,))), True),
        # one-hot encoding for the targets => (4 * S, E) int (bool)
        ((lambda __x, __y: (__x, tf.one_hot(__y, depth=embed_dim, axis=-1))), True)]
    # split args
    __operations, __replace = zip(*__pipeline)
    # udpate dataset
    return mlable.data.process(dataset=dataset, pipeline=__operations, replace=__replace)
