import numpy as np
import tensorflow as tf

import tokun.model

import llaminate.model

# FF ##########################################################################

class TransformerTest(tf.test.TestCase):

    def test_shapes(self):
        """Test with a mask tensor."""
        __num_layers, __num_heads, __batch_dim, __seq_dim, __head_dim, __hidden_dim = 2, 4, 8, 4, 4, 64
        # __t = tokun.model.AutoEncoder(token_dim=[4, 4], encoding_dim=256, embedding_dim=256, latent_dim=256, batch_dim=__batch_dim, attention=True, normalization=True)
        # __c = llaminate.model.create_cache(batch_dim=__batch_dim, cache_dim=__seq_dim, head_dim=__head_dim, num_layers=__num_layers, num_heads=__num_heads)
        # __m = llaminate.model.Transformer(num_layers=__num_layers, num_heads=__num_heads, cache_dim=__seq_dim, embed_dim=256, head_dim=__head_dim, hidden_dim=__hidden_dim, epsilon=1e-5)
        # __m.load_tokenizer(encoder=__t._encoder, decoder=__t._decoder)
        # __x = tf.ones((__batch_dim, __seq_dim, __num_heads * __head_dim))
        __x = tf.ones()
        # GPU/CPU case.
        __init_dim = 0
        # Directly tests the keras layer.
        __cache = mlable.utils.create_cache(__batch_dim, __init_dim, __num_heads, __head_dim)
        # basic attention
        __layer = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=__num_heads, key_dim=__head_dim)
        # input data
        __from_inputs = tf.zeros((__batch_dim, __seq_dim, 8), dtype=np.float32)
        # random mask
        __mask = np.random.randint(2, size=(__batch_dim, __seq_dim, __seq_dim))
        # call
        __output, __cache = __layer(query=__from_inputs, value=__from_inputs, cache=__cache, attention_mask=__mask)
        # checks
        self.assertEqual(__output.shape, (3, 4, 8))
        self.assertEqual(__cache.shape, (2, 3, 4, 2, 2))
        # without cache
        __output, __cache = __layer(query=__from_inputs, value=__from_inputs, attention_mask=__mask)
        self.assertEqual(__output.shape, (3, 4, 8))
        self.assertEqual(__cache.shape, (2, 3, 4, 2, 2))
