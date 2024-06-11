import math

import numpy as np
import tensorflow as tf

import tokun.model

import llaminate.model
import llaminate.utils

# FF ##########################################################################

class TransformerTest(tf.test.TestCase):

    def test_shapes(self):
        """Test with a mask tensor."""
        __meta = {
            'num_layers': 2,
            'num_heads': 4,
            'batch_dim': 8,
            'token_dim': [4, 4],
            'cache_dim': 4,
            'embed_dim': 256,
            'hidden_dim': 1024,
            'head_dim': 64,
            'hidden_dim': 1024,
            'epsilon': 1e-6,}
        # tokenizer
        __t = tokun.model.AutoEncoder(sequence_axis=1, feature_axis=-1, token_dim=__meta['token_dim'], encoding_dim=__meta['embed_dim'], embedding_dim=__meta['embed_dim'], hidden_dim=__meta['hidden_dim'], latent_dim=__meta['embed_dim'], batch_dim=__meta['batch_dim'], gate=False, normalization=True)
        # cache
        __c = llaminate.utils.create_cache(batch_dim=__meta['batch_dim'], cache_dim=__meta['cache_dim'], head_dim=__meta['head_dim'], num_layers=__meta['num_layers'], num_heads=__meta['num_heads'])
        # transformer
        __m = llaminate.model.Transformer(num_layers=__meta['num_layers'], num_heads=__meta['num_heads'], cache_dim=__meta['cache_dim'], embed_dim=__meta['embed_dim'], head_dim=__meta['head_dim'], hidden_dim=__meta['hidden_dim'], epsilon=__meta['epsilon'])
        # set encoder + decoder
        __m.set_tokenizer(encoder=__t._encoder, decoder=__t._decoder)
        # inputs
        __x = tf.ones((__meta['batch_dim'], math.prod(__meta['token_dim']) * __meta['cache_dim'], __meta['embed_dim']))
        # call
        __c, __y = __m(inputs=__x, cache=__c, mask=None, position=4)
        # checks
        self.assertEqual(__y.shape, __x.shape)
        self.assertEqual(len(__c), __meta['num_layers'])
        self.assertEqual(__c[0].shape, (2, __meta['batch_dim'], __meta['cache_dim'], __meta['num_heads'], __meta['head_dim']))
