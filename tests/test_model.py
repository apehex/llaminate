import math

import tensorflow as tf

import tokun.model

import llaminate.model
import llaminate.utils

# WITH CACHE ##################################################################

class CacheTransformerTest(tf.test.TestCase):

    def test_shapes(self):
        __meta = {
            'cache': {
                'num_layers': 2,
                'num_heads': 4,
                'batch_dim': 8,
                'cache_dim': 16,
                'head_dim': 64,},
            'encoder': {
                'sample_dim': 16,
                'token_dim': 128},
            'llaminate': {
                'num_layers': 2,
                'num_heads': 4,
                'embed_dim': 256,
                'head_dim': 64,
                'hidden_dim': 1024,
                'output_dim': 128,
                'epsilon': 1e-6,},}
        # cache
        __c = llaminate.utils.create_cache(**__meta['cache'])
        # transformer
        __m = llaminate.model.CacheTransformer(**__meta['llaminate'])
        # inputs
        __x = tf.ones((__meta['cache']['batch_dim'], __meta['encoder']['sample_dim'], __meta['encoder']['token_dim']))
        # call
        __y = __m.call(inputs=__x, training=False)
        # checks
        self.assertEqual(__y.shape, __x.shape)
        # infer
        __y, __c = __m.infer(inputs=__x, cache=__c, attention_mask=None, position=4, training=False)
        # checks
        self.assertEqual(__y.shape, __x.shape)
        self.assertEqual(len(__c), __meta['llaminate']['num_layers'])
        self.assertEqual(__c[0].shape, (2, __meta['cache']['batch_dim'], __meta['cache']['cache_dim'], __meta['cache']['num_heads'], __meta['cache']['head_dim']))

# WITH CACHE ##################################################################

class TransformerTest(tf.test.TestCase):

    def test_shapes(self):
        __meta = {
            'encoder': {
                'batch_dim': 2,
                'sample_dim': 8,
                'token_dim': 128},
            'llaminate': {
                'num_layers': 2,
                'num_heads': 4,
                'embed_dim': 256,
                'head_dim': 64,
                'hidden_dim': 1024,
                'output_dim': 128,
                'epsilon': 1e-6,},}
        # transformer
        __m = llaminate.model.Transformer(**__meta['llaminate'])
        # inputs
        __x = tf.ones((__meta['encoder']['batch_dim'], __meta['encoder']['sample_dim'], __meta['encoder']['token_dim']))
        # call
        __y = __m.call(inputs=__x, attention_mask=None, training=False)
        # checks
        self.assertEqual(__y.shape, __x.shape)
