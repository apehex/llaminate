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
                'cache_dim': 4,
                'head_dim': 64,},
            'tokun': {
                'sequence_axis': 1,
                'feature_axis': -1,
                'token_dim': [4, 4],
                'encoding_dim': 256,
                'embedding_dim': 256,},
            'llaminate': {
                'num_layers': 2,
                'num_heads': 4,
                'cache_dim': 4,
                'embed_dim': 256,
                'hidden_dim': 1024,
                'head_dim': 64,
                'epsilon': 1e-6,},}
        # tokenizer
        __t = tokun.model.AutoEncoder(**__meta['tokun'])
        # cache
        __c = llaminate.utils.create_cache(**__meta['cache'])
        # transformer
        __m = llaminate.model.CacheTransformer(**__meta['llaminate'])
        # set encoder + decoder
        __m.set_tokenizer(encoder=__t._encoder, decoder=__t._decoder)
        # inputs
        __x = tf.ones((__meta['cache']['batch_dim'], math.prod(__meta['tokun']['token_dim']) * __meta['cache']['cache_dim']))
        # call
        __y = __m.call(inputs=__x, training=False)
        # checks
        self.assertEqual(tf.argmax(__y, axis=-1).shape, __x.shape)
        # infer
        __y, __c = __m.infer(inputs=__x, cache=__c, attention_mask=None, position=4, training=False)
        # checks
        self.assertEqual(tf.argmax(__y, axis=-1).shape, __x.shape)
        self.assertEqual(len(__c), __meta['llaminate']['num_layers'])
        self.assertEqual(__c[0].shape, (2, __meta['cache']['batch_dim'], __meta['cache']['cache_dim'], __meta['cache']['num_heads'], __meta['cache']['head_dim']))

# WITH CACHE ##################################################################

class TransformerTest(tf.test.TestCase):

    def test_shapes(self):
        __meta = {
            'tokun': {
                'sequence_axis': 1,
                'feature_axis': -1,
                'token_dim': [4, 4],
                'encoding_dim': 256,
                'embedding_dim': 256,},
            'llaminate': {
                'num_layers': 2,
                'num_heads': 4,
                'cache_dim': 4,
                'embed_dim': 256,
                'hidden_dim': 1024,
                'head_dim': 64,
                'epsilon': 1e-6,},}
        # tokenizer
        __t = tokun.model.AutoEncoder(**__meta['tokun'])
        # transformer
        __m = llaminate.model.Transformer(**__meta['llaminate'])
        # set encoder + decoder
        __m.set_tokenizer(encoder=__t._encoder, decoder=__t._decoder)
        # inputs
        __x = tf.ones((1, math.prod(__meta['tokun']['token_dim']) * __meta['llaminate']['cache_dim']))
        # call
        __y = __m.call(inputs=__x, attention_mask=None, training=False)
        # checks
        self.assertEqual(tf.argmax(__y, axis=-1).shape, __x.shape)
