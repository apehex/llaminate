import math

import tensorflow as tf

import mlable.utils
import tokun.model

import llaminate.model
import llaminate.utils

# WITH CACHE ##################################################################

class CacheTransformerTest(tf.test.TestCase):
    def setUp(self):
        self._config_cache = {
            'num_layers': 2,
            'num_heads': 4,
            'batch_dim': 8,
            'cache_dim': 16,
            'head_dim': 64,}
        self._config_encoder = {
            'batch_dim': 8,
            'sample_dim': 16,
            'token_dim': 128}
        self._config_model = {
            'num_layers': 2,
            'num_heads': 4,
            'embed_dim': 256,
            'head_dim': 64,
            'hidden_dim': 1024,
            'output_dim': 128,
            'epsilon': 1e-6,}
        # init transformer
        self._model = llaminate.model.CacheTransformer(**self._config_model)
        # init cache
        self._cache = llaminate.utils.create_cache(**self._config_cache)
        __x = tf.zeros((self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_encoder['token_dim']))
        self._model(__x)

    def test_shapes(self):
        # inputs
        __x = tf.ones((self._config_cache['batch_dim'], self._config_encoder['sample_dim'], self._config_encoder['token_dim']))
        # call
        __y = self._model.call(inputs=__x, training=False)
        # checks
        self.assertEqual(__y.shape, __x.shape)
        # infer
        __y, self._cache = self._model.infer(inputs=__x, cache=self._cache, attention_mask=None, position=4, training=False)
        # checks
        self.assertEqual(__y.shape, __x.shape)
        self.assertEqual(len(self._cache), self._config_model['num_layers'])
        self.assertEqual(self._cache[0].shape, (2, self._config_cache['batch_dim'], self._config_cache['cache_dim'], self._config_cache['num_heads'], self._config_cache['head_dim']))

    def test_internals(self):
        # tail
        assert list(self._model._tail.kernel.shape) == [self._config_encoder['token_dim'], self._config_model['embed_dim']]
        assert list(self._model._tail.bias.shape) == [self._config_model['embed_dim']]
        # blocks
        assert len(self._model._blocks) == self._config_model['num_layers']
        # self attention
        assert all(bool(__b._attention._input_norm) for __b in self._model._blocks)
        assert all(bool(__b._attention._context_norm) for __b in self._model._blocks)
        assert all(bool(__b._attention._position) for __b in self._model._blocks)
        assert all(list(__b._attention._attention._key_dense.kernel.shape) == [self._config_model['embed_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        assert all(list(__b._attention._attention._query_dense.kernel.shape) == [self._config_model['embed_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        assert all(list(__b._attention._attention._value_dense.kernel.shape) == [self._config_model['embed_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        # ffn
        assert all(bool(__b._ffn._norm) for __b in self._model._blocks)
        assert all(list(__b._ffn._ffn._gelu.kernel.shape) == [self._config_model['embed_dim'], self._config_model['hidden_dim']] for __b in self._model._blocks)
        assert all(list(__b._ffn._ffn._linear.kernel.shape) == [self._config_model['embed_dim'], self._config_model['hidden_dim']] for __b in self._model._blocks)
        assert all(list(__b._ffn._ffn._output.kernel.shape) == [self._config_model['hidden_dim'], self._config_model['embed_dim']] for __b in self._model._blocks)
        # head
        assert list(self._model._head.kernel.shape) == [self._config_model['embed_dim'], self._config_model['output_dim']]
        # assert list(self._model._head.bias.shape) == [self._config_model['output_dim']]

    def test_null_values(self):
        # tail
        __x = tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_encoder['token_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._tail(__x), 0.5 * tf.ones([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['embed_dim']], dtype=tf.float32))
        # self attention
        __x = tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['embed_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._blocks[0]._attention(__x)[0], tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['embed_dim']], dtype=tf.float32))
        # ffn
        __x = tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['embed_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._blocks[0]._ffn(__x), tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['embed_dim']], dtype=tf.float32))
        # head
        __x = tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['embed_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._head(__x), 0.5 * tf.ones([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['output_dim']], dtype=tf.float32))

# WITHOUT CACHE ###############################################################

class TransformerTest(tf.test.TestCase):
    def setUp(self):
        self._config_encoder = {
            'batch_dim': 2,
            'sample_dim': 16,
            'token_dim': 64}
        self._config_model = {
            'num_layers': 2,
            'num_heads': 4,
            'token_dim': 64,
            'input_dim': 256,
            'embed_dim': 256,
            'head_dim': 64,
            'hidden_dim': 1024,
            'output_dim': 128,
            'epsilon': 1e-6,}
        # init transformer
        self._model = llaminate.model.Transformer(**self._config_model)
        # build
        __x = tf.zeros((self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_encoder['token_dim']))
        self._model(__x)

    def test_internals(self):
        # embeddings
        assert list(self._model._embed._embeddings.shape) == [self._config_model['input_dim'], self._config_model['embed_dim'] // self._config_encoder['token_dim']]
        # blocks
        assert len(self._model._blocks) == self._config_model['num_layers']
        # self attention
        assert all(bool(__b._attention._input_norm) for __b in self._model._blocks)
        assert all(bool(__b._attention._context_norm) for __b in self._model._blocks)
        assert all(bool(__b._attention._position) for __b in self._model._blocks)
        assert all(list(__b._attention._attention._key_dense.kernel.shape) == [self._config_model['embed_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        assert all(list(__b._attention._attention._query_dense.kernel.shape) == [self._config_model['embed_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        assert all(list(__b._attention._attention._value_dense.kernel.shape) == [self._config_model['embed_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        # ffn
        assert all(bool(__b._ffn._norm) for __b in self._model._blocks)
        assert all(list(__b._ffn._ffn._gelu.kernel.shape) == [self._config_model['embed_dim'], self._config_model['hidden_dim']] for __b in self._model._blocks)
        assert all(list(__b._ffn._ffn._linear.kernel.shape) == [self._config_model['embed_dim'], self._config_model['hidden_dim']] for __b in self._model._blocks)
        assert all(list(__b._ffn._ffn._output.kernel.shape) == [self._config_model['hidden_dim'], self._config_model['embed_dim']] for __b in self._model._blocks)
        # head
        assert list(self._model._head.kernel.shape) == [self._config_model['embed_dim'], self._config_model['output_dim']]
        # assert list(self._model._head.bias.shape) == [self._config_model['output_dim']]

    def test_shapes(self):
        # inputs
        __x = tf.ones((self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_encoder['token_dim']), dtype=tf.int32)
        # call
        __y = self._model.call(inputs=__x, attention_mask=None, training=False)
        # checks
        self.assertEqual(tuple(__y.shape), (self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['output_dim']))

    def test_null_values(self):
        # tail
        __x = tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_encoder['token_dim']], dtype=tf.int32)
        __y = self._model._embed(__x)
        __z = tf.tile(__y[:, :1, :], mlable.utils.filter_shape(__y.shape, axes=[1])) # repeat first feature vector
        self.assertAllEqual(__y , __z)
        # self attention
        __x = tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['embed_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._blocks[0]._attention(__x), tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['embed_dim']], dtype=tf.float32))
        # ffn
        __x = tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['embed_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._blocks[0]._ffn(__x), tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['embed_dim']], dtype=tf.float32))
        # head
        __x = tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['embed_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._head(__x), 0.5 * tf.ones([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['output_dim']], dtype=tf.float32))
