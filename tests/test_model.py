import math

import tensorflow as tf

import mlable.shaping

import llaminate.model
import llaminate.utils

# WITHOUT CACHE ###############################################################

class TransformerTest(tf.test.TestCase):
    def setUp(self):
        self._config_encoder = {
            'batch_dim': 2,
            'sample_dim': 16,
            'input_dim': 64}
        self._config_model = {
            'num_layers': 2,
            'num_heads': 4,
            'input_dim': 64,
            'embed_dim': 256,
            'head_dim': 64,
            'hidden_dim': 1024,
            'epsilon': 1e-6,}
        # init transformer
        self._model = llaminate.model.Transformer(**self._config_model)
        # build
        __x = tf.zeros((self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_encoder['input_dim']))
        self._model(__x)

    def test_internals(self):
        # embeddings
        assert list(self._model._embed._embeddings.shape) == [256, self._config_model['embed_dim'] // self._config_encoder['input_dim']]
        # blocks
        assert len(self._model._blocks) == self._config_model['num_layers']
        # self attention
        assert all(bool(__b._attention._input_norm) for __b in self._model._blocks)
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
        assert list(self._model._head.kernel.shape) == [self._config_model['embed_dim'], 8 * self._config_model['input_dim']]
        # assert list(self._model._head.bias.shape) == [8 * self._config_model['input_dim']]

    def test_shapes(self):
        # inputs
        __x = tf.ones((self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_encoder['input_dim']), dtype=tf.int32)
        # call
        __y = self._model.call(inputs=__x, logits=True, training=False)
        # checks
        self.assertEqual(tuple(__y.shape), (self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], 8 * self._config_model['input_dim']))

    def test_null_values(self):
        # embed
        __x = tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_encoder['input_dim']], dtype=tf.int32)
        __y = self._model._embed(__x)
        __z = tf.tile(__y[:, :1, :], mlable.shaping.filter_shape(__y.shape, axes=[1])) # repeat first feature vector
        self.assertAllEqual(__y , __z)
        # self attention
        __x = tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['embed_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._blocks[0]._attention(__x), tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['embed_dim']], dtype=tf.float32))
        # ffn
        __x = tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['embed_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._blocks[0]._ffn(__x), tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['embed_dim']], dtype=tf.float32))
        # head
        __x = tf.zeros([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], self._config_model['embed_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._head(__x), 0.5 * tf.ones([self._config_encoder['batch_dim'], self._config_encoder['sample_dim'], 8 * self._config_model['input_dim']], dtype=tf.float32))
