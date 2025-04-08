import math

import tensorflow as tf
import tensorflow_datasets as tfds

import mlable.sampling
import llaminate.pipeline

# BINARY BYTES ################################################################

class PreprocessBinaryBytesTest(tf.test.TestCase):
    def setUp(self):
        super(PreprocessBinaryBytesTest, self).setUp()
        self._config = {'batch_dim': 8, 'sample_dim': 256, 'token_dim': 16, 'data_weight': 1.0, 'padding_weight': 0.0, 'features': ['question'], 'separator': '\x1d',}
        self._preprocess = llaminate.pipeline.preprocess_factory(**self._config)
        self._dataset_before = tfds.load('mlqa/en', split='test', as_supervised=False, shuffle_files=True, batch_size=None)
        self._dataset_after = self._dataset_before.batch(self._config['batch_dim'], drop_remainder=True).map(self._preprocess)

    def test_specs(self):
        __inputs_spec, __targets_spec, __weights_spec = self._dataset_after.element_spec
        self.assertEqual(tuple(__inputs_spec.shape), (self._config['batch_dim'], self._config['sample_dim']))
        self.assertEqual(tuple(__targets_spec.shape), (self._config['batch_dim'], self._config['sample_dim'], 8))
        self.assertEqual(tuple(__weights_spec.shape), (self._config['batch_dim'], self._config['sample_dim']))
        self.assertEqual(__inputs_spec.dtype, tf.dtypes.int32)
        self.assertEqual(__targets_spec.dtype, tf.dtypes.float32)
        self.assertEqual(__weights_spec.dtype, tf.dtypes.float32)

    def test_values(self):
        __batch = iter(self._dataset_after)
        for _ in range(16):
            __x, __y, __m = next(__batch)
            # interpret the binary encodings
            __y = mlable.sampling.binary(__y, depth=-1, temp=0.0, topp=0.0, topk=0, seed=None)
            # the inputs are shifted by one token
            self.assertAllEqual(__x[:, :self._config['token_dim']], tf.zeros(shape=(self._config['batch_dim'], self._config['token_dim']), dtype=tf.dtypes.int32))
            # x and y are offset by token_dim
            self.assertAllEqual(__x[:, self._config['token_dim']:], __y[:, :-self._config['token_dim']])

    def test_weights(self):
        __batch = iter(self._dataset_after)
        for _ in range(16):
            __x, __y, __m = next(__batch)
            __m = tf.cast(__m, tf.int32)
            assert 0 < tf.reduce_sum(__m).numpy()
            assert tf.reduce_sum(__m).numpy() < tf.size(__m).numpy()
            self.assertAllEqual(__m[:, :self._config['token_dim']], tf.zeros(shape=(self._config['batch_dim'], self._config['token_dim']), dtype=tf.dtypes.float32))
            self.assertAllEqual(__x * __m, __x)
