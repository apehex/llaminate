import functools

import tensorflow as tf
import tensorflow_datasets as tfds

import tokun.pipeline
import llaminate.pipeline

# ROPE ########################################################################

class PreprocessTest(tf.test.TestCase):
    def setUp(self):
        super(PreprocessTest, self).setUp()
        self._config_categorical = {'batch_dim': 8, 'sample_dim': 128, 'token_dim': 16, 'output_dim': 256, 'features': ['question'], 'sample_weights': True, 'binary': False}
        self._config_binary = {'batch_dim': 8, 'sample_dim': 128, 'token_dim': 16, 'output_dim': 8, 'features': ['question'], 'sample_weights': True, 'binary': True}
        self._preprocess_categorical = functools.partial(llaminate.pipeline.preprocess, **self._config_categorical)
        self._preprocess_binary = functools.partial(llaminate.pipeline.preprocess, **self._config_binary)
        self._dataset_before = tfds.load('mlqa/en', split='test', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=None)
        self._dataset_categorical = self._dataset_before.batch(self._config_categorical['batch_dim']).map(self._preprocess_categorical)
        self._dataset_binary = self._dataset_before.batch(self._config_categorical['batch_dim']).map(self._preprocess_binary)

    def test_specs(self):
        __inputs_spec, __targets_spec, __weights_spec = self._dataset_categorical.element_spec
        self.assertEqual(__inputs_spec.shape, (self._config_categorical['batch_dim'], 4 * self._config_categorical['sample_dim']))
        self.assertEqual(__targets_spec.shape, (self._config_categorical['batch_dim'], 4 * self._config_categorical['sample_dim'], self._config_categorical['output_dim']))
        self.assertEqual(__weights_spec.shape, (self._config_categorical['batch_dim'], 4 * self._config_categorical['sample_dim']))
        self.assertEqual(__inputs_spec.dtype, tf.dtypes.int32)
        self.assertEqual(__targets_spec.dtype, tf.dtypes.float32)
        self.assertEqual(__weights_spec.dtype, tf.dtypes.float32)
        __inputs_spec, __targets_spec, __weights_spec = self._dataset_binary.element_spec
        self.assertEqual(__inputs_spec.shape, (self._config_binary['batch_dim'], 4 * self._config_binary['sample_dim']))
        self.assertEqual(__targets_spec.shape, (self._config_binary['batch_dim'], 4 * self._config_binary['sample_dim'], self._config_binary['output_dim']))
        self.assertEqual(__weights_spec.shape, (self._config_binary['batch_dim'], 4 * self._config_binary['sample_dim']))
        self.assertEqual(__inputs_spec.dtype, tf.dtypes.int32)
        self.assertEqual(__targets_spec.dtype, tf.dtypes.float32)
        self.assertEqual(__weights_spec.dtype, tf.dtypes.float32)

    def test_values(self):
        # categorical
        __batch = iter(self._dataset_categorical)
        for _ in range(16):
            __x, __y, __m = next(__batch)
            __x = tf.cast(__x, dtype=tf.dtypes.int32)
            __y = tokun.pipeline.interpret(__y, binary=False)
            self.assertAllEqual(__x[:, :self._config_categorical['token_dim']], tf.zeros(shape=(self._config_categorical['batch_dim'], self._config_categorical['token_dim']), dtype=tf.dtypes.int32))
            self.assertAllEqual(__x[:, self._config_categorical['token_dim']:], __y[:, :-self._config_categorical['token_dim']]) # x and y are offset by token_dim
        # binary
        __batch = iter(self._dataset_binary)
        for _ in range(16):
            __x, __y, __m = next(__batch)
            __x = tf.cast(__x, dtype=tf.dtypes.int32)
            __y = tokun.pipeline.interpret(__y, binary=True)
            self.assertAllEqual(__x[:, :self._config_categorical['token_dim']], tf.zeros(shape=(self._config_categorical['batch_dim'], self._config_categorical['token_dim']), dtype=tf.dtypes.int32))
            self.assertAllEqual(__x[:, self._config_categorical['token_dim']:], __y[:, :-self._config_categorical['token_dim']]) # x and y are offset by token_dim

    def test_weights(self):
        # categorical
        __batch = iter(self._dataset_categorical)
        for _ in range(16):
            __x, __y, __m = next(__batch)
            __x = tf.cast(__x, dtype=tf.dtypes.int32)
            __y = tf.argmax(__y, axis=-1, output_type=tf.dtypes.int32)
            __m = tf.cast(__m, dtype=tf.dtypes.int32)
            assert 0 < tf.reduce_sum(__m).numpy()
            assert tf.reduce_sum(__m).numpy() < tf.size(__m).numpy()
            self.assertAllEqual(__x * __m, __x)
        # binary
        __batch = iter(self._dataset_binary)
        for _ in range(16):
            __x, __y, __m = next(__batch)
            __x = tf.cast(__x, dtype=tf.dtypes.int32)
            __y = tf.argmax(__y, axis=-1, output_type=tf.dtypes.int32)
            __m = tf.cast(__m, dtype=tf.dtypes.int32)
            assert 0 < tf.reduce_sum(__m).numpy()
            assert tf.reduce_sum(__m).numpy() < tf.size(__m).numpy()
            self.assertAllEqual(__x * __m, __x)
