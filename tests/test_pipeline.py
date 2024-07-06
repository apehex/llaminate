import functools

import tensorflow as tf
import tensorflow_datasets as tfds

import llaminate.pipeline

# ROPE ########################################################################

class PreprocessTest(tf.test.TestCase):
    def setUp(self):
        super(PreprocessTest, self).setUp()
        self._config = {'batch_dim': 8, 'sample_dim': 128, 'token_dim': 16, 'embed_dim': 256, 'mask': True, 'features': ['question']}
        self._preprocess = functools.partial(llaminate.pipeline.preprocess, **self._config)
        self._dataset_before = tfds.load('mlqa/en', split='test', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=None)
        self._dataset_after = self._dataset_before.batch(self._config['batch_dim']).map(self._preprocess)

    def test_specs(self):
        __inputs_spec, __targets_spec, __masks_spec = self._dataset_after.element_spec
        self.assertEqual(__inputs_spec.shape, (self._config['batch_dim'], 4 * self._config['sample_dim']))
        self.assertEqual(__targets_spec.shape, (self._config['batch_dim'], 4 * self._config['sample_dim'], self._config['embed_dim']))
        self.assertEqual(__masks_spec.shape, (self._config['batch_dim'], 4 * self._config['sample_dim']))
        self.assertEqual(__inputs_spec.dtype, tf.dtypes.uint8)
        self.assertEqual(__targets_spec.dtype, tf.dtypes.float32)
        self.assertEqual(__masks_spec.dtype, tf.dtypes.bool)

    def test_values(self):
        __batch = iter(self._dataset_after)
        for _ in range(16):
            __x, __y, __m = next(__batch)
            __x = tf.cast(__x, dtype=tf.dtypes.int32)
            __y = tf.argmax(__y, axis=-1, output_type=tf.dtypes.int32)
            self.assertAllEqual(__x[:, :self._config['token_dim']], tf.zeros(shape=(self._config['batch_dim'], self._config['token_dim']), dtype=tf.dtypes.int32))
            self.assertAllEqual(__x[:, self._config['token_dim']:], __y[:, :-self._config['token_dim']]) # x and y are offset by token_dim

    def test_masks(self):
        __batch = iter(self._dataset_after)
        for _ in range(16):
            __x, __y, __m = next(__batch)
            __x = tf.cast(__x, dtype=tf.dtypes.int32)
            __y = tf.argmax(__y, axis=-1, output_type=tf.dtypes.int32)
            __m = tf.cast(__m, dtype=tf.dtypes.int32)
            assert 0 < tf.reduce_sum(__m).numpy()
            assert tf.reduce_sum(__m).numpy() < tf.size(__m).numpy()
            self.assertAllEqual(__x * __m, __x)
