import numpy as np
import tensorflow as tf

import llaminate.layers
import mlable.utils

# FF ##########################################################################

class FeedForwardBlockTest(tf.test.TestCase):
    def setUp(self):
        super(FeedForwardBlockTest, self).setUp()
        self._test_cases = [
            {
                'input_dim': 2,
                'hidden_dim': 3,
                'batch_size': 2,
                'expected_val': [11.726998, 47.998482],
                'expected_shape': (2, 1, 2),},]

    def test_ffn(self):
        for __case in self._test_cases:
            # inputs
            __inputs = tf.reshape(tf.range(1, __case['batch_size'] + 1, dtype=tf.float32), (__case['batch_size'], 1, 1))
            __inputs = tf.repeat(__inputs, __case['input_dim'], axis=-1)
            # init
            __layer = llaminate.layers.FeedForwardBlock(
                input_dim=__case["input_dim"],
                hidden_dim=__case["hidden_dim"])
            # build
            _ = __layer(__inputs)
            # set weights
            __layer._gelu.set_weights([np.ones((__case['input_dim'], __case['hidden_dim']))])
            __layer._linear.set_weights([np.ones((__case['input_dim'], __case['hidden_dim']))])
            __layer._output.set_weights([np.ones((__case['hidden_dim'], __case['input_dim']))])
            # compute
            __output = __layer(__inputs)
            # test
            np.testing.assert_array_almost_equal(__output[:, 0, 0].numpy().tolist(), __case["expected_val"])
            self.assertEqual(tuple(__output.shape), __case['expected_shape'])

# DECODER BLOCK ###############################################################

class DecoderBlockTest(tf.test.TestCase):
    def setUp(self):
        super(DecoderBlockTest, self).setUp()
        self._test_cases = [
            {
                'init': {
                    'num_heads': 2,
                    'embed_dim': 4,
                    'head_dim': 2,
                    'hidden_dim': 16},
                'input': {
                    'inputs': tf.random.uniform(shape=(2, 1, 4), dtype=tf.dtypes.float32),
                    'cache': mlable.utils.create_cache(batch_dim=2, cache_dim=3, head_dim=2, num_heads=2),
                    'position': 3},
                'output': {
                    'shape': (2, 1, 4),},
                'cache': {
                    'shape': (2, 2, 3, 2, 2),},}]

    def test_block_shape(self):
        for __case in self._test_cases:
            __layer = llaminate.layers.DecoderBlock(**__case['init'])
            __outputs, __cache = __layer(**__case['input'])
            # check the output
            if 'shape' in __case['output']:
                self.assertEqual(tuple(__outputs.shape), __case['output']['shape'])
            if 'values' in __case['output']:
                np.testing.assert_array_almost_equal(__outputs, __case['output']['values'])
            # check the cache
            if 'shape' in __case['cache']:
                self.assertEqual(tuple(__cache.shape), __case['cache']['shape'])
