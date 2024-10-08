import tensorflow as tf

import llaminate.layers
import mlable.utils

# WITH CACHE ##################################################################

class CacheDecoderBlockTest(tf.test.TestCase):
    def setUp(self):
        super(CacheDecoderBlockTest, self).setUp()
        self._test_cases = [
            {
                'init': {
                    'num_heads': 2,
                    'embed_dim': 4,
                    'head_dim': 2,
                    'hidden_dim': 16},
                'input': {
                    'inputs': tf.random.uniform(shape=(2, 3, 4), dtype=tf.dtypes.float32),
                    'cache': mlable.utils.create_cache(batch_dim=2, cache_dim=3, head_dim=2, num_heads=2),
                    'position': 2,
                    'training': False,},
                'output': {
                    'shape': (2, 3, 4),},
                'cache': {
                    'shape': (2, 2, 3, 2, 2),},},
            {
                'init': {
                    'num_heads': 2,
                    'embed_dim': 4,
                    'head_dim': 2,
                    'hidden_dim': 16},
                'input': {
                    'inputs': tf.random.uniform(shape=(2, 8, 4), dtype=tf.dtypes.float32),
                    'cache': mlable.utils.create_cache(batch_dim=2, cache_dim=16, head_dim=2, num_heads=2),
                    'position': 3,
                    'training': True,}, # ignore cache
                'output': {
                    'shape': (2, 8, 4),},
                'cache': {
                    'values': None,},},] # cache ignored during training

    def test_block_shape(self):
        for __case in self._test_cases:
            __layer = llaminate.layers.CacheDecoderBlock(**__case['init'])
            __outputs, __cache = __layer(**__case['input'])
            # check the output
            if 'shape' in __case['output']:
                self.assertEqual(tuple(__outputs.shape), __case['output']['shape'])
            if 'values' in __case['output']:
                self.assertAllClose(__outputs, __case['output']['values'])
            # check the cache
            if 'shape' in __case['cache']:
                self.assertEqual(tuple(__cache.shape), __case['cache']['shape'])
            if 'values' in __case['cache']:
                self.assertEqual(__cache, __case['cache']['values'])

# WITHOUT CACHE ###############################################################

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
                    'training': False,},
                'output': {
                    'shape': (2, 1, 4),},},
            {
                'init': {
                    'num_heads': 2,
                    'embed_dim': 4,
                    'head_dim': 2,
                    'hidden_dim': 16},
                'input': {
                    'inputs': tf.random.uniform(shape=(2, 1, 4), dtype=tf.dtypes.float32),
                    'training': True,},
                'output': {
                    'shape': (2, 1, 4),},},]

    def test_block_shape(self):
        for __case in self._test_cases:
            __layer = llaminate.layers.DecoderBlock(**__case['init'])
            __outputs = __layer(**__case['input'])
            # check the output
            if 'shape' in __case['output']:
                self.assertEqual(tuple(__outputs.shape), __case['output']['shape'])
            if 'values' in __case['output']:
                self.assertAllClose(__outputs, __case['output']['values'])
