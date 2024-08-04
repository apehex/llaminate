"""Train llaminate from scratch or from a checkpoint"""

import datetime
import functools
import itertools
import math
import os
import random
import urllib.request

import datasets as hd
import tensorflow as tf
import tensorflow_datasets as tfds

import mlable.data
import mlable.metrics

import tokun.data
import tokun.evaluation
import tokun.meta
import tokun.model
import tokun.pipeline

import llaminate.model
import llaminate.pipeline
import llaminate.utils

# MIXED PRECISION #############################################################

tf.keras.mixed_precision.set_global_policy('mixed_float16') # mixed_bfloat16 on TPUs

# DEVICES #####################################################################

tf.debugging.set_log_device_placement(False)

CPU = tf.config.list_logical_devices('CPU')
GPU = tf.config.list_logical_devices('GPU')
TPU = tf.config.list_logical_devices('TPU')

if TPU:
    RESOLVER = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(RESOLVER)
    tf.tpu.experimental.initialize_tpu_system(RESOLVER)
    DISTRIBUTION_STRATEGY = tf.distribute.TPUStrategy(RESOLVER)
elif GPU:
    DISTRIBUTION_STRATEGY = tf.distribute.MirroredStrategy(GPU)
else:
    DISTRIBUTION_STRATEGY = tf.distribute.MirroredStrategy(CPU)

print(DISTRIBUTION_STRATEGY)

# MODE ########################################################################

IMPORT = False
FREEZE = False # freeze tokun weights
TRAINING = True
BINARY = True
DEBUG = False

# MODEL PARAMETERS ############################################################

N_SEQUENCE_AXIS = 1
N_FEATURE_AXIS = -1

N_LAYERS_NUM = 32
N_HEADS_NUM = 4

N_INPUT_DIM = 256
N_OUTPUT_DIM = 8 if BINARY else 256

N_CACHE_DIM = 4 * 256 # 2048 in llama3-8B but tokun embeddings = 16 chr = 4 llama3 tokens
N_EMBED_DIM = 256
N_HIDDEN_DIM = 4 * N_EMBED_DIM
N_HEAD_DIM = N_EMBED_DIM // N_HEADS_NUM

# TOKENIZER PARAMETERS ########################################################

TOKUN_DIM = [4, 4]
TOKUN_FACTOR = math.prod(TOKUN_DIM) // 4
TOKUN_VERSION = tokun.meta.version(token_units=TOKUN_DIM, sequence_axis=1, input_dim=N_INPUT_DIM, output_dim=N_OUTPUT_DIM)

TOKUN_LABEL = '9.2'
TOKUN_PATH = 'tokun.keras'
TOKUN_URL = 'https://github.com/apehex/tokun/raw/main/models/{}/{}/{}/{}.keras'.format(*TOKUN_VERSION, TOKUN_LABEL)

# TRAINING PARAMETERS #########################################################

N_BATCH_DIM = 128
N_SAMPLE_DIM = N_CACHE_DIM * TOKUN_FACTOR

N_EPOCHS = 2

R_0, B_1, B_2 = (0.1 if IMPORT else 1.) * 0.001, 0.9, 0.99

CLASS_WEIGHTS = {__c: 0.3 if __c == 0 else 1. for __c in range(256)} # there are 3 times more 0s than other bytes

# DERIVED PARAMETERS ##########################################################

DATETIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

LLAMINATE_VERSION = [str(N_LAYERS_NUM), str(N_HIDDEN_DIM)]
LLAMINATE_LOGS_PATH = os.path.join('.logs/', *LLAMINATE_VERSION, DATETIME)
LLAMINATE_MODEL_PATH = 'llaminate.keras'

# DATASET META ################################################################

# TODO bigcode/the-stack
# TODO ArmelR/stack-exchange-instruction

DATASETS_META = {
    'pt-wikipedia': {
        'path': 'wikimedia/wikipedia',
        'name': '20231101.en',
        'splits': [f'train[{__p}%:{__p + 10}%]' for __p in range(0, 100, 10)],
        'features': ['text'],},
    'ft-retro-ascii-art': {
        'path': 'jdpressman/retro-ascii-art-v1',
        'name': None,
        'train': 'train',
        'splits': [f'train[{__p}%:{__p + 10}%]+validation[{__p}%:{__p + 10}%]' for __p in range(0, 100, 10)],
        'features': ['prompt', 'art_aic'],},
    'ft-stack-exchange': {
        'path': 'Alignment-Lab-AI/Stack-Exchange-April',
        'name': None,
        'splits': [f'train[{__p}%:{__p + 10}%]' for __p in range(0, 100, 10)],
        'features': ['question', 'answer'],},
    'ft-math': {
        'path': 'hendrycks/competition_math',
        'name': None,
        'splits': [f'train[{__p}%:{__p + 10}%]+test[{__p}%:{__p + 10}%]' for __p in range(0, 100, 10)],
        'features': ['problem', 'solution'],},}

# DOWNLOAD  DATASETS ##########################################################

DATASETS = {
    __name: [
        hd.load_dataset(path=__args['path'], name=__args['name'], split=__s).to_tf_dataset(shuffle=True, batch_size=None)
        for __s in __args['splits']]
    for __name, __args in DATASETS_META.items()}

# DATASET STATS ###############################################################

STATS = {__n: mlable.data.stats(dataset=DATASETS[__n][0], features=DATASETS_META[__n]['features'], count=2048) for __n in DATASETS}

# PREPROCESS ##################################################################

for __name in DATASETS:
    # specialized preprocessing fn
    __preprocess = functools.partial(
        llaminate.pipeline.preprocess,
        token_dim=math.prod(TOKUN_DIM),
        output_dim=N_OUTPUT_DIM,
        batch_dim=N_BATCH_DIM,
        sample_dim=N_SAMPLE_DIM,
        features=DATASETS_META[__name]['features'],
        padding_weight=0.01,
        sample_weights=True,
        binary=BINARY)
    # apply
    for __idx in range(len(DATASETS[__name])):
        DATASETS[__name][__idx] = DATASETS[__name][__idx].batch(
            N_BATCH_DIM,
            drop_remainder=True,
            num_parallel_calls=tf.data.AUTOTUNE
        ).map(__preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# CONCATENATE #################################################################

DATASET_TRAIN = functools.reduce(lambda __l, __r: __l.concatenate(__r), [DATASETS[__n][__i] for __n in (set(DATASETS.keys()) - {'ft-retro-ascii-art'}) for __i in range(len(DATASETS[__n]) - 1)]) # - {'pt-wikipedia'}
DATASET_TEST = functools.reduce(lambda __l, __r: __l.concatenate(__r), [DATASETS[__n][-1] for __n in (set(DATASETS.keys()) - {'ft-retro-ascii-art'})]) # - {'pt-wikipedia'}

# CHECK DATASET ###############################################################

print(DATASET_TRAIN.element_spec)
print(DATASET_TEST.element_spec)

print('train: {:,} samples'.format(DATASET_TRAIN.cardinality().numpy()))
print('test:  {:,} samples'.format(DATASET_TEST.cardinality().numpy()))

# IMPORT TOKENIZER ############################################################

urllib.request.urlretrieve(TOKUN_URL, TOKUN_PATH)

# METRICS #####################################################################

_Accuracy = mlable.metrics.BinaryGroupAccuracy if BINARY else mlable.metrics.CategoricalGroupAccuracy
_Loss = tf.keras.losses.BinaryCrossentropy if BINARY else tf.keras.losses.CategoricalCrossentropy

# INIT MODEL ##################################################################

with DISTRIBUTION_STRATEGY.scope():
    # TOKENIZER ###############################################################
    TOKUN = tf.keras.models.load_model(TOKUN_PATH, compile=False)
    TOKUN.trainable = not FREEZE # freeze the weights

    # METRICS #################################################################
    byte_accuracy = _Accuracy(group=1, name='byte_accuracy')
    character_accuracy = _Accuracy(group=4, name='character_accuracy')
    token_accuracy = _Accuracy(group=math.prod(TOKUN_DIM), name='token_accuracy')

    # WEIGHTS #################################################################
    if IMPORT and os.path.isfile(LLAMINATE_MODEL_PATH):
        LLAMINATE = tf.keras.models.load_model(LLAMINATE_MODEL_PATH, compile=False)
    else:
        LLAMINATE = llaminate.model.Transformer(num_layers=N_LAYERS_NUM, num_heads=N_HEADS_NUM, cache_dim=N_CACHE_DIM, embed_dim=N_EMBED_DIM, head_dim=N_HEAD_DIM, hidden_dim=N_HIDDEN_DIM)
        LLAMINATE.set_tokenizer(encoder=TOKUN._encoder, decoder=TOKUN._decoder)

    # INPUT ###################################################################
    # __input = tf.keras.Input(shape=(4 * TOKUN_FACTOR * N_CACHE_DIM,), batch_size=N_BATCH_DIM)
    # LLAMINATE = tf.keras.models.Model(__input, LLAMINATE(__input))

    # COMPILE #################################################################
    LLAMINATE.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=R_0, beta_1=B_1, beta_2=B_2),
        loss=_Loss(from_logits=False, label_smoothing=0., axis=-1, reduction='sum_over_batch_size', name='cce_loss'),
        weighted_metrics=[byte_accuracy, character_accuracy, token_accuracy])

# TRAIN #######################################################################

if TRAINING:
    with DISTRIBUTION_STRATEGY.scope():
        # callbacks
        cp_callback = tf.keras.callbacks.ModelCheckpoint(LLAMINATE_MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=LLAMINATE_LOGS_PATH, histogram_freq=1, embeddings_freq=0, profile_batch=(16, 32), write_graph=False, write_images=True)
        # datasets
        ds_splits = DATASETS['pt-wikipedia']
        ds_test = ds_splits[-1].take(128).prefetch(tf.data.AUTOTUNE)
        # loop on each split
        for __split in ds_splits[:-1]:
            # model fitting
            TRAINING_HISTORY = LLAMINATE.fit(
                x=__split,
                batch_size=None,
                epochs=N_EPOCHS,
                validation_split=None,
                validation_data=ds_test,
                validation_freq=list(range(1, N_EPOCHS + 1, 1)),
                class_weight=CLASS_WEIGHTS,
                verbose=1,
                callbacks=[cp_callback, tb_callback])
