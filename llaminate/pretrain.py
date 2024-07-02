"""Train llaminate from scratch or from a checkpoint"""

import datetime
import functools
import itertools
import math
import os
import random
import urllib.request

import datasets as ds
import tensorflow as tf
import tensorflow_datasets as tfds

import mlable.data
import mlable.io
import mlable.metrics
import mlable.optimizers

import tokun.data
import tokun.evaluation
import tokun.meta
import tokun.model
import tokun.pipeline

import llaminate.model
import llaminate.pipeline
import llaminate.utils

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

DEBUG = False
IMPORT = False
FREEZE = True # freeze tokun weights
TRAINING = True

# MODEL PARAMETERS ############################################################

N_SEQUENCE_AXIS = 1
N_FEATURE_AXIS = -1

N_LAYERS_NUM = 16
N_HEADS_NUM = 4

N_CACHE_DIM = 256 # 2048 in llama3-8B but tokun embeddings = 16 chr = 4 llama3 tokens
N_EMBED_DIM = 256
N_HIDDEN_DIM = 4 * N_EMBED_DIM
N_HEAD_DIM = N_EMBED_DIM // N_HEADS_NUM

LLAMINATE_PATH = 'llaminate.keras'

# TOKENIZER PARAMETERS ########################################################

TOKUN_DIM = [16, 4]
TOKUN_FACTOR = math.prod(TOKUN_DIM) // 4
TOKUN_VERSION = tokun.meta.version(units=TOKUN_DIM, axis=1)

TOKUN_LABEL = '7.7'
TOKUN_PATH = 'tokun.keras'
TOKUN_URL = 'https://github.com/apehex/tokun/raw/main/models/{}/{}/{}.keras'.format(*TOKUN_VERSION, TOKUN_LABEL)

# TRAINING PARAMETERS #########################################################

N_BATCH_DIM = 128
N_SAMPLE_DIM = N_CACHE_DIM * TOKUN_FACTOR

N_EPOCHS = 8

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
        'train': 'train[:90%]',
        'test': 'train[-10%:]',
        'features': ['text'],},
    'ft-retro-ascii-art': {
        'path': 'jdpressman/retro-ascii-art-v1',
        'name': None,
        'train': 'train',
        'test': 'validation',
        'features': ['prompt', 'art_aic'],},
    'ft-stack-exchange': {
        'path': 'Alignment-Lab-AI/Stack-Exchange-April',
        'name': None,
        'train': 'train[:90%]',
        'test': 'train[-10%:]',
        'features': ['question', 'answer'],},
    'ft-math': {
        'path': 'hendrycks/competition_math',
        'name': None,
        'train': 'train',
        'test': 'test',
        'features': ['problem', 'solution'],},}

# DOWNLOAD  DATASETS ##########################################################

DATASETS = {
    __name: {
        'train': ds.load_dataset(path=__args['path'], name=__args['name'], split=__args['train']).to_tf_dataset(shuffle=True, batch_size=None),
        'test': ds.load_dataset(path=__args['path'], name=__args['name'], split=__args['test']).to_tf_dataset(shuffle=True, batch_size=None),}
    for __name, __args in DATASETS_META.items()}

# DATASET STATS ###############################################################

STATS = {__n: {'min': 0, 'max': 0, 'mean': 0} for __n in DATASETS}

for __name in DATASETS:
    # sample each dataset
    __m = DATASETS_META[__name]
    __b = iter(DATASETS[__name]['train'])
    __s = [next(__b) for _ in range(128)]
    __l = [len(tf.strings.join(inputs=[__e[__f] for __f in __m['features']], separator='\x1d').numpy()) for __e in __s]
    # save the stats
    STATS[__name]['min'] = min(__l)
    STATS[__name]['max'] = max(__l)
    STATS[__name]['mean'] = tf.reduce_mean(__l).numpy()

# PREPROCESS ##################################################################

for __name in DATASETS:
    DATASETS[__name]['train'] = llaminate.pipeline.preprocess(dataset=DATASETS[__name]['train'], token_dim=math.prod(TOKUN_DIM), embed_dim=N_EMBED_DIM, sample_dim=N_SAMPLE_DIM, features=DATASETS_META[__name]['features'])
    DATASETS[__name]['test'] = llaminate.pipeline.preprocess(dataset=DATASETS[__name]['test'], token_dim=math.prod(TOKUN_DIM), embed_dim=N_EMBED_DIM, sample_dim=N_SAMPLE_DIM, features=DATASETS_META[__name]['features'])

# CONCATENATE #################################################################

DATASET_TRAIN = functools.reduce(lambda __l, __r: __l.concatenate(__r), [DATASETS[__n]['train'] for __n in (set(DATASETS.keys()) - {'ft-retro-ascii-art'})]) # - {'pt-wikipedia'}
DATASET_TEST = functools.reduce(lambda __l, __r: __l.concatenate(__r), [DATASETS[__n]['test'] for __n in (set(DATASETS.keys()) - {'ft-retro-ascii-art'})]) # - {'pt-wikipedia'}

# CHECK DATASET ###############################################################

print(DATASET_TRAIN.element_spec)
print(DATASET_TEST.element_spec)

print('train: {:,} samples'.format(DATASET_TRAIN.cardinality().numpy()))
print('test:  {:,} samples'.format(DATASET_TEST.cardinality().numpy()))

# IMPORT TOKENIZER ############################################################

urllib.request.urlretrieve(TOKUN_URL, TOKUN_PATH)

# INIT MODEL ##################################################################

with DISTRIBUTION_STRATEGY.scope():
    # TOKENIZER ###############################################################
    TOKUN = tf.keras.models.load_model(TOKUN_PATH, compile=False)
    TOKUN.trainable = not FREEZE # freeze the weights

    # METRICS #################################################################
    byte_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=1, name='byte_accuracy')
    character_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=4, name='character_accuracy')
    token_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=math.prod(TOKUN_DIM), name='token_accuracy')

    # WEIGHTS #################################################################
    if IMPORT and os.path.isfile(LLAMINATE_MODEL_PATH):
        LLAMINATE = tf.keras.models.load_model(LLAMINATE_MODEL_PATH, compile=False)
    else:
        LLAMINATE = llaminate.model.Transformer(num_layers=N_LAYERS_NUM, num_heads=N_HEADS_NUM, cache_dim=N_CACHE_DIM, embed_dim=N_EMBED_DIM, head_dim=N_HEAD_DIM, hidden_dim=N_HIDDEN_DIM)

    # INIT ####################################################################
    LLAMINATE.set_tokenizer(encoder=TOKUN._encoder, decoder=TOKUN._decoder)

    # INPUT ###################################################################
    # __input = tf.keras.Input(shape=(4 * TOKUN_FACTOR * N_CACHE_DIM,), batch_size=N_BATCH_DIM)
    # LLAMINATE = tf.keras.models.Model(__input, LLAMINATE(__input))

    # COMPILE #################################################################
    LLAMINATE.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=R_0, beta_1=B_1, beta_2=B_2),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss'),
        metrics=[byte_accuracy, character_accuracy, token_accuracy])

# TRAIN #######################################################################

if TRAINING:
    with DISTRIBUTION_STRATEGY.scope():
        # callbacks
        cp_callback = tf.keras.callbacks.ModelCheckpoint(LLAMINATE_MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=LLAMINATE_LOGS_PATH)
        # model fitting
        TRAINING_HISTORY = LLAMINATE.fit(
            x=DATASETS['ft-stack-exchange']['train'].batch(N_BATCH_DIM).prefetch(1),
            batch_size=None,
            epochs=N_EPOCHS,
            validation_split=None,
            validation_data=DATASETS['ft-stack-exchange']['test'].batch(N_BATCH_DIM).prefetch(1),
            validation_freq=list(range(1, N_EPOCHS + 1, 1)),
            class_weight=CLASS_WEIGHTS,
            verbose=1,
            callbacks=[cp_callback, tb_callback])
