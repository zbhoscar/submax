import os.path as osp
import os
import tensorflow as tf
import numpy as np
import time
from pprint import pprint

import data_io.basepy as basepy
import zc3d_npy_base as base
import zfeatures_cliptxt2segmentnpy as io

import pickle

# Basic model parameters as external flags.
timestamp = time.strftime("%y%m%d%H%M%S", time.localtime())
tags = tf.flags
tags.DEFINE_integer('batch_size', 30, 'batch size.')
tags.DEFINE_integer('epoch_num', 400, 'epoch number.')
tags.DEFINE_float('learning_rate_base', 0.001, 'learning rate base')
tags.DEFINE_float('moving_average_decay', 0.99, 'moving average decay')
tags.DEFINE_float('regularization_scale', 0.0005, 'regularization scale')
tags.DEFINE_string('npy_file_path', '/absolute/datasets/anoma_1632_c3d_clips_features', 'npy file path')
tags.DEFINE_string('set_gpu', '0', 'Single gpu version, index select')
tags.DEFINE_string('save_setting', osp.join('/absolute/tensorflow_models', timestamp, timestamp + '.ckpt'),
                   'where to store tensorflow models')
F = tags.FLAGS

LABD1 = 0.00008
LABD2 = 0.00008
C3D_FEATURE_LENGTH = 4096
SEGMENT_NUMBER = 32
# BATCH_SIZE = F.batch_size
# EPOCH_NUMBER = F.epoch_num
# SAVE_PATH = F.save_setting

json_file = 'somefile.json'
data = F

f = open(json_file,'wb')
pickle.dumps(data, f)

f2 = open(json_file,'rb')
res = pickle.loads(f2)
