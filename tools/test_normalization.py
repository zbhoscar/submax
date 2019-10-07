import tensorflow as tf
import numpy as np
import os

slim = tf.contrib.slim


ap = [[5.,12.],[3.,4.]]
at = tf.convert_to_tensor(ap, dtype=tf.float32)
an = tf.nn.l2_normalize(at, axis=1)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)
with tf.Session(config=config) as sess:
    result = sess.run(an)

    print('wow')
