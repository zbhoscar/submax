# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf
import time
# import numpy as np

import c3d_model
import zmotion_clip2tfrecords as base_io
import data_io.basepy as basepy

# # Basic model parameters as external flags.
flags = tf.flags
flags.DEFINE_string('txt_list_path', './temp/to_c3d_0.txt', 'read .tfr from txt')
flags.DEFINE_string('set_gpu', '0', 'Single gpu version, index select')
flags.DEFINE_integer('batch_size', 1, 'batch size.')
FLAGS = flags.FLAGS

# CLIPS_TFRECS_PATH = CLIPS_TFRECS_PATH.replace('datasets', 'ext3t')
EVAL_RESULT_FOLDER = basepy.check_or_create_path(
    base_io.CLIPS_TFRECS_PATH.replace('tfrecords', 'c3d_features').replace('datasets', 'ext3t'),
    create=True, show=True)


def _variable_on_cpu(name, shape, initializer):
    # with tf.device('/cpu:%d' % cpu_id):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('losses', weight_decay)
    return var


def get_input(file_path_list, num_epochs=None, is_training=True, batch_size=64):
    with tf.name_scope('input'):
        classb, videob, indexb, cropb, cb, rb, wb, hb, imageb = base_io.read_tfrecords(file_path_list,
                                                                                       num_epochs=num_epochs,
                                                                                       is_training=is_training,
                                                                                       batch_size=batch_size)
    return classb, videob, indexb, cropb, cb, rb, wb, hb, imageb


def run_test():
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Remove 2 videos with no motion crop, in normal_train set
    #   empty folder / empty .tfr / no .txt
    #   see zvideo_info.py part3 to find out
    # Split 9 videos in shorter length, in normal_train set
    # Empty tfrecords make no effects
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    list = basepy.read_txt_lines2list(FLAGS.txt_list_path)
    list = [i[0] for i in list]
    classb, videob, indexb, cropb, cb, rb, wb, hb, imageb = get_input(
        list,
        num_epochs=1, is_training=False, batch_size=FLAGS.batch_size)

    with tf.variable_scope('var_name'):  # as var_scope:
        weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
        }
        biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
        }

    features = c3d_model.inference_c3d(imageb, 0.6, 1, weights, biases)
    # norm_score = tf.nn.softmax(logits)

    timestamp, step = time.time(), 0
    model_name = "./sports1m_finetuning_ucf101.model"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.set_gpu
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    saver = tf.train.Saver()

    init_op = (tf.local_variables_initializer(), tf.global_variables_initializer())

    with tf.Session(config=config) as sess:
        # sess = tf.Session(config=config)
        sess.run(init_op)

        saver.restore(sess, model_name)
        print("Model Loading Done!")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # one step to see
        # a, b, c, d, e = sess.run([classb, videob, clipb, segb, features])
        # print('wow')
        print('program begins, timestamp %s' % time.asctime(time.localtime(time.time())))

        try:
            while True:
                a, b, c, d, ac, ar, aw, ah, e = sess.run([classb, videob, indexb, cropb, cb, rb, wb, hb, features])

                with tf.device('/cpu:0'):
                    l2e = e     # e / np.linalg.norm(e, ord=2, axis=1, keepdims=True)
                    for i in range(len(c)):
                        class_video_name = str(a[i], encoding='utf-8') + '@' + str(b[i], encoding='utf-8')
                        feature_txt_path = os.path.join(EVAL_RESULT_FOLDER, class_video_name + '.txt')

                        _ = basepy.write_txt_add_lines(feature_txt_path, str(l2e[i].tolist()),
                                                       str(c[i]), str(d[i]),
                                                       str(ac[i]), str(ar[i]), str(aw[i]), str(ah[i]),
                                                       str(max(l2e[i])), str(min(l2e[i])))

                    step += 1
                    if time.time() - timestamp > 1800:
                        localtime = time.asctime(time.localtime(time.time()))
                        average_time_per_step = (time.time() - timestamp) / step
                        print('program ongoing, timestamp %s, per step %s sec' % (localtime, average_time_per_step))
                        step, timestamp = 0, time.time()

        except Exception as error:
            coord.request_stop(error)

        coord.request_stop()
        coord.join(threads)

    print("done, at %s" % time.asctime(time.localtime(time.time())))
    print('debug symbol')


def main(_):
    run_test()


if __name__ == '__main__':
    tf.app.run()
