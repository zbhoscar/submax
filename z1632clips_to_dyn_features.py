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
import numpy as np
import time

import c3d_model
import z1632clips_frame2tfrecords as base_io
import data_io.basepy as basepy

# Basic model parameters as external flags.
flags = tf.flags
gpu_num = 2
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
FLAGS = flags.FLAGS
BATCH_SIZE = 1


def get_input(file_path_list, num_epochs=None, is_training=True, batch_size=64, preprocessing='dynamic_image'):
    with tf.name_scope('input'):
        classb, videob, clipb, segb, imageb = base_io.read_tfrecords(file_path_list,
                                                                     num_epochs=num_epochs,
                                                                     is_training=is_training,
                                                                     batch_size=batch_size,
                                                                     preprocessing=preprocessing)
    return classb, videob, clipb, segb, imageb


def run_test():
    # test_list_file = 'list/test.list'
    # num_test_videos = len(list(open(test_list_file,'r')))
    # print("Number of test videos={}".format(num_test_videos))

    # Get the sets of images and labels for training, validation, and
    # images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)

    classb, videob, clipb, segb, imageb = get_input(basepy.get_1tier_file_path_list(base_io.CLIPS_TFRECS_PATH),
                                                    num_epochs=1, is_training=False, batch_size=BATCH_SIZE)

    features = tf.nn.max_pool(imageb, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    features = tf.reshape(features, [BATCH_SIZE, -1])
    features = tf.nn.l2_normalize(features, axis=1)

    timestamp, step = time.time(), 0
    eval_result_folder = basepy.check_or_create_path(
        base_io.DATASET_PATH + '_%d%d_dyn_clips_features' % (base_io.CLIP_LENGTH, base_io.SEGMENT_NUMBER),
        create=True, show=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    # saver = tf.train.Saver()

    init_op = (tf.local_variables_initializer(), tf.global_variables_initializer())

    with tf.Session(config=config) as sess:
        # sess = tf.Session(config=config)
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # one step to see
        # a, b, c, d, e = sess.run([classb, videob, clipb, segb, features])
        # print('wow')
        print('program begins, timestamp %s' % time.asctime(time.localtime(time.time())))

        try:
            while True:
                a, b, c, d, e = sess.run([classb, videob, clipb, segb, features])

                for i in range(len(c)):
                    class_video_name = str(a[i], encoding='utf-8') + '@' + str(b[i], encoding='utf-8')
                    feature_txt_path = os.path.join(eval_result_folder, class_video_name + '.txt')

                    _ = basepy.write_txt_add_lines(feature_txt_path, str(e[i].tolist()), str(d[i]), str(c[i]))

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
