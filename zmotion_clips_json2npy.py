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

import os.path as osp
import os
import tensorflow as tf
import time
import numpy as np

import c3d_model
import json
import cv2
import data_io.basepy as basepy
import multiprocessing as mp

SET_GPU = [[(0, 2), (1, 2), (2, 2), (3, 2)],
           [(0, 3), (3, 4)]][1]
BATCH_SIZE = 1
JSON_FILE_LIST = '/home/zbh/Desktop/absolute/datasets/anoma_motion_all_json_type_1'
EVAL_RESULT_FOLDER = '/absolute/datasets/anoma_motion_c3d_features_type_1_simple_1001_new'
DATASET_PATH = '/home/zbh/Desktop/absolute/datasets/anoma'

SPLIT_NUM, GPU_LIST = sum([i[1] for i in SET_GPU]), []
for gpu_id, num in SET_GPU:
    GPU_LIST.extend([str(gpu_id)] * num)


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


def get_merge_list(json_path_list, reduce_method='simple', clip_len=16, reduce_num=1001):
    output = []
    for json_path in json_path_list:
        # eg. json_path: '/absolute/datasets/anoma_motion_all_json_type_1/normal_train@Normal_Videos308_3_x264.json'
        # get original list form json:
        with open(json_path, 'r') as f:
            clips_info = json.load(f)
        # how to reduce clip json
        if reduce_method == 'simple':
            clips_info = [
                i for i in clips_info if i[2] % clip_len == 0] if len(clips_info) > reduce_num * 3 else clips_info
            clips_info = sorted(clips_info, key=lambda x: x[-1], reverse=True)[:reduce_num]
        else:
            raise ValueError('Wrong reduce method: %s' % reduce_method)
        output.extend(clips_info)
    return output


def run_test(json_path_list, dataset_path=None, eval_result_folder=None, batch_size=1, set_gpu='0'):
    clips_list_in_one = get_merge_list(json_path_list)
    # MAKE SURE NPY WRITING WORK
    clips_list_in_one = clips_list_in_one + clips_list_in_one[:batch_size]

    with tf.name_scope('input'):
        image_batch = tf.placeholder(tf.float32, [batch_size, 16, 112, 112, 3], name='image_batch')

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

    features = c3d_model.inference_c3d(image_batch, 0.6, 1, weights, biases)
    # norm_score = tf.nn.softmax(logits)

    timestamp, step = time.time(), 0
    model_name = "./sports1m_finetuning_ucf101.model"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    os.environ["CUDA_VISIBLE_DEVICES"] = set_gpu
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    saver = tf.train.Saver()

    init_op = (tf.local_variables_initializer(), tf.global_variables_initializer())

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        saver.restore(sess, model_name)
        print("Model Loading Done!")

        print('Program begins, timestamp %s ...' % time.asctime(time.localtime(time.time())))
        try:
            sym_error = ValueError
            max_index = len(clips_list_in_one)
            np_mean = np.load('./crop_mean.npy')
            dict_key = None
            while True:
                batch_start, batch_end = step * batch_size, step * batch_size + batch_size

                if batch_end > max_index:
                    raise sym_error

                clips_list_batch = clips_list_in_one[batch_start:batch_end]

                image_data = json_clips_to_np(clips_list_batch, dataset_path)

                rand1, rand2 = np.random.randint(0, 8), np.random.randint(0, 8)
                image_data = image_data[:, :, rand1:rand1 + 112, rand2:rand2 + 112, :] - np_mean

                e = sess.run(features, feed_dict={image_batch: image_data})

                l2e = e / np.linalg.norm(e, ord=2, axis=1, keepdims=True)  # e
                for j, [class_name, video_name, index, nj, c, r, w, h, mean_motion] in enumerate(clips_list_batch):
                    class_video_name = class_name + '@' + video_name
                    np_as_line = np.append(l2e[j], np.array(
                        [index, nj, c, r, w, h, mean_motion, max(l2e[j]), min(l2e[j])], dtype='float32'))

                    if class_video_name != dict_key and dict_key is None:
                        dict_value = np.expand_dims(np_as_line, axis=0)
                        dict_key = class_video_name
                    elif class_video_name == dict_key:
                        dict_value = np.concatenate((dict_value, np.expand_dims(np_as_line, axis=0)))
                    elif class_video_name != dict_key and dict_key is not None:
                        np.save(osp.join(eval_result_folder, dict_key), dict_value)
                        dict_value = np.expand_dims(np_as_line, axis=0)
                        dict_key = class_video_name
                    else:
                        raise sym_error

                step += 1
                if time.time() - timestamp > 1800:
                    localtime = time.asctime(time.localtime(time.time()))
                    average_time_per_step = (time.time() - timestamp) / step
                    print('Program ongoing, timestamp %s, per step %.6f sec' % (localtime, average_time_per_step))
                    step, timestamp = 0, time.time()

        except sym_error:
            print('------ Finish ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))


def json_clips_to_np(json_clips, dataset_path=None, clip_len=16):
    out_put = []
    for class_name, video_name, index, nj, c, r, w, h, mean_motion in json_clips:
        original_video_path = osp.join(dataset_path, class_name, video_name)
        frames_path = basepy.get_1tier_file_path_list(original_video_path, suffix='.jpg')
        frame_list = sorted(frames_path, key=lambda x: int(osp.basename(x).split('.')[0]))

        one = [cv2.imread(frame_list[index + i])[r:r + h, c:c + w] for i in range(clip_len)]
        out_put.append(one)
    return np.array(out_put, dtype='float32')


def main(_):
    remaining_list, split_list = basepy.get_remaining_to_multi(
        basepy.get_1tier_file_path_list(JSON_FILE_LIST),
        basepy.get_1tier_file_path_list(basepy.check_or_create_path(EVAL_RESULT_FOLDER), suffix='.npy'),
        divide_num=SPLIT_NUM, if_print=True)

    p = mp.Pool(SPLIT_NUM)
    for j, em in enumerate(split_list):
        p.apply_async(run_test, args=(em, DATASET_PATH, EVAL_RESULT_FOLDER, BATCH_SIZE, GPU_LIST[j]))
    p.close()
    p.join()

    print('------ Finish ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))


if __name__ == '__main__':
    tf.app.run()
