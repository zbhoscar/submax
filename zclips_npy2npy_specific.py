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

NPY_FILE_FOLDER, SEGMENT_NUM = (('/absolute/datasets/anoma_motion_pyramid_120_85_c3d_npy', 32),
                                ('/absolute/datasets/anoma_motion_pyramid_80_56_c3d_npy', 32),
                                ('/absolute/datasets/anoma_motion_pyramid_60_42_c3d_npy', 32),
                                ('/absolute/datasets/anoma_motion_original_c3d_npy', 32))[0]
MULTISCALE, MULTIREGION = (('single', 1), ('single', 4), ('pyramid', 1), ('pyramid', 4), (None, None))[0]
try:
    EVAL_RESULT_FOLDER = NPY_FILE_FOLDER.replace('_motion_', '_motion_4training_') \
        .replace('_pyramid_', '_%s_' % MULTISCALE) \
        .replace('_c3d_npy', '_%dregion_c3d_npy' % MULTIREGION)
except:
    EVAL_RESULT_FOLDER = NPY_FILE_FOLDER.replace('_motion_', '_motion_4training_')
SPLIT_NUM = mp.cpu_count()


def npy_list_preprocessing(npy_file_list, eval_result_folder, multiscale, multiregion, segment_num):
    for npy_file in npy_file_list:
        npy_preprocessing(npy_file, eval_result_folder, multiscale, multiregion, segment_num)


def npy_preprocessing(npy_file, eval_result_folder, multiscale, multiregion, segment_num):
    print('converting %s ...' % npy_file)
    npy_data = np.load(npy_file)
    if multiregion == 1:
        line_split = [i for i in range(npy_data.shape[0]) if i % 4 == 0]
        for j, line in enumerate(line_split):
            motion_value_split = npy_data[line:line + 4, -2]
            max_index = np.argmax(motion_value_split)
            line_split[j] = line + max_index
        npy_data = npy_data[line_split]

    # npy_data.shape[1] // 4096
    if multiscale == 'single':
        npy_data = np.concatenate((npy_data[:, :4096], npy_data[:, 8192:]), axis=1)
    elif multiscale == 'pyramid':
        npy_data = np.concatenate((np.maximum(npy_data[:, :4096], npy_data[:, 4096:8192]), npy_data[:, 8192:]), axis=1)

    if multiregion == 4:
        new_npy_data = np.array(
            [merge_1region_2segment(np.array([line for line in npy_data if line[4097] == 0]), segment_num=segment_num),
             merge_1region_2segment(np.array([line for line in npy_data if line[4097] == 1]), segment_num=segment_num),
             merge_1region_2segment(np.array([line for line in npy_data if line[4097] == 2]), segment_num=segment_num),
             merge_1region_2segment(np.array([line for line in npy_data if line[4097] == 3]), segment_num=segment_num)])
        new_npy_data = new_npy_data.reshape([-1, new_npy_data.shape[-1]], order='F')
    else:
        new_npy_data = merge_1region_2segment(npy_data, segment_num=segment_num)

    npy_result_file = osp.join(eval_result_folder, osp.basename(npy_file))
    np.save(npy_result_file, new_npy_data)


def merge_1region_2segment(npy_data, segment_num=32):
    # npy_data must in size 4096 + n
    npy_data_num = npy_data.shape[0]
    segment_split = [int(i * (npy_data_num) / segment_num) for i in range(segment_num)]
    npy_segment_data = []
    for j, segment in enumerate(segment_split):
        try:
            start, end = [segment, max(segment + 1,segment_split[j + 1])]
        except:
            start, end = [segment, npy_data_num]
        npy_segment_data.append(np.average(npy_data[start:end], axis=0))
    return np.array(npy_segment_data)


def main(_):
    remaining_list, split_list = basepy.get_remaining_to_multi(
        basepy.get_1tier_file_path_list(NPY_FILE_FOLDER, '.npy'),
        basepy.get_1tier_file_path_list(basepy.check_or_create_path(EVAL_RESULT_FOLDER), suffix='.npy'),
        divide_num=SPLIT_NUM, if_print=True)

    print('%s to %s :' % (NPY_FILE_FOLDER, EVAL_RESULT_FOLDER))
    # npy_list_preprocessing(remaining_list, EVAL_RESULT_FOLDER, MULTISCALE, MULTIREGION)
    p = mp.Pool(SPLIT_NUM)
    for j, em in enumerate(split_list):
        p.apply_async(npy_list_preprocessing, args=(em, EVAL_RESULT_FOLDER, MULTISCALE, MULTIREGION, SEGMENT_NUM))
    p.close()
    p.join()

    # END
    print('------ Finish ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))


if __name__ == '__main__':
    tf.app.run()
