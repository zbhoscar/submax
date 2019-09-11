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

JSON_FILE_LIST, DATASET_PATH = (
    ('/absolute/datasets/anoma_motion_pyramid_120_85_all_json_test', '/absolute/datasets/anoma'),
    ('/absolute/datasets/anoma_motion_pyramid_80_65_all_json', '/absolute/datasets/anoma'),
    ('/absolute/datasets/anoma_motion_pyramid_60_42_all_json', '/absolute/datasets/anoma'),
    'TYPE')[0]

EVAL_RESULT_FOLDER = JSON_FILE_LIST.replace('all_json', 'c3d_npy')

SET_GPU = [(0, 2), (1, 2), (2, 0), (3, 8)]
SPLIT_NUM, GPU_LIST, BATCH_SIZE = sum([i[1] for i in SET_GPU]), [], 1  # BATCH_SIZE: MUST be 1 to FIT pyramid
for gpu_id, num in SET_GPU:
    GPU_LIST.extend([str(gpu_id)] * num)


def _variable_on_cpu(name, shape, initializer):
    # with tf.device('/cpu:%d' % cpu_id):
    # with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('losses', weight_decay)
    return var


def run_test(json_path_list, dataset_path=None, eval_result_folder=None, batch_size=1, set_gpu='0'):
    clips_list_in_one = get_merge_list(json_path_list)
    # MAKE SURE NPY WRITING WORK

    clips_list_in_one.append(False)

    print('Program begins, timestamp %s ...' % time.asctime(time.localtime(time.time())))
    np_mean, step, stop_try = np.load('./crop_mean.npy'), 0, ValueError
    time_in_video_start, step_in_video, dict_key, dict_value, height = time.time(), 0, None, None, None
    try:
        while clips_list_in_one[step]:
            # batch_size MUST be 1
            one_clip = clips_list_in_one[step]

            e = np.random.rand(2, 4096)

            l2e = e / np.linalg.norm(e, ord=2, axis=1, keepdims=True)  # e
            l2e = l2e.flatten()

            class_name, video_name = one_clip[:2]
            class_video_name = class_name + '@' + video_name

            rest_in_clip = one_clip[2:]  # index, nj, (c, r, w, h) * n, mean_motion
            np_as_line = np.append(l2e, np.array(rest_in_clip + [max(e[0]) + min(e[0])], dtype='float32'))

            if class_video_name != dict_key:
                dict_key = class_video_name
                print('calculating %s...' % dict_key)
                height = [clip for clip in clips_list_in_one[:-1] if clip[1] == video_name].__len__()

                dict_value = np.empty([height, np_as_line.shape[0]], dtype='float32')
                dict_value[step_in_video] = np_as_line
            else:
                step_in_video += 1
                dict_value[step_in_video] = np_as_line

                if step_in_video == height - 1:
                    np.save(osp.join(eval_result_folder, dict_key), dict_value)
                    eval_npy=np.load(osp.join('/absolute/datasets/anoma_motion_pyramid_120_85_c3d_npy', class_video_name+'.npy'))
                    print(dict_value.shape, eval_npy.shape, height)
                    print('time in all: %.6f sec, per/step: %.6f sec, %d steps in all.'
                          % (time.time() - time_in_video_start, (time.time() - time_in_video_start)/height, height))
                    time_in_video_start, step_in_video, dict_key, dict_value, height = time.time(), 0, None, None, None
            step += 1
    except stop_try:
        print('------ Finish ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))

    # if class_video_name != dict_key and dict_key is None:
    #     dict_value = np.expand_dims(np_as_line, axis=0)
    #     dict_key = class_video_name
    #     print('calculating %s...' % dict_key)
    #     time_test4 = time.time()
    # elif class_video_name == dict_key:
    #     dict_value = np.concatenate((dict_value, np.expand_dims(np_as_line, axis=0)))
    # elif class_video_name != dict_key and dict_key is not None:
    #     np.save(osp.join(eval_result_folder, dict_key), dict_value)
    #     dict_value.shape[0],
    #     print('time in all: %.6f sec, per/step: %.6f sec.'
    #           % (time.time() - time_test4, (time.time() - time_test4)/(step-step_stamp)))
    #     # if set_gpu == '2':
    #     #     print('File %s.npy done, at step %d, next %s.' % (dict_key, step, class_video_name))
    #     del dict_value, dict_key
    #     dict_value = np.expand_dims(np_as_line, axis=0)
    #     dict_key = class_video_name
    #     print('calculating %s...' % dict_key)
    #     step_stamp = step
    # else:
    #     raise stop_try


def get_merge_list(json_path_list, frame_select=16):
    output = []
    for json_path in json_path_list:
        # eg. json_path = '/absolute/datasets/anoma_motion_all_json_type_1/normal_train@Normal_Videos308_3_x264.json'
        # get original list form json:

        with open(json_path, 'r') as f:
            clips_info = json.load(f)
        if frame_select:  # frame_select=16
            clips_info_to_add = []
            for i in clips_info:
                if i[2] % frame_select == 0:
                    clips_info_to_add.append(i)
        else:
            clips_info_to_add = clips_info

        # class_name, video_name = osp.basename(json_path)[:-5].split('@')
        #
        # video_path = osp.join(DATASET_PATH, class_name, video_name)
        # frame_list = basepy.get_1tier_file_path_list(video_path, suffix='.jpg')
        # frame_list = sorted(frame_list, key=lambda x: int(osp.basename(x).split('.')[0]))

        # if frame_list

        output.extend(clips_info_to_add)
    return output


def multi_json_clip_to_np(json_clips, dataset_path=None, clip_len=16, visualization='none'):
    return np.concatenate([one_json_clip_to_np(json_clip,
                                               dataset_path=dataset_path,
                                               clip_len=clip_len,
                                               visualization=visualization)
                           for json_clip in json_clips])


def suffix_in_dataset_path(dataset_path):
    if 'anoma' in dataset_path.lower():
        return '.jpg'
    elif 'ped' in dataset_path.lower():
        return '.tif'
    else:
        raise ValueError('Undefined suffix in %s' % dataset_path)


def json_clip_analysis(json_clip):
    class_name, video_name, index, nj = json_clip[:4]
    area_and_valus, get_areas_done = json_clip[4:], False
    split_stamp = int(len(area_and_valus) / 4)
    while not get_areas_done:
        temp_area = area_and_valus[:split_stamp * 4]
        if False not in [i == int(i) for i in temp_area]:
            clips_area = [int(j) for j in area_and_valus[:split_stamp * 4]]
            values = area_and_valus[split_stamp * 4:]
            get_areas_done = True
        else:
            split_stamp = split_stamp - 1

    return class_name, video_name, int(index), int(nj), clips_area, values


def one_json_clip_to_np(json_clip, dataset_path=None, clip_len=16, visualization='none'):
    frame_suffix = suffix_in_dataset_path(dataset_path)
    out_put = []
    # for class_name, video_name, index, nj, c, r, w, h, mean_motion in json_clips:
    class_name, video_name, index, nj, clips_area, values = json_clip_analysis(json_clip)
    original_video_path = osp.join(dataset_path, class_name, video_name)
    frames_path = basepy.get_1tier_file_path_list(original_video_path, suffix=frame_suffix)
    frame_list = sorted(frames_path, key=lambda x: int(osp.basename(x).split('.')[0]))

    if len(clips_area) % 4 != 0:
        raise ValueError('Wrong json clip: ', json_clip)
    else:
        # [5, 6, 7, 8, 9, 10, 11, 12] --> [[5, 6, 7, 8], [9, 10, 11, 12]]
        clips_area = [[i, clips_area[j + 1], clips_area[j + 2], clips_area[j + 3]]
                      for j, i in enumerate(clips_area) if j % 4 == 0]

    if visualization == 'pre':  # if True, dataset_path MUST a copy of original dataset_path
        for i in range(clip_len):
            image_path = frame_list[index + i]
            image_data = cv2.imread(image_path)
            for j, [c, r, w, h] in enumerate(clips_area):
                image_data = cv2.rectangle(image_data, (c, r), (c + w, r + h), int(255 / (j + 1)), 2)
            cv2.imwrite(image_path, image_data)
        return None
    elif visualization == 'post':
        post_prob = values[-1]
        for i in range(clip_len):
            image_path = frame_list[index + i]
            image_data = cv2.imread(image_path)
            for j, [c, r, w, h] in enumerate(clips_area):
                image_data = cv2.rectangle(image_data, (c, r), (c + w, r + h), (0, int(255 * post_prob), 0,), 2)
            cv2.imwrite(image_path, image_data)
    elif visualization == 'none':
        for c, r, w, h in clips_area:
            one = [cv2.resize(cv2.imread(frame_list[index + i])[r:r + h, c:c + w],
                              (112, 112))
                   for i in range(clip_len)]
            out_put.append(one)
        return np.array(out_put, dtype='float32')
    else:
        raise ValueError('Wrong type of visualization: %s .' % visualization)


def main(_):
    remaining_list, split_list = basepy.get_remaining_to_multi(
        basepy.get_1tier_file_path_list(JSON_FILE_LIST),
        basepy.get_1tier_file_path_list(basepy.check_or_create_path(EVAL_RESULT_FOLDER), suffix='.npy'),
        divide_num=SPLIT_NUM, if_print=True)

    run_test(remaining_list[1800:1805], DATASET_PATH, EVAL_RESULT_FOLDER, BATCH_SIZE, GPU_LIST[0])
    # p = mp.Pool(SPLIT_NUM)
    # for j, em in enumerate(split_list):
    #     p.apply_async(run_test, args=(em, DATASET_PATH, EVAL_RESULT_FOLDER, BATCH_SIZE, GPU_LIST[j]))
    # p.close()
    # p.join()


def json_visualization(class_at_video='RoadAccidents@RoadAccidents043_x264',
                       dataset_path=DATASET_PATH,
                       json_clip_path=JSON_FILE_LIST,
                       visualization_path='./temp/test_visualization',
                       visualization='post'):
    import shutil

    class_name, video_name = class_at_video.split('@')
    orig_path, visual_path = osp.join(dataset_path, class_name, video_name), \
                             osp.join(visualization_path, class_name, video_name)
    shutil.copytree(orig_path, visual_path)

    json_file_path = osp.join(json_clip_path, class_at_video + '.json')
    clips_list_in_one = get_merge_list([json_file_path])
    for one_clip in clips_list_in_one:
        _ = one_json_clip_to_np(one_clip, visualization_path, clip_len=8, visualization=visualization)


def patch_visualization(json_folder='/absolute/tensorflow_models/190624093140/190624093140.ckpt-51_ped_json',
                        dataset_path='/absolute/datasets/UCSDped2_reform',
                        json_clip_path='/absolute/tensorflow_models/190624093140/190624093140.ckpt-51_ped_json',
                        visualization_path='/absolute/tensorflow_models/190624093140/190624093140.ckpt-51_ped_json',
                        visualization='post'):
    # class_at_video_list = ['RoadAccidents@RoadAccidents043_x264', 'RoadAccidents@RoadAccidents043_x264']
    json_list = [i[:-5] for i in os.listdir(json_folder) if i.endswith('.json')]
    for i in json_list:
        _ = json_visualization(i, dataset_path=dataset_path, json_clip_path=json_clip_path,
                               visualization_path=visualization_path, visualization=visualization)


if __name__ == '__main__':
    tf.app.run()
