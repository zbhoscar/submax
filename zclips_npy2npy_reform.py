from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import os
import tensorflow as tf
import time
import numpy as np

import data_io.basepy as basepy
import multiprocessing as mp


def main(_):
    tags = tf.flags
    # Net config
    tags.DEFINE_string('npy_file_path',
                       '/absolute/datasets/anoma_motion_pyramid_120_85_c3d_npy',
                       'npy file folder to be reformed.')
    tags.DEFINE_string('testing_list',
                       '/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt',
                       'soru.')
    tags.DEFINE_boolean('multiprocessing', True, 'choose multiprocessing or not.')
    tags.DEFINE_integer('var0', 0, 'choose NPY_FILE_FOLDER, SEGMENT_NUM, TEST_FILE.')
    tags.DEFINE_integer('var1', 0, 'choose MULTISCALE, MULTIREGION.')
    F = tags.FLAGS
    REFORM_TYPE, REFORM_NUM = (('maxtop', 1000), ('segment', 32))[F.var0]
    MULTISCALE, MULTIREGION = (('pyramid', 4), ('pyramid', 1), ('single', 4), ('single', 1), (None, None))[F.var1]

    _ = npy_reform(F.npy_file_path, MULTISCALE, MULTIREGION, REFORM_TYPE, REFORM_NUM, F.multiprocessing, F.test_file)


def npy_reform(npy_file_folder_path,
               multiscale, multiregion, reform_type, reform_num,
               if_multiprocessing, test_file):
    try:
        results_folder_path = npy_file_folder_path.replace('_motion_', '_motion_reformed_') \
            .replace('_pyramid_', '_%s_' % multiscale) \
            .replace('_c3d_npy', '_%dregion_c3d_npy' % multiregion)
    except:
        results_folder_path = npy_file_folder_path.replace('_motion_', '_motion_reformed_')
    results_folder_path = results_folder_path.replace('_c3d_npy', '_%s_%d_c3d_npy' % (reform_type, reform_num))
    test_str = str(basepy.read_txt_lines2list(test_file, sep='  '))

    print('Converting %s to %s :' % (npy_file_folder_path, results_folder_path))
    multiprocessing_num = int(mp.cpu_count() / 4)
    remaining_list, split_list = basepy.get_remaining_to_multi(
        basepy.get_1tier_file_path_list(npy_file_folder_path, '.npy'),
        basepy.get_1tier_file_path_list(basepy.check_or_create_path(results_folder_path), suffix='.npy'),
        divide_num=multiprocessing_num, if_print=True)
    # npy_list_preprocessing(remaining_list, EVAL_RESULT_FOLDER, MULTISCALE, MULTIREGION)
    if if_multiprocessing:
        p = mp.Pool(multiprocessing_num)
        for j, em in enumerate(split_list):
            p.apply_async(npy_list_preprocessing,
                          args=(em,
                                results_folder_path, multiscale, multiregion, reform_type, reform_num, test_str))
        p.close()
        p.join()
    else:
        npy_list_preprocessing(remaining_list,
                               results_folder_path, multiscale, multiregion, reform_type, reform_num, test_str)
    # END
    print('Converting DONE ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))
    return results_folder_path


def npy_list_preprocessing(npy_file_list, eval_result_folder,
                           multiscale, multiregion, reform_type, number_in_one, test_str):
    for npy_file in npy_file_list:
        npy_preprocessing(npy_file, eval_result_folder, multiscale, multiregion, reform_type, number_in_one, test_str)


def npy_preprocessing(npy_file, eval_result_folder, multiscale, multiregion, reform_type, reform_num, test_str):
    # print('processing %s ...' % npy_file)
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

    # DIFFERENT STRATEGY IN TRAINING AND TESTING
    if osp.basename(npy_file).split('@')[1].split('.')[0] in test_str:
        new_npy_data = npy_data
    else:
        if reform_type == 'segment':
            if multiregion == 4:
                new_npy_data = np.array(
                    [merge_1region_2segment(np.array([line for line in npy_data if line[4097] == 0]), reform_num),
                     merge_1region_2segment(np.array([line for line in npy_data if line[4097] == 1]), reform_num),
                     merge_1region_2segment(np.array([line for line in npy_data if line[4097] == 2]), reform_num),
                     merge_1region_2segment(np.array([line for line in npy_data if line[4097] == 3]), reform_num)])
                new_npy_data = new_npy_data.reshape([-1, new_npy_data.shape[-1]], order='F')
            else:
                new_npy_data = merge_1region_2segment(npy_data, reform_num)
        elif reform_type == 'maxtop':
            if multiregion == 4:
                new_npy_data = np.array(
                    [max_1region_select(np.array([line for line in npy_data if line[4097] == 0]), reform_num),
                     max_1region_select(np.array([line for line in npy_data if line[4097] == 1]), reform_num),
                     max_1region_select(np.array([line for line in npy_data if line[4097] == 2]), reform_num),
                     max_1region_select(np.array([line for line in npy_data if line[4097] == 3]), reform_num)])
                new_npy_data = new_npy_data.reshape([-1, new_npy_data.shape[-1]], order='F')
            else:
                new_npy_data = max_1region_select(npy_data, reform_num)
        else:
            raise ValueError('Wrong reform_type: %s' % reform_type)

    npy_result_file = osp.join(eval_result_folder, osp.basename(npy_file))
    np.save(npy_result_file, new_npy_data)


def merge_1region_2segment(npy_data, segment_num):
    # npy_data must in size 4096 + n
    npy_data_num = npy_data.shape[0]
    segment_split = [int(i * (npy_data_num) / segment_num) for i in range(segment_num)]
    npy_segment_data = []
    for j, segment in enumerate(segment_split):
        try:
            start, end = [segment, max(segment + 1, segment_split[j + 1])]
        except:
            start, end = [segment, npy_data_num]
        npy_segment_data.append(np.average(npy_data[start:end], axis=0))
    return np.array(npy_segment_data)


def max_1region_select(npy_data, num_from_max):
    # npy_data must in size 4096 + n
    data = npy_data[np.argsort(-npy_data[:, -2])]
    return data[:num_from_max]


def clear_npy_for_testing(npy_file_folder, test_str):
    # del reform folder: npy_file_folder
    npy_list = basepy.get_1tier_file_path_list(npy_file_folder, '.npy')
    j = 0
    for npy in npy_list:
        if osp.basename(npy).split('@')[1].split('.')[0] in str(test_str):
            os.remove(npy)
            print('remove %d-th file: %s' % (j, npy))
            j += 1


if __name__ == '__main__':
    tf.app.run()
