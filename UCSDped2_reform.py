import data_io.basepy as basepy
import os.path as osp
import random
import os
import cv2
import shutil
import numpy as np
import json
import time

DATASET_PATH = '/absolute/datasets/UCSDped2'
TRAIN_TEST_LIST_FOLDER = '/absolute/datasets/UCSDped2_split_list/10_fold_001'


def make_fold_split_list(path_list, valid_num, anom_in_train, norm_in_train, write_to=None):
    # remove DATASET_PATH -> 'Anomaly/Anomaly010'
    path_list = [osp.join(i.split('/')[-2], i.split('/')[-1]) for i in path_list]
    # divide to anom norm
    anom_list = [i for i in path_list if 'anomaly' in i.lower()]
    norm_list = [i for i in path_list if 'normal' in i.lower()]

    _ = basepy.check_or_create_path(TRAIN_TEST_LIST_FOLDER, create=True, show=False)
    for i in range(valid_num):
        random.shuffle(anom_list)
        random.shuffle(norm_list)
        txt_for_train = osp.join(TRAIN_TEST_LIST_FOLDER, 'v'+str(i).zfill(2)+'_train.txt')
        txt_for_test = txt_for_train.replace('train', 'test')

        list_for_train = anom_list[:anom_in_train] + norm_list[:norm_in_train]
        list_for_test = anom_list[anom_in_train:] + norm_list[norm_in_train:]

        _ = [basepy.write_txt_add_lines(txt_for_train, i, sep=' ') for i in list_for_train]
        _ = [basepy.write_txt_add_lines(txt_for_test, i, sep=' ') for i in list_for_test]


def main():
    video_path_list = basepy.get_2tier_folder_path_list(DATASET_PATH)

    video_reform_list = [i.replace('Test', 'Anomaly') if 'Test' in i else i.replace('Train', 'Normal')
                         for i in video_path_list]
    video_reform_list = [i.replace('UCSDped2', 'UCSDped2_reform')for i in video_reform_list]

    for j,i in enumerate(video_path_list):
        if not osp.isdir(video_reform_list[j]):
            shutil.copytree(i, video_reform_list[j])

    _ = make_fold_split_list(video_reform_list, 10, 6, 4, TRAIN_TEST_LIST_FOLDER)


    print('------ Finish ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))


if __name__ == '__main__':
    main()
