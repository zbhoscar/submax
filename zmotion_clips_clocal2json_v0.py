import os.path as osp
import multiprocessing as mp
import numpy as np
import cv2
import data_io.basepy as basepy
import random
import json
import time


DATASET_PATH = '/absolute/datasets/UCFCrime2Local'
ANNOTATION_PATH = '/absolute/datasets/UCFCrime2Local/Txt annotations'
# CLIPS_JSON_PATH = CLIPS_JSON_PATH.replace('datasets', 'ext3t')
CLIPS_JSON_PATH = DATASET_PATH + '_motion_all_json'


def reform_txt2json(sample_path_list, tfrecords_path):
    for sample_path in sample_path_list:
        # eg. sample_path: '/absolute/datasets/UCFCrime2Local/Txt annotations/Arrest009.txt'
        read_txt = basepy.read_txt_lines2list(sample_path, ' ')
        # Specific for 'UCFCrime2Local/Txt annotations'
        read_txt = [[int(a1), int(a2), int(a3), int(a4), int(a5), int(a6), int(a7), int(a8), int(a9), a10[1:-1]]
                    for a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 in read_txt][:-16]

        list4json, video_name, nj, mean_value = [], str(osp.basename(sample_path).split('.')[0]) + '_x264', 0, 1
        for trackid, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label in read_txt:
            list4json.append([label, video_name, frame, nj, xmin, ymin, xmax - xmin, ymax - ymin, float(mean_value)])

        tfrecord_name = '%s@%s' % (label, video_name)
        tfrecord_path = osp.join(tfrecords_path, tfrecord_name + '.json')
        with open(tfrecord_path, 'w') as f:
            json.dump(list4json, f)
        del list4json, label


def main():
    # get remaining list for lasting
    remaining_list, _ = basepy.get_remaining_to_multi(
        basepy.get_1tier_file_path_list(ANNOTATION_PATH, suffix='.txt'),
        basepy.get_1tier_file_path_list(basepy.check_or_create_path(CLIPS_JSON_PATH, create=True, show=True)),
        if_print=True)

    # single processing
    reform_txt2json(remaining_list, CLIPS_JSON_PATH)

    print('------ Finish ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))


if __name__ == '__main__':
    main()
