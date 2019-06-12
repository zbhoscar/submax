import os.path as osp
import multiprocessing as mp
import numpy as np
import cv2
import data_io.basepy as basepy
import random
import json
import time

# TRACK_WINDOW: cv2 format: c, r, w, h                    # -> start, -v start, -> length, -v length
# AREA_CROPS: numpy format: shape = (240, 320, 3)         # (h, w, channel)
# 0, (h + edge) / 2, (h - edge) / 2, h , 0, (w + edge) / 2, (w - edge) / 2 , w
DATASET_PATH, FRAME_SUFFIX, TRACK_WINDOW, AREA_CROPS, CLIP_LEN, STEP, OPTICAL, CRITERIA, TYPE = (
    ('/absolute/datasets/anoma', '.jpg', (70, 50, 50, 50),
     ((0, 150, 0, 190), (0, 150, 130, 320), (90, 240, 0, 190), (90, 240, 130, 320)),
     16, 8, 2, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1), 'standard_50'),
    ('/absolute/datasets/anoma', '.jpg', (50, 30, 120, 120),
     ((0, 180, 0, 220), (0, 180, 100, 320), (60, 240, 0, 220), (60, 240, 100, 320)),
     16, 8, 2, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1), 'standard_120'),
    ('/absolute/datasets/anoma', '.jpg', (50, 30, 120, 120),
     ((0, 180, 0, 220), (0, 180, 100, 320), (60, 240, 0, 220), (60, 240, 100, 320)),
     16, 8, 2, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1), 'pyramid_120_85_60'),
    ('/absolute/datasets/UCSDped1', '.tif', (40, 20, 80, 80),
     ((0, 119, 0, 159), (0, 119, 79, 238), (39, 158, 0, 159), (39, 158, 79, 238)),
     16, 8, 2, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1), 'standard_80'),
    ('/absolute/datasets/UCSDped1', '.tif', (40, 20, 80, 80),
     ((0, 119, 0, 159), (0, 119, 79, 238), (39, 158, 0, 159), (39, 158, 79, 238)),
     16, 8, 2, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1), 'pyramid_80_60'))[4]

# CLIPS_JSON_PATH = CLIPS_JSON_PATH.replace('datasets', 'ext3t')
CLIPS_JSON_PATH = DATASET_PATH + '_motion_%s_all_json' % TYPE


def write_all_clips2json(sample_path_list, tfrecords_path):
    for sample_path in sample_path_list:
        # eg. sample_path: '/absolute/datasets/anoma/Abuse/Abuse001_x264'k
        # eg. sample_path: '/absolute/datasets/anoma/normal_train/Normal_Videos308_3_x264'
        video_name = osp.basename(sample_path)
        class_name = osp.basename(osp.dirname(sample_path))

        tfrecord_name = '%s@%s' % (class_name, video_name)
        tfrecord_path = osp.join(tfrecords_path, tfrecord_name + '.json')

        frames_path = basepy.get_1tier_file_path_list(sample_path, suffix=FRAME_SUFFIX)
        frame_list = sorted(frames_path, key=lambda x: int(osp.basename(x).split('.')[0]))

        # get every STEP frame index
        every_step_list = [i for i in range(len(frame_list) - CLIP_LEN) if i % STEP == 0]

        list4json = []
        for index in every_step_list:
            frames = [cv2.imread(frame_list[index + i]) for i in range(CLIP_LEN + 1)]
            grays = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in frames]
            flows = [cv2.calcOpticalFlowFarneback(grays[i * OPTICAL], grays[(i + 1) * OPTICAL],
                                                  None, 0.5, 3, 15, 3, 7, 1.5, 0)
                     for i in range(int(CLIP_LEN / OPTICAL))]
            motion = np.sum([np.linalg.norm(flow, ord=None, axis=2) for flow in flows], axis=0)

            for nj, area in enumerate(AREA_CROPS):
                motion_crop = motion[area[0]:area[1], area[2]:area[3]]
                _, window_temp = cv2.meanShift(motion_crop, TRACK_WINDOW, CRITERIA)
                c, r, w, h = window_temp
                mobject = motion_crop[r:r + h, c:c + w]
                mean_value = np.mean(mobject)

                if 'standard' in TYPE:
                    list4json.append([class_name, video_name,
                                      index, nj, c + area[2], r + area[0], w, h, float(mean_value)])
                elif 'pyramid' in TYPE:
                    area_list = [c + area[2], r + area[0], w, h]
                    pyramid_edges = [int(i) for i in TYPE.split('_')[1:]]
                    for j, edge in enumerate(pyramid_edges):
                        if j > 0:
                            pyramid_window = (int((pyramid_edges[j - 1] - edge) / 2),
                                              int((pyramid_edges[j - 1] - edge) / 2), edge, edge)
                            _, pyramid_temp = cv2.meanShift(mobject, pyramid_window, CRITERIA)
                            pc, pr, pw, ph = pyramid_temp
                            area_list = area_list + [pc + c + area[2], pr + r + area[0], pw, ph]
                    list4json.append([class_name, video_name, index, nj] + area_list + [float(mean_value)])

        with open(tfrecord_path, 'w') as f:
            json.dump(list4json, f)
        del list4json


def main():
    _ = basepy.check_or_create_path(CLIPS_JSON_PATH, create=True, show=True)

    # write tfrecords
    # sample_path_list = basepy.get_2tier_folder_path_list(DATASET_PATH)
    remaining_list, _ = basepy.get_remaining_to_multi(basepy.get_2tier_folder_path_list(DATASET_PATH),
                                                      basepy.get_1tier_file_path_list(CLIPS_JSON_PATH), if_print=True)
    random.shuffle(remaining_list)
    # single processing
    # write_all_clips2json(remaining_list, CLIPS_JSON_PATH)
    basepy.non_output_multiprocessing(write_all_clips2json, remaining_list, CLIPS_JSON_PATH,
                                      num=int(mp.cpu_count()))

    print('------ Finish ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))


if __name__ == '__main__':
    main()
