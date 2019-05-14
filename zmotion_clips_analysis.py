import zmotion_clips2tfrecords as base
import data_io.basepy as basepy
import os.path as osp
import os
import cv2
import shutil
import numpy as np

VIDEO_FOLDER_SET = basepy.get_2tier_folder_path_list('/absolute/datasets/anoma')

VIDEO_FOLDER_PATH = VIDEO_FOLDER_SET[np.random.randint(0, len(VIDEO_FOLDER_SET))]
# VIDEO_FOLDER_PATH = '/absolute/datasets/anoma/Stealing/Stealing075_x264'
# VIDEO_FOLDER_PATH = '/absolute/datasets/anoma/Arson/Arson019_x264'
# VIDEO_FOLDER_PATH = '/absolute/datasets/anoma/Explosion/Explosion046_x264'
VIDEO_FOLDER_PATH = [VIDEO_FOLDER_PATH]

CLIPS_TFRECS_PATH = '/absolute/datasets/anoma_motion16_tfrecords'


def high_motion_visualization(sample_path_list=VIDEO_FOLDER_PATH, tfrecords_path='./temp/high_motion_visualization'):
    _ = basepy.check_or_create_path(tfrecords_path)
    _ = base.write_tfrecords(sample_path_list, tfrecords_path, visualization=True)


def divide_long_video_folder(video_folder_all=VIDEO_FOLDER_SET):
    info = []
    for video_folder_path in video_folder_all:
        frames_path = basepy.get_1tier_file_path_list(video_folder_path, suffix='.jpg')
        frames_path = sorted(frames_path, key=lambda x: int(osp.basename(x).split('.')[0]))

        frame_num = len(frames_path)
        frame_end = int(osp.basename(frames_path[-1]).split('.')[0])

        frame_sta = cv2.imread(frames_path[1])
        frame_shp = frame_sta.shape
        info.append([osp.basename(video_folder_path), (frame_num, frame_end), frame_shp])

    norm = [(i[0], i[1]) for i in info if 'normal' in i[0].lower()]
    anom = [(i[0], i[1]) for i in info if 'normal' not in i[0].lower()]

    norm = sorted(norm, key=lambda x: x[1][0], reverse=True)
    anom = sorted(anom, key=lambda x: x[1][0], reverse=True)

    max_anom = anom[0][1][0]
    need_divide = [i for i in norm if i[1][0] > max_anom]
    for video_name in need_divide:
        # ('Normal_Videos308_x264', [976504, 976504])
        video_folder = [i for i in video_folder_all if video_name[0] in i][0]
        sep = int(video_name[1][0] / 100000 + 1)
        frame_list = basepy.get_1tier_file_path_list(video_folder)
        frame_list = sorted(frame_list, key=lambda x: int(osp.basename(x).split('.')[0]))

        index = [int(i * len(frame_list) / sep) for i in range(sep + 1)]
        for i in range(sep):
            new_folder = basepy.check_or_create_path(video_folder.replace('_x264', '_%i_x264' % i))
            _ = [shutil.move(j, new_folder) for j in frame_list[index[i]:index[i + 1]]]


def find_empty_motion_video(tfrecs_path=CLIPS_TFRECS_PATH):
    tfr_path_list = basepy.get_1tier_file_path_list(tfrecs_path)
    folders = [i for i in tfr_path_list if osp.isdir(i)]
    tfrecos = [i[:-9].split('@')[1] for i in tfr_path_list if '.tfr' in i]
    txtfile = [i[:-9].split('@')[1] for i in tfr_path_list if '.txt' in i]

    tfr_not_txt = [i for i in tfrecos if i not in txtfile]

    empty_folder = [i for i in folders if not os.listdir(i)]
    return tfr_not_txt, empty_folder
