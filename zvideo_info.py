import data_io.basepy as basepy
import os.path as osp
import cv2
import json
import os
import shutil
from pprint import pprint
import numpy as np


# def main():
video_folder_all = basepy.get_2tier_folder_path_list('/absolute/datasets/anoma')

info =[]
for video_folder_path in video_folder_all:

    frames_path = basepy.get_1tier_file_path_list(video_folder_path, suffix='.jpg')
    frames_path = sorted(frames_path, key=lambda x: int(osp.basename(x).split('.')[0]))

    frame_num = len(frames_path)
    frame_end = int(osp.basename(frames_path[-1]).split('.')[0])

    frame_sta = cv2.imread(frames_path[1])
    frame_shp = frame_sta.shape

    info.append([osp.basename(video_folder_path), (frame_num, frame_end), frame_shp])

print('wow')

json_file = 'video_info.json'

with open(json_file, 'w') as f:
    json.dump(info, f)

# ##################################################################################

import json

json_file = 'video_info.json'
with open(json_file, 'r') as f:
    info = json.load(f)

norm = [(i[0],i[1]) for i in info if 'normal' in i[0].lower()]
anom = [(i[0],i[1]) for i in info if 'normal' not in i[0].lower()]

norm = sorted(norm, key=lambda x: x[1][0], reverse=True)
anom = sorted(anom, key=lambda x: x[1][0], reverse=True)

max_anom = anom[0][1][0]

need_divide = [i for i in norm if i[1][0]>max_anom]

video_folder_all = basepy.get_2tier_folder_path_list('/absolute/datasets/anoma')
for video_name in need_divide:
    # ('Normal_Videos308_x264', [976504, 976504])
    video_folder = [i for i in video_folder_all if video_name[0] in i][0]
    sep=int(video_name[1][0]/100000+1)
    frame_list = basepy.get_1tier_file_path_list(video_folder)
    frame_list = sorted(frame_list,   key=lambda x:int(osp.basename(x).split('.')[0]))

    index = [int(i * len(frame_list) / sep) for i in range(sep + 1)]
    for i in range(sep):
        new_folder = basepy.check_or_create_path(video_folder.replace('_x264', '_%i_x264' % i))
        _ = [shutil.move(j, new_folder) for j in frame_list[index[i]:index[i+1]]]

# hist, bins = np.histogram(mobject, bins=[0, 0.1, 1, 2, 5, 10, 100])


# ##################################################################################
