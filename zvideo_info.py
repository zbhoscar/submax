import data_io.basepy as basepy
import os.path as osp
import cv2
import json
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

json_file = 'video_info.json'

with open(json_file, 'r') as f:
    dread = json.load(f)

a = [(i[0],i[1]) for i in info if 'normal' in i[0].lower()]