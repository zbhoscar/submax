import data_io.basepy as basepy
import os.path as osp
import cv2
import numpy as np
import os

# def main():
video_folder_all = basepy.get_2tier_folder_path_list('/absolute/datasets/anoma')

video_folder_path = video_folder_all[np.random.randint(0, len(video_folder_all))]
# video_folder_path = '/absolute/datasets/anoma/Stealing/Stealing075_x264'
# video_folder_path = '/absolute/datasets/anoma/Arson/Arson019_x264'
# video_folder_path = '/absolute/datasets/anoma/Explosion/Explosion046_x264'

frames_path = basepy.get_1tier_file_path_list(video_folder_path, suffix='.jpg')
frame_list = sorted(frames_path, key=lambda x: int(osp.basename(x).split('.')[0]))

CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
CLIP_LENGTH = 16
_OPTICAL = 2
STEP = 8

# cv2 format: c, r, w, h                    # -> start, -v start, -> length, -v length
TRACK_WINDOW = [(70, 50, 50, 50),
                (50, 30, 120, 120)][1]
# np  format: shape = (240, 320, 3)         # (h, w, channel)
AREA_CROPS = [[(0, 150, 0, 190),(0, 150, 130, 320),(90, 240, 0, 190),(90, 240, 130, 320)],
              [(0, 180, 0, 220),(0, 180, 100, 320),(60, 240, 0, 220),(60, 240, 100, 320)]][1]

P_folder_path = '/absolute/ext3t/anoma_motion16_tfrecords/test'
class_name = 'class'
video_name = 'video'
tfrecord_file_name = class_name + '@' + video_name

for index in range(len(frame_list) - CLIP_LENGTH):
    # frame = cv2.imread(frame_list[index])
    if index % STEP == 0:
        frames = [cv2.imread(frame_list[index + i]) for i in range(CLIP_LENGTH + 1)]
        grays = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in frames]
        flows = [cv2.calcOpticalFlowFarneback(grays[i * _OPTICAL], grays[(i + 1) * _OPTICAL],
                                              None, 0.5, 3, 15, 3, 7, 1.5, 0)
                 for i in range(int(CLIP_LENGTH / _OPTICAL))]
        motion = np.sum([np.linalg.norm(flow, ord=None, axis=2) for flow in flows], axis=0)

        for nj, area in enumerate(AREA_CROPS):
            motion_crop = motion[area[0]:area[1], area[2]:area[3]]
            retval, window_temp = cv2.meanShift(motion_crop, TRACK_WINDOW, CRITERIA)
            c, r, w, h = window_temp
            mobject = motion_crop[r:r + h, c:c + w]
            # hist, bins = np.histogram(mobject, bins=[0, 0.1, 1, 2, 5, 10, 100])
            # print(window_temp, hist)

            frame = frames[0]
            if np.mean(mobject) > 0.07 * CLIP_LENGTH:
                # _ = basepy.write_txt_add_lines(os.path.join(P_folder_path, 'motion_crops.txt'),
                #                                class_name, video_name,
                #                                str(index), str(nj), str(c), str(r), str(w), str(h))
                # frame_crops_path = [
                #     os.path.join(P_folder_path,
                #                  tfrecord_file_name + '_' + str(index) + '_' + str(nj) + '_' + str(i)) + '.jpg'
                #     for i in range(CLIP_LENGTH)]
                #
                # _ = [cv2.imwrite(frame_crops_path[i], frames[i][r:r + h, c:c + w]) for i in range(CLIP_LENGTH)]

                frame = cv2.rectangle(frame, (c + area[2], r + area[0]), (c + area[2] + w, r + area[0] + h), 255, 2)

        cv2.imshow('ofrgb', frame)
        cv2.waitKey(int(1))
        print(video_folder_path, index)

print('wow')
