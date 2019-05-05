import data_io.basepy as basepy
import os.path as osp
import cv2
import numpy as np

# def main():
video_folder_all = basepy.get_2tier_folder_path_list('/absolute/datasets/anoma')

video_folder_path = video_folder_all[np.random.randint(0, len(video_folder_all))]
# video_folder_path = '/absolute/datasets/anoma/Burglary/Burglary096_x264'

frames_path = basepy.get_1tier_file_path_list(video_folder_path, suffix='.jpg')
frames_path = sorted(frames_path, key=lambda x: int(osp.basename(x).split('.')[0]))

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
clip_len = 16
_optical = 2

# cv2 format: c, r, w, h                    # -> start, -v start, -> length, -v length
track_window = (70, 50, 50, 50)
# np  format: shape = (240, 320, 3)         # (h, w, channel)
area_crop = [(0, 150, 0, 190),
             (0, 150, 130, 320),
             (90, 240, 0, 190),
             (90, 240, 130, 320)]

for index in range(len(frames_path) - _optical - clip_len):
    frame = cv2.imread(frames_path[index])

    if index % clip_len != -1:
        motion = np.zeros_like(frame[:,:,0])
        for i in range(8):
            prvsframe = cv2.imread(frames_path[index + _optical * i])
            prvsgray = cv2.cvtColor(prvsframe, cv2.COLOR_BGR2GRAY)
            nextframe = cv2.imread(frames_path[index + _optical * (i +1)])
            nextgray = cv2.cvtColor(nextframe, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvsgray, nextgray, None, 0.5, 3, 15, 3, 7, 1.5, 0)
            motion = motion + np.linalg.norm(flow, ord=None, axis=2)

        for j, i in enumerate(area_crop):
            motion_crop = motion[i[0]:i[1], i[2]:i[3]]
            retval, window_temp = cv2.meanShift(motion_crop, track_window, criteria)

            c, r, w, h = window_temp
            mobject = motion_crop[r:r + h, c:c + w]
            hist, bins = np.histogram(mobject, bins=[0, 0.1, 1, 2, 5, 10, 100])
            print(window_temp, hist)

            if np.mean(mobject) > 0.1 * clip_len:
                frame = cv2.rectangle(frame, (c + i[2], r + i[0]), (c + i[2] + w, r + i[0] + h), 255, 2)
            # frame = cv2.rectangle(frame, (160, 120), (320, 240), 255, 2)

    #
    # for j, (c,r,w,h) in enumerate(windows):
    #     frame = cv2.rectangle(frame, (c + area_crop[j][2], r + area_crop[j][0]),
    #                           (c + area_crop[j][2] + w, r + area_crop[j][0] + h), 255, 2)

    cv2.imshow('ofrgb', frame)
    cv2.waitKey(int(1))
    print(video_folder_path, index)

print('wow')
