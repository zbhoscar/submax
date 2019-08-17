# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import data_io.basepy as basepy
import json
import time
import matplotlib.pyplot as plt
import os.path as osp
from pprint import pprint

import cv2

VIDEO_FRAME_PATH, FRAME_SUFFIX, STEP = (('/absolute/datasets/anoma/Burglary/Burglary005_x264', '.jpg', 10),
                                        ('/absolute/datasets/UCSDped2_reform/Anomaly/Anomaly009', '.tif', 2))[1]


from pyheatmap.heatmap import HeatMap



def main():
    video_name = osp.basename(VIDEO_FRAME_PATH)

    frames_path = basepy.get_1tier_file_path_list(VIDEO_FRAME_PATH, suffix=FRAME_SUFFIX)
    frame_list = sorted(frames_path, key=lambda x: int(osp.basename(x).split('.')[0]))

    # for j,frame_file in enumerate(frame_list[:-(STEP-1)]):
    for j, frame_file in enumerate(frame_list[108:118]):

        frames = cv2.imread(frame_file)
        frames_next = cv2.imread(frame_list[j+1])
        grays = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        grays_next = cv2.cvtColor(frames_next, cv2.COLOR_BGR2GRAY)
        flows = cv2.calcOpticalFlowFarneback(grays, grays_next, None, 0.5, 3, 15, 3, 7, 1.5, 0)
        # flows = cv2.calcOpticalFlowFarneback(grays, grays_next, None, 0.5, 3, 15, 3, 7, 1.5, 0)

        motion = np.linalg.norm(flows, ord=None, axis=2)
        H, W = motion.shape

        frame_name = osp.basename(frame_file).split('.')[0]
        x_txt_name = '%s_x.txt' % frame_name
        y_txt_name = '%s_y.txt' % frame_name
        z_txt_name = '%s_z.txt' % frame_name

        # # list_x, list_y, list_z = [], [], []
        # for height in range(H):
        #     for weight in range(W):
        #         basepy.write_txt_add_lines(x_txt_name, str(height))
        #         basepy.write_txt_add_lines(y_txt_name, str(weight))
        #         basepy.write_txt_add_lines(z_txt_name, str(motion[height][weight]))

        flow_uv = flows
        # Load normalized flow image of shape [H,W,2]
        # flow_uv = np.load('./data/flow_example_data.npy')

        # Apply the coloring (for OpenCV, set convert_to_bgr=True)
        flow_color = flow_to_color(flow_uv, convert_to_bgr=False)

        # hm = HeatMap(motion)
        # hm.clickmap(save_as="hit.png")
        # hm.heatmap(save_as="heat.png")

        # Display the image
        plt.imshow(flow_color)
        plt.show()



    print('------ Finish ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))


def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)


if __name__ == '__main__':
    main()
