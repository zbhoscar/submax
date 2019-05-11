import data_io.basepy as basepy
import data_io.basetf as basetf
import tensorflow as tf
import os
import multiprocessing as mp
import numpy as np

SEGMENT_NUMBER = 32
CLIP_LENGTH = 16
DATASET_PATH = '/absolute/datasets/anoma'
CLIPS_TFRECS_PATH = DATASET_PATH + '_%d%d_tfrecords' % (CLIP_LENGTH, SEGMENT_NUMBER)

frame_list = [i for i in range(481)]
segment_index = [0] + [round((i + 1) * len(frame_list) / SEGMENT_NUMBER) for i in range(SEGMENT_NUMBER)]



for i in range(SEGMENT_NUMBER):

    segment_length = segment_index[i + 1] - segment_index[i]
    # divide a segment to some fixed-length clips, get the LIST of start INDEX of each clip, with minimum overlap
    overlap_start = basepy.get_overlap_start_index(segment_length, CLIP_LENGTH, overlap=0)

    for j, clip_start in enumerate(overlap_start):

        image_raw_array = []
        for k in range(CLIP_LENGTH):  # '1.jpg'
            frame_index_in_list = segment_index[i] + clip_start + k
            frame_file_name = frame_list[frame_index_in_list]
            print(frame_file_name)
            # frame_file_path = os.path.join(sample_path, frame_file_name)
            #
            # image_raw = tf.gfile.FastGFile(frame_file_path, 'rb').read()
            # image_raw_array.append(image_raw)
