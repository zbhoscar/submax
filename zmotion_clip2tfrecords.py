import data_io.basetf as basetf
import tensorflow as tf
import os
import os.path as osp
import multiprocessing as mp
import numpy as np
import cv2
import data_io.basepy as basepy
import random

STEP = 8
CLIP_LENGTH = 16
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
_OPTICAL = 2

DATASET_PATH = '/absolute/datasets/anoma'
CLIPS_TFRECS_PATH = DATASET_PATH + '_motion%d_tfrecords' % CLIP_LENGTH
# CLIPS_TFRECS_PATH = CLIPS_TFRECS_PATH.replace('datasets', 'ext3t')

# cv2 format: c, r, w, h                    # -> start, -v start, -> length, -v length
TRACK_WINDOW = (70, 50, 50, 50)
# np  format: shape = (240, 320, 3)         # (h, w, channel)
AREA_CROPS = [(0, 150, 0, 190),
              (0, 150, 130, 320),
              (90, 240, 0, 190),
              (90, 240, 130, 320)]


def write_tfrecords(sample_path_list, tfrecords_path):
    for sample_path in sample_path_list:
        # eg. sample_path: '/absolute/datasets/anoma/Abuse/Abuse001_x264'
        # eg. sample_path: '/absolute/datasets/anoma/Normal/Normal_Videos308_3_x264'
        video_name = osp.basename(sample_path)
        class_name = osp.basename(osp.dirname(sample_path))

        tfrecord_name = '%s@%s' % (class_name, video_name)
        tfrecord_path = osp.join(tfrecords_path, tfrecord_name + '.tfr')
        txt_crop_path = osp.join(tfrecords_path, tfrecord_name + '.txt')
        imfloder_path = basepy.check_or_create_path(osp.join(tfrecords_path, tfrecord_name))

        # clean wrong .tfrecord file: exists but 0 bytes
        if osp.isfile(tfrecord_path) and osp.getsize(tfrecord_path) == 0:
            os.remove(tfrecord_path)
        # if tfrecord not exist, then write one
        if not osp.isfile(tfrecord_path):
            writer = tf.python_io.TFRecordWriter(tfrecord_path)

            frames_path = basepy.get_1tier_file_path_list(sample_path, suffix='.jpg')
            frame_list = sorted(frames_path, key=lambda x: int(osp.basename(x).split('.')[0]))

            for index in range(len(frame_list) - CLIP_LENGTH):
                if index % STEP == 0:
                    frames = [cv2.imread(frame_list[index + i]) for i in range(CLIP_LENGTH + 1)]
                    grays = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in frames]
                    flows = [cv2.calcOpticalFlowFarneback(grays[i * _OPTICAL], grays[(i + 1) * _OPTICAL],
                                                          None, 0.5, 3, 15, 3, 7, 1.5, 0)
                             for i in range(int(CLIP_LENGTH / _OPTICAL))]
                    motion = np.sum([np.linalg.norm(flow, ord=None, axis=2) for flow in flows], axis=0)

                    for nj, area in enumerate(AREA_CROPS):
                        motion_crop = motion[area[0]:area[1], area[2]:area[3]]
                        _, window_temp = cv2.meanShift(motion_crop, TRACK_WINDOW, CRITERIA)
                        c, r, w, h = window_temp
                        mobject = motion_crop[r:r + h, c:c + w]

                        mean_value = np.mean(mobject)
                        if mean_value > 0.07 * CLIP_LENGTH:
                            _ = basepy.write_txt_add_lines(txt_crop_path,
                                                           class_name, video_name, str(index), str(nj),
                                                           str(c + area[2]), str(r + area[0]), str(w), str(h),
                                                           str(mean_value)[:5])
                            frame_crops_path = [
                                osp.join(imfloder_path,
                                         tfrecord_name + '_' + str(index) + '_' + str(nj) + '_' + str(i) + '.jpg')
                                for i in range(CLIP_LENGTH)]

                            _ = [cv2.imwrite(frame_crops_path[i],
                                             frames[i][r + area[0]:r + h + area[0], c + area[2]:c + w + area[2]])
                                 for i in range(CLIP_LENGTH)]
                            #  frame = cv2.rectangle(frame,
                            #                       (c + area[2], r + area[0]),
                            #                       (c + area[2] + w, r + area[0] + h), 255, 2)

                            image_raw_array = [tf.gfile.FastGFile(i, 'rb').read() for i in frame_crops_path]

                            example = tf.train.Example(features=tf.train.Features(
                                feature={'class_name': basetf.bytes_feature(tf.compat.as_bytes(class_name)),
                                         'video_name': basetf.bytes_feature(tf.compat.as_bytes(video_name)),
                                         'frame_index': basetf.int64_feature(index),
                                         'crop_area': basetf.int64_feature(nj),
                                         'c': basetf.int64_feature(c),
                                         'r': basetf.int64_feature(r),
                                         'w': basetf.int64_feature(w),
                                         'h': basetf.int64_feature(h),
                                         'frame0': basetf.bytes_feature(image_raw_array[0]),
                                         'frame1': basetf.bytes_feature(image_raw_array[1]),
                                         'frame2': basetf.bytes_feature(image_raw_array[2]),
                                         'frame3': basetf.bytes_feature(image_raw_array[3]),
                                         'frame4': basetf.bytes_feature(image_raw_array[4]),
                                         'frame5': basetf.bytes_feature(image_raw_array[5]),
                                         'frame6': basetf.bytes_feature(image_raw_array[6]),
                                         'frame7': basetf.bytes_feature(image_raw_array[7]),
                                         'frame8': basetf.bytes_feature(image_raw_array[8]),
                                         'frame9': basetf.bytes_feature(image_raw_array[9]),
                                         'frame10': basetf.bytes_feature(image_raw_array[10]),
                                         'frame11': basetf.bytes_feature(image_raw_array[11]),
                                         'frame12': basetf.bytes_feature(image_raw_array[12]),
                                         'frame13': basetf.bytes_feature(image_raw_array[13]),
                                         'frame14': basetf.bytes_feature(image_raw_array[14]),
                                         'frame15': basetf.bytes_feature(image_raw_array[15]),
                                         }))
                            writer.write(example.SerializeToString())
                            del frame_crops_path, image_raw_array
                        del motion_crop, window_temp, c, r, w, h, mobject, mean_value
                    del frames, grays, flows, motion
            writer.close()
            del frames_path, frame_list
        del video_name, class_name, tfrecord_name, tfrecord_path, txt_crop_path, imfloder_path


def read_tfrecords(tfrecords_path_file_list, num_epochs=1, is_training=False, batch_size=64, preprocessing='standard'):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(tfrecords_path_file_list, num_epochs=num_epochs,
                                                    shuffle=is_training)
    # filename_queue = tf.train.string_input_producer(
    #    ['/home/zbh/Desktop/alpha_2_zbh/UCF101pic_tfrecords/v_ApplyEyeMakeup_g02_c04.tfrecords'])
    _, serialized_example = reader.read(filename_queue)
    ft = tf.parse_single_example(serialized_example,
                                 features={'class_name': tf.FixedLenFeature([], tf.string),
                                           'video_name': tf.FixedLenFeature([], tf.string),
                                           'frame_index': tf.FixedLenFeature([], tf.int64),
                                           'crop_area': tf.FixedLenFeature([], tf.int64),
                                           'c': tf.FixedLenFeature([], tf.int64),
                                           'r': tf.FixedLenFeature([], tf.int64),
                                           'w': tf.FixedLenFeature([], tf.int64),
                                           'h': tf.FixedLenFeature([], tf.int64),
                                           'frame0': tf.FixedLenFeature([], tf.string),
                                           'frame1': tf.FixedLenFeature([], tf.string),
                                           'frame2': tf.FixedLenFeature([], tf.string),
                                           'frame3': tf.FixedLenFeature([], tf.string),
                                           'frame4': tf.FixedLenFeature([], tf.string),
                                           'frame5': tf.FixedLenFeature([], tf.string),
                                           'frame6': tf.FixedLenFeature([], tf.string),
                                           'frame7': tf.FixedLenFeature([], tf.string),
                                           'frame8': tf.FixedLenFeature([], tf.string),
                                           'frame9': tf.FixedLenFeature([], tf.string),
                                           'frame10': tf.FixedLenFeature([], tf.string),
                                           'frame11': tf.FixedLenFeature([], tf.string),
                                           'frame12': tf.FixedLenFeature([], tf.string),
                                           'frame13': tf.FixedLenFeature([], tf.string),
                                           'frame14': tf.FixedLenFeature([], tf.string),
                                           'frame15': tf.FixedLenFeature([], tf.string),
                                           })
    if preprocessing is 'standard':
        # decode jpeg image
        np_mean = np.load('./crop_mean.npy')
        clip = []
        for i in range(CLIP_LENGTH):
            index = 'frame%d' % i
            frame = tf.image.decode_jpeg(ft[index])
            frame = basetf.aspect_preserving_resize(frame, 112)
            # frame = tf.image.convert_image_dtype(frame, dtype=tf.float32)
            frame = tf.to_float(frame)
            clip.append(frame)
        clip = tf.convert_to_tensor(basetf.central_crop(clip, 112, 112)) - tf.convert_to_tensor(np_mean)
    elif preprocessing is 'dynamic_image':
        cpt = CLIP_LENGTH
        xi_shu = [2 * (cpt - t + 1) - (cpt + 1) * (basepy.Ht(cpt) - basepy.Ht(t - 1)) for t in range(1, cpt + 1)]
        # FOR sample_to IN multi_sample_to_siamese:
        clip = []
        for i in range(cpt):
            index = 'frame%d' % i
            frame = tf.image.decode_jpeg(ft[index])
            frame.set_shape([None, None, 3])
            frame = tf.image.resize_images(frame, [130, 130])
            frame = tf.image.rgb_to_grayscale(frame)
            frame = tf.image.convert_image_dtype(frame, dtype=tf.float32) * xi_shu[i]
            clip.append(frame)
        clip = tf.convert_to_tensor(sum(clip))
    else:
        raise ValueError('Wrong preprocessing type: %s.' % preprocessing)

    # other feature
    features_list = ft['class_name'], ft['video_name'], ft[
        'frame_index'], ft['crop_area'], ft['c'], ft['r'], ft['w'], ft['h'], clip
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size

    class_batch, name_batch, index_batch, crop_batch, c_batch, r_batch, w_batch, h_batch, clip_batch = \
        tf.train.shuffle_batch(features_list,
                               batch_size=batch_size,
                               capacity=capacity,
                               num_threads=int(mp.cpu_count()),
                               min_after_dequeue=min_after_dequeue
                               ) if is_training else tf.train.batch(features_list,
                                                                    batch_size=batch_size,
                                                                    capacity=capacity,
                                                                    num_threads=int(mp.cpu_count()))

    return class_batch, name_batch, index_batch, crop_batch, c_batch, r_batch, w_batch, h_batch, clip_batch


def main():
    _ = basepy.check_or_create_path(CLIPS_TFRECS_PATH, create=True, show=True)

    # write tfrecords
    sample_path_list = basepy.get_2tier_folder_path_list(DATASET_PATH)
    random.shuffle(sample_path_list)
    # single processing
    # write_tfrecords(sample_path_list, CLIPS_TFRECS_PATH)
    basepy.non_output_multiprocessing(write_tfrecords, sample_path_list, CLIPS_TFRECS_PATH,
                                      num=int(mp.cpu_count()))
    print('writing done')

    # read tfrecords, examples:
    # tfrecords_path_file_list = basepy.get_1tier_file_list(clips_tfrecords_path)
    # a, b, c, d, e =read_tfrecords(tfrecords_path_file_list, num_epochs=1, is_training=False, batch_size=64)


if __name__ == '__main__':
    main()
