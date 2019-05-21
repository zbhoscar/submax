import data_io.basetf as basetf
import tensorflow as tf
import os
import os.path as osp
import multiprocessing as mp
import numpy as np
import cv2
import data_io.basepy as basepy
import random
import json
import time

CLIP_LEN, REDUCE_NUM, REDUCE_METHOD = [[16, 1001, 'simple']][0]

DATASET_PATH = '/absolute/datasets/anoma'
JSON_FILE_PATH = '/absolute/datasets/anoma_motion_all_json_type_1'
CLIPS_TFRECS_PATH = '../temp/test_cmp'


def write_tfrecords(json_path_list, tfrecords_path=CLIPS_TFRECS_PATH, dataset_path=DATASET_PATH,
                    reduce_method=REDUCE_METHOD, reduce_num=REDUCE_NUM, clip_len=CLIP_LEN):
    for json_path in json_path_list:
        # eg. json_path: '/absolute/datasets/anoma_motion_all_json_type_1/normal_train@Normal_Videos308_3_x264.json'
        # get original list form json:
        with open(json_path, 'r') as f:
            clips_info = json.load(f)
        # how to reduce clip json
        if reduce_method == 'simple':
            clips_info = [
                i for i in clips_info if i[2] % clip_len == 0] if len(clips_info) > reduce_num * 3 else clips_info
            clips_info = sorted(clips_info, key=lambda x: x[-1], reverse=True)[:reduce_num]
        else:
            raise ValueError('Wrong reduce method: %s' % reduce_method)

        tfrecord_name = str(osp.basename(json_path).split('.')[0])
        class_name, video_name = tfrecord_name.split('@')

        tfrecord_path = osp.join(tfrecords_path, tfrecord_name + '.tfr')
        motioncv_path = basepy.check_or_create_path(osp.join(tfrecords_path, tfrecord_name))

        original_video_path = osp.join(dataset_path, class_name, video_name)
        frames_path = basepy.get_1tier_file_path_list(original_video_path, suffix='.jpg')
        frame_list = sorted(frames_path, key=lambda x: int(osp.basename(x).split('.')[0]))

        # clean wrong .tfrecord file: exists but 0 bytes
        if osp.isfile(tfrecord_path) and osp.getsize(tfrecord_path) == 0:
            os.remove(tfrecord_path)
        # if tfrecord not exist, then write one
        if not osp.isfile(tfrecord_path):
            writer = tf.python_io.TFRecordWriter(tfrecord_path)
            for _, _, index, nj, c, r, w, h, _ in clips_info:
                frames = [cv2.imread(frame_list[index + i])[r:r + h, c:c + w] for i in range(clip_len)]
                frame_crops_path = [
                    osp.join(motioncv_path,
                             tfrecord_name + '_' + str(index) + '_' + str(nj) + '_' + str(i) + '.jpg')
                    for i in range(clip_len)]

                _ = [cv2.imwrite(frame_crops_path[i], frames[i]) for i in range(clip_len)]

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
            writer.close()


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
        for i in range(CLIP_LEN):
            index = 'frame%d' % i
            frame = tf.image.decode_jpeg(ft[index])
            frame = basetf.aspect_preserving_resize(frame, 112)
            # frame = tf.image.convert_image_dtype(frame, dtype=tf.float32)
            frame = tf.to_float(frame)
            clip.append(frame)
        clip = tf.convert_to_tensor(basetf.central_crop(clip, 112, 112)) - tf.convert_to_tensor(np_mean)
    elif preprocessing is 'dynamic_image':
        cpt = CLIP_LEN
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

    class_batch, name_batch, index_batch, nj_batch, c_batch, r_batch, w_batch, h_batch, data_batch = \
        tf.train.shuffle_batch(features_list,
                               batch_size=batch_size,
                               capacity=capacity,
                               num_threads=int(mp.cpu_count()),
                               min_after_dequeue=min_after_dequeue
                               ) if is_training else tf.train.batch(features_list,
                                                                    batch_size=batch_size,
                                                                    capacity=capacity,
                                                                    num_threads=int(mp.cpu_count()))

    return class_batch, name_batch, index_batch, nj_batch, c_batch, r_batch, w_batch, h_batch, data_batch


def direct_json_to_np(json_path_list, tfrecords_path=CLIPS_TFRECS_PATH, dataset_path=DATASET_PATH,
                    reduce_method=REDUCE_METHOD, reduce_num=REDUCE_NUM, clip_len=CLIP_LEN):
    out_put = []
    for json_path in json_path_list:
        # eg. json_path: '/absolute/datasets/anoma_motion_all_json_type_1/normal_train@Normal_Videos308_3_x264.json'
        # get original list form json:
        with open(json_path, 'r') as f:
            clips_info = json.load(f)
        # how to reduce clip json
        if reduce_method == 'simple':
            clips_info = [
                i for i in clips_info if i[2] % clip_len == 0] if len(clips_info) > reduce_num * 3 else clips_info
            clips_info = sorted(clips_info, key=lambda x: x[-1], reverse=True)[:reduce_num]
        else:
            raise ValueError('Wrong reduce method: %s' % reduce_method)

        tfrecord_name = str(osp.basename(json_path).split('.')[0])
        class_name, video_name = tfrecord_name.split('@')


        original_video_path = osp.join(dataset_path, class_name, video_name)
        frames_path = basepy.get_1tier_file_path_list(original_video_path, suffix='.jpg')
        frame_list = sorted(frames_path, key=lambda x: int(osp.basename(x).split('.')[0]))

        # for _, _, index, nj, c, r, w, h, _ in clips_info:
        #     frames = [cv2.imread(frame_list[index + i])[r:r + h, c:c + w] for i in range(clip_len)]
        one = [np.array([cv2.imread(frame_list[index + i])[r:r + h, c:c + w] for i in range(clip_len)],dtype='float32')
               for _, _, index, nj, c, r, w, h, _ in clips_info]
        out_put.append(one)
    return out_put


def main():
    _ = basepy.check_or_create_path(CLIPS_TFRECS_PATH, create=True, show=True)

    # write tfrecords
    sample_path_list = basepy.get_1tier_file_path_list(JSON_FILE_PATH, suffix='.json')
    random.shuffle(sample_path_list)
    test_list = sample_path_list[0:5]
    t1 = time.time()
    # single processing
    write_tfrecords(test_list, CLIPS_TFRECS_PATH, DATASET_PATH, REDUCE_METHOD, REDUCE_NUM, CLIP_LEN)
    t2=time.time()
    # basepy.non_output_multiprocessing(write_tfrecords, sample_path_list[1:2], CLIPS_TFRECS_PATH, DATASET_PATH,
    #                                   REDUCE_METHOD, REDUCE_NUM, CLIP_LEN, num=int(mp.cpu_count()))

    see = direct_json_to_np(test_list, CLIPS_TFRECS_PATH, DATASET_PATH, REDUCE_METHOD, REDUCE_NUM, CLIP_LEN)
    t3 = time.time()

    print('%f, %f' % (t2-t1, t3-t2))

    print('------ Finish ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))


if __name__ == '__main__':
    main()
