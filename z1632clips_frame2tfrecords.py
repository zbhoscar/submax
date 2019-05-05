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


def write_tfrecords(sample_path_list, tfrecords_path):
    for sample_path in sample_path_list:

        video_name = os.path.basename(sample_path)
        class_name = os.path.basename(os.path.dirname(sample_path))

        tfrecord_file_name = '%s@%s' % (class_name, video_name)
        tfrecord_file_path = os.path.join(tfrecords_path, tfrecord_file_name)

        # clean wrong .tfrecord file: exists but 0 bytes
        if os.path.isfile(tfrecord_file_path) and os.path.getsize(tfrecord_file_path) == 0:
            os.remove(tfrecord_file_path)
        # if tfrecord not exist, then write one
        if not os.path.isfile(tfrecord_file_path):

            writer = tf.python_io.TFRecordWriter(tfrecord_file_path)

            frame_list = basepy.sort_list_by_name(os.listdir(sample_path))
            segment_index = basepy.get_segment_start_index(len(frame_list), SEGMENT_NUMBER, CLIP_LENGTH)

            for i in range(SEGMENT_NUMBER):

                segment_length = segment_index[i + 1] - segment_index[i]
                overlap_start = basepy.get_overlap_start_index(segment_length, CLIP_LENGTH, overlap=0)

                for j, clip_start in enumerate(overlap_start):

                    image_raw_array = []
                    for k in range(CLIP_LENGTH):
                        frame_index_in_list = segment_index[i] + clip_start + k
                        frame_file_name = frame_list[frame_index_in_list]
                        frame_file_path = os.path.join(sample_path, frame_file_name)

                        image_raw = tf.gfile.FastGFile(frame_file_path, 'rb').read()
                        image_raw_array.append(image_raw)

                    example = tf.train.Example(features=tf.train.Features(
                        feature={'class_name': basetf._bytes_feature(tf.compat.as_bytes(class_name)),
                                 'video_name': basetf._bytes_feature(tf.compat.as_bytes(video_name)),
                                 'clip_index': basetf._int64_feature(j),
                                 'segment_index': basetf._int64_feature(i),
                                 'frame0': basetf._bytes_feature(image_raw_array[0]),
                                 'frame1': basetf._bytes_feature(image_raw_array[1]),
                                 'frame2': basetf._bytes_feature(image_raw_array[2]),
                                 'frame3': basetf._bytes_feature(image_raw_array[3]),
                                 'frame4': basetf._bytes_feature(image_raw_array[4]),
                                 'frame5': basetf._bytes_feature(image_raw_array[5]),
                                 'frame6': basetf._bytes_feature(image_raw_array[6]),
                                 'frame7': basetf._bytes_feature(image_raw_array[7]),
                                 'frame8': basetf._bytes_feature(image_raw_array[8]),
                                 'frame9': basetf._bytes_feature(image_raw_array[9]),
                                 'frame10': basetf._bytes_feature(image_raw_array[10]),
                                 'frame11': basetf._bytes_feature(image_raw_array[11]),
                                 'frame12': basetf._bytes_feature(image_raw_array[12]),
                                 'frame13': basetf._bytes_feature(image_raw_array[13]),
                                 'frame14': basetf._bytes_feature(image_raw_array[14]),
                                 'frame15': basetf._bytes_feature(image_raw_array[15]),
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
                                           'clip_index': tf.FixedLenFeature([], tf.int64),
                                           'segment_index': tf.FixedLenFeature([], tf.int64),
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
            frame = basetf._aspect_preserving_resize(frame, 112)
            # frame = tf.image.convert_image_dtype(frame, dtype=tf.float32)
            frame = tf.to_float(frame)
            clip.append(frame)
        clip = tf.convert_to_tensor(basetf._central_crop(clip, 112, 112)) - tf.convert_to_tensor(np_mean)
    elif preprocessing is 'dynamic_image':
        T = CLIP_LENGTH
        xi_shu = [2 * (T - t + 1) - (T + 1) * (basepy.Ht(T) - basepy.Ht(t - 1)) for t in range(1, T + 1)]
        # FOR sample_to IN multi_sample_to_siamese:
        clip = []
        for i in range(T):
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
    features_list = ft['class_name'], ft['video_name'], ft['clip_index'], ft['segment_index'], clip
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size

    video_class_batch, video_name_batch, video_label_batch, frame_index_batch, image_data_batch = \
        tf.train.shuffle_batch(features_list,
                               batch_size=batch_size,
                               capacity=capacity,
                               num_threads=int(mp.cpu_count()),
                               min_after_dequeue=min_after_dequeue
                               ) if is_training else tf.train.batch(features_list,
                                                                    batch_size=batch_size,
                                                                    capacity=capacity,
                                                                    num_threads=int(mp.cpu_count()))

    return video_class_batch, video_name_batch, video_label_batch, frame_index_batch, image_data_batch


def main():
    _ = basepy.check_or_create_path(CLIPS_TFRECS_PATH, create=True, show=True)

    # write tfrecords
    sample_path_list = basepy.get_2tier_folder_path_list(DATASET_PATH)
    # single processing
    # write_tfrecords(sample_path_list, clips_tfrecords_path)
    basepy.non_output_multiprocessing(write_tfrecords, sample_path_list, CLIPS_TFRECS_PATH,
                                      num=int(mp.cpu_count()))
    print('writing done')

    # read tfrecords, examples:
    # tfrecords_path_file_list = basepy.get_1tier_file_list(clips_tfrecords_path)
    # a, b, c, d, e =read_tfrecords(tfrecords_path_file_list, num_epochs=1, is_training=False, batch_size=64)


if __name__ == '__main__':
    main()
