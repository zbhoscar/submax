import os.path as osp
import os
import tensorflow as tf

import oz_flags
import data_io.basepy as baseio
import multiprocessing as mp
import preprocessing.vgg_preprocessing as preprocessing

F = oz_flags.tags.FLAGS
STR = 'sf_UCF-101.hm.20_task_siam.6.rec_snum_6_cstr_0.0_csec_1.0_ecod_dynamic_spsi_randomcrop_spty_backwards'

class HighMotionTFR(object):
    def __init__(self, data_root='/absolute', inputdata=STR, epoch_num=200, is_training=True, batch_size=64):
        self.tfrecords_dpath = osp.join(data_root, oz_flags.StrAna(inputdata).source_folder())
        self.source_dpath = osp.join(data_root, oz_flags.StrAna(inputdata).source_folder().split('.')[0])
        self.epoch_num = epoch_num
        self.is_training = is_training
        self.batch_size = batch_size

    def make_tfrecords_list(self, sample_list, ext='.tfrecord'):
        """ sample_list is orig sample_list from split file
            eg. sample_list[i] = ['ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01', '1']
            to make the tfrecords folder flat, replace '/' to '@'
        :return
            list[i] = '/absolute/datasets/UCF101frames_tfrecords/YoYo@v_YoYo_g08_c01.tfrecord'
        """
        baseio.check_or_create_path(self.tfrecords_dpath, create=True, show=False)
        return [osp.join(self.tfrecords_dpath, i[0].replace('/', '@') + ext) for i in sample_list]

    def check_and_create_tfrecords(self, tfrecords_list, class_list, subsegment_len = 20):
        """
        :param tfrecords_list:
            tfrecords_list[i] = '/absolute/datasets/UCF101frames_tfrecords/YoYo@v_YoYo_g08_c01.tfrecord'
            v_ApplyEyeMakeup_g08_c01 is a folder name that decodes the .avi into frames.jpg in source_folder,
            now write the frames in v_ApplyEyeMakeup_g08_c01 into the .tfrecord
        :param class_list:
            class_list[i]  = ['1', ApplyEyeMakeup]
        :return:
            No return, just check existing and create
        """
        for tfrecord in tfrecords_list:
            # clean wrong .tfrecord file: exists but 0 bytes
            if osp.isfile(tfrecord) and osp.getsize(tfrecord) == 0:
                os.remove(tfrecord)
            # if tfrecord not exist, then write one
            if not osp.isfile(tfrecord):
                source_cor_dpath = osp.join(self.source_dpath, osp.basename(tfrecord).replace('@', '/').split('.')[0])
                video_name = osp.basename(source_cor_dpath)
                class_name = osp.basename(osp.dirname(source_cor_dpath))
                # make the label 1~101 to 0~100, [['1', 'ApplyEyeMakeup'], ['2', 'ApplyLipstick']] > 0
                video_label = [x[1] for x in class_list].index(class_name)

                video_fpath = osp.join(self.source_dpath, class_name, video_name + '.avi')

                # get natural order of frames_names in video_frame_path, eg. ['1.jpg','2.jpg',...,'198.jpg']
                order = sorted(os.listdir(source_cor_dpath), key=lambda x: int(x.split('.')[0]))
                # write whole frames in video_frame_path in one .tfrecord
                writer = tf.python_io.TFRecordWriter(tfrecord)
                for j in order:  # '1.jpg'
                    frame_fpath = osp.join(source_cor_dpath, j)
                    frame_index = int(osp.splitext(j)[0])
                    image_raw = tf.gfile.FastGFile(frame_fpath, 'rb').read()
                    example = tf.train.Example(features=tf.train.Features(
                        feature={'class_name': baseio._bytes_feature(tf.compat.as_bytes(class_name)),
                                 'video_name': baseio._bytes_feature(tf.compat.as_bytes(video_name)),
                                 'video_label': baseio._int64_feature(video_label),
                                 'frame_index': baseio._int64_feature(frame_index),
                                 'image_raw': baseio._bytes_feature(image_raw),
                                 }))
                    writer.write(example.SerializeToString())
                writer.close()
                print('%s has just been written.' % tfrecord)
            # Check needed, in case of bad data
            if osp.getsize(tfrecord) == 0:
                exit('\nZBH: something wrong at %s' % tfrecord)

    def get_tfrecords_input(self, tfrecords_list, img_size):
        reader = tf.TFRecordReader()
        filename_queue = tf.train.string_input_producer(tfrecords_list, num_epochs=self.epoch_num,
                                                        shuffle=self.is_training)
        # filename_queue = tf.train.string_input_producer(
        #    ['/home/zbh/Desktop/alpha_2_zbh/UCF101pic_tfrecords/v_ApplyEyeMakeup_g02_c04.tfrecords'])
        _, serialized_example = reader.read(filename_queue)
        ft = tf.parse_single_example(serialized_example,
                                     features={'class_name': tf.FixedLenFeature([], tf.string),
                                               'video_name': tf.FixedLenFeature([], tf.string),
                                               'video_label': tf.FixedLenFeature([], tf.int64),
                                               'frame_index': tf.FixedLenFeature([], tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string)})
        # decode jpeg image
        image = tf.image.decode_jpeg(ft['image_raw'])
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = preprocessing.preprocess_image(image, img_size[0], img_size[1], is_training=self.is_training)
        # image = baseio.simple_preprocessing(image, img_size=img_size, shuffle=self.shuffle)

        # other feature
        features_list = ft['class_name'], ft['video_name'], ft['video_label'], ft['frame_index'], image
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * self.batch_size

        video_class_batch, video_name_batch, video_label_batch, frame_index_batch, image_data_batch = \
            tf.train.shuffle_batch(features_list,
                                   batch_size=self.batch_size,
                                   capacity=capacity,
                                   num_threads=int(mp.cpu_count()),
                                   min_after_dequeue=min_after_dequeue
                                   ) if self.is_training else tf.train.batch(features_list,
                                                                             batch_size=self.batch_size,
                                                                             capacity=capacity,
                                                                             num_threads=int(mp.cpu_count()))

        return video_class_batch, video_name_batch, video_label_batch, frame_index_batch, image_data_batch
