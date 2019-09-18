import os.path as osp
import os
import tensorflow as tf
import numpy as np
import time
from pprint import pprint

import data_io.basepy as basepy
import data_io.basetf as basetf
import zc3d_npy_base as base
import zdefault_dict


def main(_):
    timestamp = time.strftime("%y%m%d%H%M%S", time.localtime())
    tags = tf.flags
    # Net config
    tags.DEFINE_integer('batch_size', 64, 'batch size.')
    tags.DEFINE_integer('epoch_num', 202, 'epoch number.')
    tags.DEFINE_float('learning_rate_base', 0.0005, 'learning rate base')
    tags.DEFINE_float('moving_average_decay', 0.99, 'moving average decay')
    tags.DEFINE_float('regularization_scale', 0.00003, 'regularization scale')
    tags.DEFINE_string('fusion', 'standard', 'fusion ways in feature extraction')
    tags.DEFINE_string('npy_file_path',
                       '/absolute/datasets/anoma_motion_reformed_pyramid_120_85_4region_maxtop_1000_c3d_npy',
                       'npy file path')
    tags.DEFINE_string('testing_list',
                       '/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt',
                       'default to UCFCrime, else specific to UCSD.')
    tags.DEFINE_string('set_gpu', '0', 'Single gpu version, index select')
    # lasting
    tags.DEFINE_string('lasting', '', 'a TensorFlow model path for lasting')
    # every ? epochs to save
    tags.DEFINE_integer('saving_interval', 5, 'every ? epochs to save')
    F = tags.FLAGS

    SEGMENT_NUM = int(F.npy_file_path.split('_')[-3])
    SAVE_FILE_PATH = osp.join('/absolute/tensorflow_models', timestamp + '_' + osp.basename(F.npy_file_path),
                              timestamp + '.ckpt')
    JSON_FILE_PATH = osp.join(basepy.check_or_create_path(osp.dirname(SAVE_FILE_PATH), show=True), 'keys.json')
    D = basepy.DictCtrl(zdefault_dict.EXPERIMENT_KEYS).save2path(JSON_FILE_PATH,
                                                                 batch_size=F.batch_size,
                                                                 epoch_num=F.epoch_num,
                                                                 learning_rate_base=F.learning_rate_base,
                                                                 moving_average_decay=F.moving_average_decay,
                                                                 regularization_scale=F.regularization_scale,
                                                                 npy_file_path=F.npy_file_path,
                                                                 segment_num=SEGMENT_NUM,
                                                                 set_gpu=F.set_gpu,
                                                                 lambda1=0.00008,
                                                                 lambda2=0.00008,
                                                                 fusion=F.fusion,
                                                                 lasting=F.lasting,
                                                                 testing_list=F.testing_list,
                                                                 saving_interval=F.saving_interval,
                                                                 )

    _ = [print('Preparing training ...... D values:')] + [print('    ', i, ":", D[i]) for i in D]
    _ = network_train(D, SAVE_FILE_PATH)


def network_train(d, ckpt_file_path):
    # with tf.device('/cpu:0'):
    feature_path_list = basepy.get_1tier_file_path_list(d['npy_file_path'], suffix='.npy')
    # train_txt = '/absolute/datasets/Anomaly-Detection-Dataset/Anomaly_Train.txt'
    test_list = basepy.read_txt_lines2list(d['testing_list'], sep=' ')
    test_list = base.reform_train_list(test_list, feature_path_list, if_print=False)
    train_list = [i for i in feature_path_list if i not in test_list]
    print('TRAINING: %d training examples in all' % len(train_list))
    # UNDO feature_dict for memo free
    feature_dict = base.read_npy_file_path_list(train_list)

    # set saving for every saving_interval epochs
    anomaly_keys = [i for i in feature_dict.keys() if 'normal' not in i.lower()]
    normal_keys = [i for i in feature_dict.keys() if 'normal' in i.lower()]
    samples_in_one_epoch = min(len(anomaly_keys), len(normal_keys))
    step2show = [int(i * samples_in_one_epoch / d['batch_size'])
                 for i in range(d['epoch_num']) if i % int(d['saving_interval'] / 3) == 0]
    step2save = [int(i * samples_in_one_epoch / d['batch_size'])
                 for i in range(d['epoch_num']) if i % d['saving_interval'] == 0]

    # get list in all *epochs in one
    anomaly_list = basepy.repeat_list_for_epochs(anomaly_keys, epoch_num=d['epoch_num'], shuffle=True)
    normal_list = basepy.repeat_list_for_epochs(normal_keys, epoch_num=d['epoch_num'], shuffle=True)
    max_index = min(len(anomaly_list), len(normal_list))
    anomaly_list = anomaly_list[:max_index] + [False] * d['batch_size']
    normal_list = normal_list[:max_index] + [False] * d['batch_size']

    # NET SETTING
    with tf.name_scope('input'):
        input_anom = tf.placeholder(tf.float32, [d['batch_size'], d['segment_num'], d['feature_len']], name='anom')
        input_norm = tf.placeholder(tf.float32, [d['batch_size'], d['segment_num'], d['feature_len']], name='norm')

    with tf.name_scope('forward-propagation'):
        score_anomaly = base.network_fn(input_anom,
                                        fusion=d['fusion'],
                                        feature_len=d['feature_len'],
                                        segment_num=d['segment_num'])
        score_normal = base.network_fn(input_norm,
                                       fusion=d['fusion'],
                                       feature_len=d['feature_len'],
                                       segment_num=d['segment_num'])

    with tf.name_scope('loss'):
        mil_loss = tf.maximum(0., 1. - tf.reduce_max(score_anomaly, axis=1) + tf.reduce_max(score_normal, axis=1))
        regu = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(d['regularization_scale']), tf.trainable_variables())
        mean_mil = tf.reduce_mean(mil_loss)
        loss = mean_mil + regu

    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope('moving_average'):
        variable_averages = tf.train.ExponentialMovingAverage(d['moving_average_decay'], global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(learning_rate=d['learning_rate_base']
                                            ).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')

    print(('* ' + 'Variables to be trained' + ' *').center(60, '*'))
    pprint(tf.trainable_variables())
    print('Model .ckpt save path: %s' % ckpt_file_path)

    saver = tf.train.Saver(max_to_keep=100)

    init_op = tf.global_variables_initializer()
    os.environ["CUDA_VISIBLE_DEVICES"] = d['set_gpu']
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        print('program begins, timestamp %s' % time.asctime(time.localtime(time.time())))
        if d['lasting']:
            restore_ckpt = basetf.get_ckpt_path(d['lasting'])
            saver_goon = tf.train.Saver()
            saver_goon.restore(sess, restore_ckpt)

        step = 0
        while anomaly_list[step * d['batch_size'] + d['batch_size'] - 1]:
            time1 = time.time()
            batch_start, batch_end = step * d['batch_size'], step * d['batch_size'] + d['batch_size']

            anomaly_in = np.empty((d['batch_size'], d['segment_num'], d['feature_len']), dtype='float32')
            normal_in = np.empty((d['batch_size'], d['segment_num'], d['feature_len']), dtype='float32')
            for j, i in enumerate(anomaly_list[batch_start:batch_end]):
                anomaly_in[j] = base.reform_np_array(feature_dict[i], reform=d['segment_num'])
            for j, i in enumerate(normal_list[batch_start:batch_end]):
                normal_in[j] = base.reform_np_array(feature_dict[i], reform=d['segment_num'])

            time2 = time.time()
            # print(anomaly_in.shape)
            loss_, _, mean_mil_, regu_ = sess.run([loss, train_op, mean_mil, regu],
                                  feed_dict={input_anom: anomaly_in, input_norm: normal_in})
            if step in step2show:
                print('After %5d steps, loss = %.5e, mil = %.5e, regu = %.5e, '
                      'feed: %.3fsec, train: %.3fsec' %
                      (step, loss_, mean_mil_, regu_, time2 - time1, time.time() - time2))
            if step in step2save:
                print('Save tfrecords at step %5d / %4d epochs.'
                      % (step, d['saving_interval'] * step2save.index(step)))
                saver.save(sess, ckpt_file_path, global_step=global_step)
            step += 1

    print('Model .ckpt save path: %s' % ckpt_file_path)
    print('TRAINING Finish ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))
    return osp.dirname(ckpt_file_path)


def ones(keys, feature_dict, D):
    anoma_keys, norma_keys = keys
    anoma_in = [base.reform_np_array(feature_dict[key], reform=D['segment_num']) for key in anoma_keys]
    norma_in = [base.reform_np_array(feature_dict[key], reform=D['segment_num']) for key in norma_keys]
    return np.array(anoma_in), np.array(norma_in)


def network_train_memo_bk():
    #     # with tf.device('/cpu:0'):
    #     feature_path_list = basepy.get_1tier_file_path_list(D['npy_file_path'], suffix='.npy')
    #     # train_txt = '/absolute/datasets/Anomaly-Detection-Dataset/Anomaly_Train.txt'
    #     train_list = basepy.read_txt_lines2list(D['training_list'], sep=' ')
    #     train_list = base.reform_train_list(train_list, feature_path_list)
    #
    #     feature_dict = base.read_npy_file_path_list(train_list)
    #
    #     anomaly_keys = [i for i in feature_dict.keys() if 'normal' not in i.lower()]
    #     normal_keys = [i for i in feature_dict.keys() if 'normal' in i.lower()]
    #
    #     anomaly_list = basepy.repeat_list_for_epochs(anomaly_keys, epoch_num=D['epoch_num'], shuffle=True)
    #     normal_list = basepy.repeat_list_for_epochs(normal_keys, epoch_num=D['epoch_num'], shuffle=True)
    #
    #     samples_in_one_epoch = min(len(anomaly_keys), len(normal_keys))
    #     step2show = [int(i * samples_in_one_epoch / D['batch_size'])
    #                  for i in range(D['epoch_num']) if i % int(D['saving_interval'] / 3) == 0]
    #     step2save = [int(i * samples_in_one_epoch / D['batch_size'])
    #                  for i in range(D['epoch_num']) if i % D['saving_interval'] == 0]
    #
    #     with tf.name_scope('input'):
    #         input_anom = tf.placeholder(tf.float32, [D['batch_size'], D['segment_num'], D['feature_len']], name='anom')
    #         input_norm = tf.placeholder(tf.float32, [D['batch_size'], D['segment_num'], D['feature_len']], name='norm')
    #
    #     with tf.name_scope('forward-propagation'):
    #         score_anomaly = base.network_fn(input_anom,
    #                                         fusion=D['fusion'],
    #                                         feature_len=D['feature_len'],
    #                                         segment_num=D['segment_num'])
    #         score_normal = base.network_fn(input_norm,
    #                                        fusion=D['fusion'],
    #                                        feature_len=D['feature_len'],
    #                                        segment_num=D['segment_num'])
    #
    #     with tf.name_scope('loss'):
    #         restrict2 = tf.convert_to_tensor(
    #             [tf.reduce_mean(score_anomaly[i] ** 2) for i in range(D['batch_size'])],
    #             dtype=tf.float32)
    #         mil_loss = tf.maximum(0., 1. - tf.reduce_max(score_anomaly, axis=1) + tf.reduce_max(score_normal, axis=1))
    #         regu = tf.contrib.layers.apply_regularization(
    #             tf.contrib.layers.l2_regularizer(D['regularization_scale']), tf.trainable_variables())
    #         mean_mil = tf.reduce_mean(mil_loss)
    #         # l1, l2 = D['lambda1'] * tf.reduce_mean(restrict1), D['lambda2'] * tf.reduce_mean(restrict2)
    #         l2 = D['lambda2'] * tf.reduce_mean(restrict2)
    #         # if D['fusion'] == 'standard':
    #         #     loss = mean_mil + l2 + regu
    #         # elif D['fusion'] == 'segments' or D['fusion'] == 'average' or D['fusion'] == 'attention':
    #         #     loss = mean_mil + regu
    #         # else:
    #         #     raise ValueError('Wrong fusion type: %s' % D['fusion'])
    #         loss = mean_mil + regu
    #
    #     global_step = tf.Variable(0, trainable=False)
    #     with tf.name_scope('moving_average'):
    #         variable_averages = tf.train.ExponentialMovingAverage(D['moving_average_decay'], global_step)
    #         variable_averages_op = variable_averages.apply(tf.trainable_variables())
    #
    #     with tf.name_scope('train_step'):
    #         train_step = tf.train.AdamOptimizer(learning_rate=D['learning_rate_base']
    #                                             ).minimize(loss, global_step=global_step)
    #         with tf.control_dependencies([train_step, variable_averages_op]):
    #             train_op = tf.no_op(name='train')
    #
    #     print(('* ' + 'Variables to be trained' + ' *').center(60, '*'))
    #     pprint(tf.trainable_variables())
    #     print('Model .ckpt save path: %s' % SAVE_FILE_PATH)
    #
    #     saver = tf.train.Saver(max_to_keep=100)
    #
    #     init_op = tf.global_variables_initializer()
    #     os.environ["CUDA_VISIBLE_DEVICES"] = D['set_gpu']
    #     gpu_options = tf.GPUOptions(allow_growth=True)
    #     config = tf.ConfigProto(gpu_options=gpu_options)
    #     with tf.Session(config=config) as sess:
    #         sess.run(init_op)
    #         print('program begins, timestamp %s' % time.asctime(time.localtime(time.time())))
    #
    #         if D['lasting']:
    #             restore_ckpt = basetf.get_ckpt_path(D['lasting'])
    #             saver_goon = tf.train.Saver()
    #             saver_goon.restore(sess, restore_ckpt)
    #         try:
    #             max_index = min(len(anomaly_list), len(normal_list))
    #             while True:
    #                 step = sess.run(global_step)
    #                 time1 = time.time()
    #
    #                 batch_start, batch_end = step * D['batch_size'], step * D['batch_size'] + D['batch_size']
    #                 if batch_end > max_index:
    #                     raise ValueError
    #
    #                 anomaly_in = np.empty((D['batch_size'], D['segment_num'], D['feature_len']), dtype='float32')
    #                 normal_in = np.empty((D['batch_size'], D['segment_num'], D['feature_len']), dtype='float32')
    #                 for j, i in enumerate(anomaly_list[batch_start:batch_end]):
    #                     anomaly_in[j] = base.reform_np_array(feature_dict[i], reform=D['segment_num'])
    #                 for j, i in enumerate(normal_list[batch_start:batch_end]):
    #                     normal_in[j] = base.reform_np_array(feature_dict[i], reform=D['segment_num'])
    #
    #                 time2 = time.time()
    #                 # print(anomaly_in.shape)
    #                 l, _, a, c, d = sess.run([loss, train_op, mean_mil, l2, regu],
    #                                          feed_dict={input_anom: anomaly_in, input_norm: normal_in})
    #                 if step in step2show:
    #                     print('After %5d steps, loss = %.5e, mil = %.5e, l2 = %.5e, regu = %.5e, '
    #                           'feed: %.3fsec, train: %.3fsec' %
    #                           (step, l, a, c, d, time2 - time1, time.time() - time2))
    #                 if step in step2save:
    #                     print('Save tfrecords at step %5d / %4d epochs.'
    #                           % (step, D['saving_interval'] * step2save.index(step)))
    #                     saver.save(sess, SAVE_FILE_PATH, global_step=global_step)
    #         except ValueError:
    #             print('Model .ckpt save path: %s' % SAVE_FILE_PATH)
    #     print('------ Finish ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))
    pass


if __name__ == '__main__':
    tf.app.run()
