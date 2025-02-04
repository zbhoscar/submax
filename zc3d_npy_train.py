import os.path as osp
import os
import tensorflow as tf
import numpy as np
import time
from pprint import pprint
import random

import data_io.basepy as basepy
import data_io.basetf as basetf
import zc3d_npy_base as base
import zdefault_dict


def main(_):
    tags = tf.flags
    # Net config
    tags.DEFINE_integer('batch_size', 64, 'batch size.')
    tags.DEFINE_integer('epoch_num', 1202, 'epoch number.')
    tags.DEFINE_float('learning_rate_base', 0.001, 'learning rate base')
    tags.DEFINE_float('moving_average_decay', 0.99, 'moving average decay')
    tags.DEFINE_float('regularization_scale', 0.00003, 'regularization scale')
    tags.DEFINE_string('fusion', 'standard', 'fusion ways in feature extraction')
    tags.DEFINE_string('npy_file_path',
                       '/absolute/datasets/anoma_motion_reformed_pyramid_120_85_4region_maxtop_500_c3d_npy',
                       'npy file path')
    tags.DEFINE_string('testing_list',
                       '/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt',
                       'default to UCFCrime, else specific to UCSD.')
    tags.DEFINE_string('set_gpu', '0', 'Single gpu version, index select')
    # lasting
    tags.DEFINE_string('lasting', '', 'a TensorFlow model path for lasting')
    # every ? epochs to save
    tags.DEFINE_integer('saving_interval', 10, 'every ? epochs to save')
    F = tags.FLAGS

    _ = network_train(F, F.npy_file_path)
    # _ = network_train(D, SAVE_FILE_PATH)


def list2np_array(npy_list, reform_num=1000):
    return np.array([base.reform_np_array(np.load(npy), reform=reform_num) for npy in npy_list])


def decode_flags(F, npy_reformed_file_path):
    # set np height from 'anoma_motion_reformed_pyramid_120_85_4region_maxtop_1000_c3d_npy'
    segment_num = int(npy_reformed_file_path.split('_')[-3])
    segment_num = segment_num * 4 if '4region' in npy_reformed_file_path else segment_num

    timestamp = time.strftime("%y%m%d%H%M%S", time.localtime())
    save_file_path = osp.join('/absolute/tensorflow_models', timestamp + '_' + osp.basename(npy_reformed_file_path),
                              timestamp + '.ckpt')

    json_file_path = osp.join(basepy.check_or_create_path(osp.dirname(save_file_path), show=True), 'keys.json')

    D = basepy.DictCtrl(zdefault_dict.EXPERIMENT_KEYS).save2path(json_path=json_file_path,
                                                                 batch_size=F.batch_size,
                                                                 epoch_num=F.epoch_num,
                                                                 learning_rate_base=F.learning_rate_base,
                                                                 moving_average_decay=F.moving_average_decay,
                                                                 regularization_scale=F.regularization_scale,
                                                                 npy_file_path=npy_reformed_file_path,
                                                                 segment_num=segment_num,
                                                                 set_gpu=F.set_gpu,
                                                                 lambda1=0.00008,
                                                                 lambda2=0.00008,
                                                                 fusion=F.fusion,
                                                                 lasting=F.lasting,
                                                                 testing_list=F.testing_list,
                                                                 saving_interval=F.saving_interval)
    _ = [print('Preparing training ...... D values:')] + [print('    ', i, ":", D[i]) for i in D]
    return D, save_file_path


def network_train(tf_flags, npy_reformed_file_path, top_k=20):
    # decode flags
    d, ckpt_file_path = decode_flags(tf_flags, npy_reformed_file_path)
    # make example list
    feature_path_list = basepy.get_1tier_file_path_list(d['npy_file_path'], suffix='.npy')
    test_list = basepy.read_txt_lines2list(d['testing_list'], sep=' ')
    test_list = base.reform_train_list(test_list, feature_path_list, if_print=False)
    train_list = [i for i in feature_path_list if i not in test_list]
    print('TRAINING: %d training examples in all' % len(train_list))
    anomaly_npy_list = [i for i in train_list if 'normal' not in i.lower()]
    normal_npy_list = [i for i in train_list if 'normal' in i.lower()]

    anomaly_npy_reformed = list2np_array(anomaly_npy_list, reform_num=d['segment_num'])
    normal_npy_reformed = list2np_array(normal_npy_list, reform_num=d['segment_num'])

    # set saving for every saving_interval epochs
    samples_in_one_epoch = min(len(anomaly_npy_list), len(normal_npy_list))
    step2show = [int(i * samples_in_one_epoch / d['batch_size'])
                 for i in range(d['epoch_num']) if i % int(d['saving_interval'] / 3) == 0]
    step2save = [int(i * samples_in_one_epoch / d['batch_size'])
                 for i in range(d['epoch_num']) if i % d['saving_interval'] == 0]
    step_in_all = list(range(step2save[-1] + 1)) + [False]

    # NET SETTING
    with tf.name_scope('input'):
        input_anom = tf.placeholder(tf.float32, [d['batch_size'], d['segment_num'], d['feature_len']], name='anom')
        input_norm = tf.placeholder(tf.float32, [d['batch_size'], d['segment_num'], d['feature_len']], name='norm')

    with tf.name_scope('forward-propagation'):
        score_anomaly = base.network_fn(input_anom,
                                        fusion=d['fusion'], feature_len=d['feature_len'], segment_num=d['segment_num'])
        score_normal = base.network_fn(input_norm,
                                       fusion=d['fusion'], feature_len=d['feature_len'], segment_num=d['segment_num'])

    with tf.name_scope('loss'):
        top_max_anomaly, _ = tf.nn.top_k(score_anomaly, top_k)
        mil_loss = tf.maximum(0., 1 - tf.reduce_min(top_max_anomaly, axis=1) + tf.reduce_max(score_normal, axis=1))
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
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
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
        while step_in_all[step] is not False:
            time1 = time.time()
            anomaly_in = anomaly_npy_reformed[np.random.choice(anomaly_npy_reformed.shape[0], d['batch_size'])]
            normal_in = normal_npy_reformed[np.random.choice(normal_npy_reformed.shape[0], d['batch_size'])]

            time2 = time.time()
            loss_, _, mean_mil_, regu_ = sess.run([loss, train_op, mean_mil, regu],
                                                  feed_dict={input_anom: anomaly_in, input_norm: normal_in})
            if step in step2show:
                print('After %5d steps, loss = %.5e, mil = %.5e, regu = %.5e, feed: %.3fsec, train: %.3fsec' %
                      (step, loss_, mean_mil_, regu_, time2 - time1, time.time() - time2))
            if step in step2save:
                print('Save tfrecords at step %5d / %4d epochs.'
                      % (step, d['saving_interval'] * step2save.index(step)))
                saver.save(sess, ckpt_file_path, global_step=global_step)
            step += 1

    print('Model .ckpt save path: %s' % ckpt_file_path)
    print('TRAINING Finish ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))
    return osp.dirname(ckpt_file_path)


if __name__ == '__main__':
    tf.app.run()
