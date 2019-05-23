import tensorflow as tf
import data_io.basepy as basepy
import random
import numpy as np
import os.path as osp
import copy

slim = tf.contrib.slim


def arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc


def regression(inputs,
               is_training=True,
               dropout_keep_prob=0.85,
               scope='regression',
               fc_conv_padding='VALID'):
    with tf.variable_scope(scope, 'regression', [inputs], reuse=tf.AUTO_REUSE) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            # Use conv2d instead of fully_connected layers.
            regress = tf.expand_dims(tf.expand_dims(inputs, 1), 1, name='input/expand')
            regress = slim.dropout(regress, dropout_keep_prob, is_training=is_training, scope='dropout0')

            regress = slim.conv2d(regress, 512, [1, 1], padding=fc_conv_padding, scope='regress1',
                                  activation_fn=tf.nn.relu)
            regress = slim.dropout(regress, dropout_keep_prob, is_training=is_training, scope='dropout1')

            regress = slim.conv2d(regress, 32, [1, 1], scope='regress2',
                                  activation_fn=None)
            regress = slim.dropout(regress, dropout_keep_prob, is_training=is_training, scope='dropout2')

            # # Convert end_points_collection into a end_point dict.
            # end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            regress = slim.conv2d(regress, 1, [1, 1], scope='score/expand',
                                  activation_fn=tf.nn.sigmoid)
            regress = tf.squeeze(regress, [1, 2], name='score')

            return regress


def network_fn(inputs, fusion='standard', feature_len=4096, segment_num=32, attention_l=1024, **kwargs):
    with slim.arg_scope(arg_scope(weight_decay=0.0005)):
        if fusion == 'standard':
            reshaped_inputs = tf.reshape(inputs, [-1, feature_len])
            anomaly_score = regression(reshaped_inputs, **kwargs)
            outputs = tf.reshape(anomaly_score, [-1, segment_num])
        elif fusion == 'segments':
            reshaped_inputs = tf.reduce_max(inputs, axis=1)
            outputs = regression(reshaped_inputs, **kwargs)
        elif fusion == 'average':
            reshaped_inputs = tf.reduce_mean(inputs, axis=1)
            outputs = regression(reshaped_inputs, **kwargs)
        elif fusion == 'attention':
            with tf.variable_scope(fusion, 'regression', [inputs], reuse=tf.AUTO_REUSE):
                hk = tf.reshape(inputs, [-1, feature_len])
                _v = tf.get_variable('para_v', [attention_l, feature_len])
                th = tf.tanh(tf.matmul(_v, hk, transpose_b=True))
                _w = tf.get_variable('para_w', [1, attention_l])
                ep = tf.exp(tf.matmul(_w, th))
                ot = tf.reshape(tf.transpose(ep), [-1, segment_num])
                l1 = ot / tf.norm(ot, ord=1, axis=1, keepdims=True)
                fn = tf.expand_dims(l1, axis=2) * inputs
                reshaped_inputs = tf.reduce_sum(fn, axis=1)
            outputs = regression(reshaped_inputs, **kwargs)
        else:
            raise ValueError('Wrong fusion type: %s' % fusion)
    return outputs


def network_fn_list(inputs, **kwargs):
    outputs = []
    for k in inputs:
        with slim.arg_scope(arg_scope(weight_decay=0.0005)):
            anomaly_score = regression(k, **kwargs)
            outputs.append(anomaly_score)
    return outputs


def get_np_from_txt(txt_file_path, renum=1001):
    feature = basepy.read_txt_lines2list(txt_file_path)
    try:
        feature = random.sample(feature, renum)
    except ValueError:
        quotient, remainder = divmod(renum, len(feature))
        feature = feature * quotient + random.sample(feature, remainder)
    return np.array([i[0] for i in feature], dtype='float32')


def reform_train_list(org_txt_list, reform_txt_list):
    """
    Reform for some changes in list
    :param org_txt_list:    [['Abuse/Abuse001_x264.mp4'], ['Abuse/Abuse002_x264.mp4'],...]
    :param reform_txt_list: ['/absolute/datasets/anoma_motion16_tfrecords/Shoplifting@Shoplifting041_x264.txt',
                             '/absolute/datasets/anoma_motion16_tfrecords/normal_train@Normal_Videos308_0_x264.txt',]
    :return:    similar to reform_txt_list
    """
    print('List reform:')
    new_txt_list = []
    remove = 0
    replace = 0
    for trainee in org_txt_list:
        video_name = osp.basename(trainee[0]).split('_x264')[0]
        reform_txt = [i for i in reform_txt_list if video_name in i]
        if not reform_txt:
            print('Remove  %s from txt_list' % video_name)
            remove += 1
        elif len(reform_txt) > 1:
            new_txt_list.extend(reform_txt)
            print('Replace %s to' % video_name, reform_txt)
            replace += 1
        else:
            new_txt_list.extend(reform_txt)
    print('List reform DONE, remove %d videos, replace %d videos' % (remove, replace))
    return new_txt_list


def reform_np_array(np_array, reform=1001, model='standard'):
    if np_array.shape[0] > reform:
        np_copy = copy.deepcopy(np_array)
        np.random.shuffle(np_copy)
        np_output = np_copy[:reform, :4096]
    elif np_array.shape[0] == reform:
        np_output = np_array[:, :4096]
    else:
        quotient = reform // np_array.shape[0]
        np_temp = np_array.repeat(quotient, axis=0)
        np_output = np.concatenate((np_temp, np_array[:reform - len(np_temp)]), axis=0)[:, :4096]
    return np_output


def read_npy_file_path_list(npy_file_path_list, class_name_in_keys=True, sep='@'):
    feed_data = {}
    for npy_file_path in npy_file_path_list:
        # '/absolute/ext3t/anoma_motion16_npy_rand_2019_c50/normal_train@Normal_Videos167_x264.npy'
        # 'normal_train@Normal_Videos167_x264'
        file_name = osp.basename(osp.splitext(npy_file_path)[0])
        if class_name_in_keys:
            feed_data[file_name] = np.load(npy_file_path)
        else:
            # Normal_Videos167_x264
            file_name = osp.splitext(file_name.split(sep)[-1])[0]
            feed_data[file_name] = np.load(npy_file_path)
    return feed_data
