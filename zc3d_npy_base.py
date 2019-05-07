import tensorflow as tf
import data_io.basepy as basepy
import random
import numpy as np

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


def get_np_from_txt(txt_name, txt_path_list, renum=1001):
    txt_path = [i for i in txt_path_list if txt_name in i][0]
    feature = basepy.read_txt_lines2list(txt_path)
    try:
        feature = random.sample(feature, renum)
    except ValueError:
        quotient, remainder = divmod(renum, len(feature))
        feature = feature * quotient + random.sample(feature, remainder)
    return np.array([i[0] for i in feature], dtype='float32')