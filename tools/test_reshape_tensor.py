import tensorflow as tf
import numpy as np
import os

slim = tf.contrib.slim


a = 30
b = 32
c = 4096

def net(inputs):
    with tf.variable_scope('test', 'regression', [inputs], reuse=tf.AUTO_REUSE) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            return tf.reduce_mean(tf.reshape(inputs,[-1, 4096]),axis=1)

test = [ [[i*b+j]*c for j in range(b)] for i in range(a)]
# test_placeholder = tf.convert_to_tensor(test, dtype=tf.float32)
test_placeholder = tf.placeholder(tf.float32, [a,b,c])
end = net(test_placeholder)
end = tf.reshape(end, [-1, b])

init_op = tf.global_variables_initializer()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)
with tf.Session(config=config) as sess:
    sess.run(init_op)
    result = sess.run(end, feed_dict={test_placeholder:np.array(test,dtype='float32')})


    print('wow')
    print('wow')


