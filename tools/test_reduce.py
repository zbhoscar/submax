import numpy as np
import tensorflow as tf

feature_len = 4096
segment_num = 32
attention_l = 1024

# a = tf.reduce_mean(np.random.rand(128,32,4096), axis=1)
inputs = tf.convert_to_tensor(np.random.rand(64,32,4096),dtype=tf.float32)
# h = tf.reshape(a, [-1, 4096])
#
# v = tf.get_variable('para_v',[2000,4096])
#
# vh = tf.matmul(v, tf.transpose(h))
#
# tanh = tf.nn.tanh(vh)
#
# w = tf.get_variable('para_w',[1,2000])
#
# e = tf.exp(tf.matmul(w,tanh))
#
# outputs = tf.reshape(tf.transpose(e), [-1, 30])
#
# en = outputs / tf.norm(outputs, ord=1, axis=1, keepdims=True)
#
# fn = tf.expand_dims(en, axis=2) * a
#
# tf.reduce_sum(fn, axis=1)

hk = tf.reshape(inputs, [-1, feature_len])
_v = tf.get_variable('para_v', [attention_l, feature_len])
th = tf.tanh(tf.matmul(_v, hk, transpose_b=True))
_w = tf.get_variable('para_w', [1, attention_l])
ep = tf.exp(tf.matmul(_w, th))
ot = tf.reshape(tf.transpose(ep), [-1, segment_num])
l1 = ot / tf.norm(ot, ord=1, axis=1, keepdims=True)
fn = tf.expand_dims(l1, axis=2) * inputs
reshaped_inputs = tf.reduce_sum(fn, axis=1)


test1=tf.convert_to_tensor([[1,2],[3,4],[5,6]], dtype=tf.float32)
test3=tf.convert_to_tensor([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]], dtype=tf.float32)
test2=tf.convert_to_tensor([[1],[10],[100]],dtype=tf.float32)
en = tf.truediv(test1,test2)
with tf.Session() as sess:
    print(sess.run(en))
    print(sess.run(tf.norm(test1, ord=1, axis=1, keepdims=True)))
    print(sess.run(tf.truediv(test1, tf.norm(test1, ord=1, axis=1, keepdims=True))))
    print(sess.run(tf.expand_dims(test1, axis=2) * test3))

