import data_io.basepy as base
import tensorflow as tf
# # import data_io.base_tf as base_tf
# import matplotlib.pyplot as plt
import os

import z1632clips_frame2tfrecords as test
batch_size = 1

tfrecords_path = '../guinea'

tfrecords_file_path_list = base.get_1tier_file_path_list(tfrecords_path, suffix='.tfrec')

with tf.name_scope('input'):
    first, second, third, t4, t5 = test.read_tfrecords(tfrecords_file_path_list,
                                               batch_size=batch_size,
                                               num_epochs=1)

init_op = tf.local_variables_initializer()

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
# gpu_options = tf.GPUOptions(allow_growth=True)  # , per_process_gpu_memory_fraction=0.32)
# config = tf.ConfigProto(gpu_options=gpu_options)  # allow_soft_placement=True,
#
# with tf.Session(config=config) as sess:
with tf.Session() as sess:

    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # a, b, c = sess.run([first, second, third, t4, t5])
    # print('test stamp')

    try:
        while True:
            a, b, c,d,e = sess.run([first, second, third,t4,t5])
            for i in range(batch_size):
                print(a[i],b[i],c[i],d[i])
    except Exception as error:
        coord.request_stop(error)
    coord.request_stop()
    coord.join(threads)

    # try:
    #     while not coord.should_stop():
    #         a, b, c = sess.run([first, second, third])
    # except tf.errors.OutOfRangeError:
    #     print("Epoch limit reached.")
    # finally:
    #     coord.request_stop()
    # coord.join(threads)


# plt.imshow(e[0,0,:,:,:][:, :, [2, 1, 0]])
print('wow')
print('end')