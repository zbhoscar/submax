import old_motion_clips2tfrecords as base_io
import tensorflow as tf
import time
import os

def get_input(file_path_list, num_epochs=None, is_training=True, batch_size=64):
    with tf.name_scope('input'):
        classb, videob, indexb, cropb, cb, rb, wb, hb, imageb = base_io.read_tfrecords(file_path_list,
                                                                                       num_epochs=num_epochs,
                                                                                       is_training=is_training,
                                                                                       batch_size=batch_size)
    return classb, videob, indexb, cropb, cb, rb, wb, hb, imageb


def run_test():
    # test_list_file = 'list/test.list'
    # num_test_videos = len(list(open(test_list_file,'r')))
    # print("Number of test videos={}".format(num_test_videos))

    # Get the sets of images and labels for training, validation, and
    # images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #                    Need to delete the empty video folder manually
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    classb, videob, indexb, cropb, cb, rb, wb, hb, imageb = get_input(
        ['/absolute/datasets/anoma_motion16_tfrecords_c50/normal_train@Normal_Videos715_x264.tfr',
         '/absolute/datasets/anoma_motion16_tfrecords_c50/normal_train@Normal_Videos719_x264.tfr',
         '/absolute/datasets/anoma_motion16_tfrecords_c50/normal_train@Normal_Videos307_5_x264.tfr'],
        num_epochs=1, is_training=False, batch_size=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    init_op = (tf.local_variables_initializer(), tf.global_variables_initializer())

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # one step to see
        # a, b, c, d, e = sess.run([classb, videob, clipb, segb, features])
        # print('wow')
        print('program begins, timestamp %s' % time.asctime(time.localtime(time.time())))

        try:
            while True:
                a, b, c, d, ac, ar, aw, ah, e = sess.run([classb, videob, indexb, cropb, cb, rb, wb, hb, imageb])

                # with tf.device('/cpu:0'):
                #     # l2e = e / np.linalg.norm(e, ord=2, axis=1, keepdims=True)
                #     l2e = e
                #     for i in range(len(c)):
                #         class_video_name = str(a[i], encoding='utf-8') + '@' + str(b[i], encoding='utf-8')
                #         feature_txt_path = os.path.join(EVAL_RESULT_FOLDER, class_video_name + '.txt')
                #
                #         _ = basepy.write_txt_add_lines(feature_txt_path, str(l2e[i].tolist()),
                #                                        str(c[i]), str(d[i]),
                #                                        str(ac[i]), str(ar[i]), str(aw[i]), str(ah[i]),
                #                                        str(max(l2e[i])), str(min(l2e[i])))
                #
                #     step += 1
                #     if time.time() - timestamp > 1800:
                #         localtime = time.asctime(time.localtime(time.time()))
                #         average_time_per_step = (time.time() - timestamp) / step
                #         print('program ongoing, timestamp %s, per step %s sec' % (localtime, average_time_per_step))
                #         step, timestamp = 0, time.time()
                print(a, b, c, d)
                print(e[0].shape)

        except Exception as error:
            coord.request_stop(error)

        coord.request_stop()
        coord.join(threads)

    print('wow')


def main(_):
    run_test()


if __name__ == '__main__':
    tf.app.run()