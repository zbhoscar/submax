import tensorflow as tf
import cv2
import numpy as np

frame_path = '/absolute/datasets/anoma/Abuse/Abuse001_x264/00001.jpg'

imga = cv2.imread(frame_path)
imgc = imga[50:100, 50:100]
data_encode = np.array(imgc[1])
str_encode = data_encode.tostring()
frame_cv = tf.image.decode_jpeg(str_encode)

cv2.imwrite('test.jpg', imgc)

frame_raw = tf.gfile.FastGFile('test.jpg', 'rb').read()
frame_tf = tf.image.decode_jpeg(frame_raw)

with tf.Session() as sess:

    b = sess.run(frame_tf)

    print('wow')
