import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# image_path = './tools/0.jpg'
image_path = '/home/zbh/PycharmProjects/anoma_v0.1/guinea/92.jpg'


def cv2_imread(image_path, to_float=False):
    # cv2.imread.dtype = uint8
    img_np = cv2.imread(image_path)
    img_np = img_np.astype(np.float32) / 255 if to_float else img_np
    return img_np


def cv2_imshow(img, msec=3000):
    # cv2.destroyAllWindows()
    # cv2.startWindowThread()
    cv2.namedWindow('image')
    cv2.imshow('image', img)
    cv2.waitKey(msec)
    cv2.destroyWindow("image")


def plt_imshow_single(img):
    plt.figure("figure")
    # plt.imshow(img)
    # most cases: BGR -> RGB
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis('on')
    plt.title('title')
    plt.show()


def mping_imread(image_path):
    return mpimg.imread(image_path)


def pillow_imread(image_path):
    return Image.open(image_path)


def pillow2plt_imshow_multi(img):
    gray = img.convert('L')
    r, g, b = img.split()
    img_merged = Image.merge('RGB', (r, g, b))
    plt.figure('figure')
    plt.suptitle('suptitle')

    plt.subplot(2, 3, 1), plt.title('title231')
    plt.imshow(img), plt.axis('off')

    plt.subplot(2, 3, 2), plt.title('title232')
    plt.imshow(gray, cmap='gray'), plt.axis('off')

    plt.subplot(2, 3, 3), plt.title('title233')
    plt.imshow(img_merged), plt.axis('off')

    plt.subplot(2, 3, 4), plt.title('title234')
    plt.imshow(r, cmap='gray'), plt.axis('off')

    plt.subplot(2, 3, 5), plt.title('title2315')
    plt.imshow(g, cmap='gray'), plt.axis('off')

    plt.subplot(2, 3, 6), plt.title('title236')
    plt.imshow(b, cmap='gray'), plt.axis('off')

    plt.show()
    # plt.savefig('test' + '.png')


# test cv2
cv2_imshow(cv2_imread(image_path))
# test pillow read and plt show
pillow2plt_imshow_multi(pillow_imread(image_path))
