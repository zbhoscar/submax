import tensorflow as tf
import os


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def get_ckpt_path(flags_in):
    """
    Get ckpt file path from flags_in, flags_in is in form of path string.
    :param flags_in:    '/absolute/tensorflow_models/190305175801'          or
                        '/absolute/tensorflow_models/190305175801/190305175801.ckpt-6001'
    :return:        '/absolute/tensorflow_models/190305175801/190305175801.ckpt-10001'       or
                    '/absolute/tensorflow_models/190305175801/190305175801.ckpt-6001'
    """
    return tf.train.latest_checkpoint(flags_in) if os.path.isdir(flags_in) else flags_in


def repeat_string(string, index_list, sep=', '):
    """
    For example:        repeat_string("'frame%d: basetf._bytes_feature(image_raw_array[%d])'", [(0,0),(1,1)])
    :param string:      "'frame%d: basetf._bytes_feature(image_raw_array[%d])'"
    :param index_list:  [(0,0),(1,1)]
    :return:            
        "'frame0': basetf._bytes_feature(image_raw_array[0]), 'frame1': basetf._bytes_feature(image_raw_array[1]), "
    """
    string_all = ''
    for i in index_list:
        string_all = string_all + string % i + sep
    return string_all


def simple_preprocessing(imageo, img_size=(224, 224), shuffle=True):
    """
    A simple method of image pre-processing
    :param imageo:          input image
    :param img_size:        output image size
    :param shuffle:         if shuffle
    :return:                output image data
    """
    if shuffle:
        image = tf.random_crop(imageo, [img_size[0], img_size[1], 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=63. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        image = tf.image.per_image_standardization(image)
    else:
        image = tf.image.resize_image_with_crop_or_pad(imageo, img_size[0], img_size[1])
        image.set_shape([img_size[0], img_size[1], 3])
        image = tf.image.per_image_standardization(image)
    return image


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.

    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.

    Args:
      image: an image of shape [height, width, channels].
      offset_height: a scalar tensor indicating the height offset.
      offset_width: a scalar tensor indicating the width offset.
      crop_height: the height of the cropped image.
      crop_width: the width of the cropped image.

    Returns:
      the cropped (and resized) image.

    Raises:
      InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
    """Crops the given list of images.

    The function applies the same crop to each image in the list. This can be
    effectively applied when there are multiple image inputs of the same
    dimension such as:

      image, depths, normals = _random_crop([image, depths, normals], 120, 150)

    Args:
      image_list: a list of image tensors of the same dimension but possibly
        varying channel.
      crop_height: the new height.
      crop_width: the new width.

    Returns:
      the image_list with cropped images.

    Raises:
      ValueError: if there are multiple image inputs provided with different size
        or the images are smaller than the crop dimensions.
    """
    if not image_list:
        raise ValueError('Empty image_list.')

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    #
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
    with tf.control_dependencies(asserts):
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width,
                  crop_height, crop_width) for image in image_list]


def _central_crop(image_list, crop_height, crop_width):
    """Performs central crops of the given image list.

    Args:
      image_list: a list of image tensors of the same dimension but possibly
        varying channel.
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.

    Returns:
      the list of cropped images.
    """
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2

        outputs.append(_crop(image, offset_height, offset_width,
                             crop_height, crop_width))
    return outputs


def _mean_image_subtraction(image, means):
    """Subtracts the given means from each image channel.

    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.

    Returns:
      the centered image.

    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def _smallest_size_at_least(height, width, smallest_side):
    """Computes new shape with the smallest side equal to `smallest_side`.

    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.

    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: and int32 scalar tensor indicating the new width.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(tf.rint(height * scale))
    new_width = tf.to_int32(tf.rint(width * scale))
    return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
    """Resize images preserving the original aspect ratio.

    Args:
      image: A 3-D image `Tensor`.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      resized_image: A 3-D tensor containing the resized image.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                             align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image
