
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange

import tensorflow as tf

# target image size for training and evaluation
IMAGE_SIZE = 24

# global constants
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_cifar10(filename_queue):
    """
    Takes a filename queue as input, and generate a result with a label
    and a uint8image operators.
    """

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    label_bytes = 1

    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth

    # every record consists of a label followed by the image
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    
    # convert from a string to a vector of uint8
    record_bytes = tf.decode_raw(value, tf.uint8)

    # extract the label from the byte vector
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
        [result.depth, result.height, result.width])

    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result

def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    """
    Create batch operator from two queued operators:
    https://www.tensorflow.org/programmers_guide/reading_data
    """
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size)

    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
            for i in xrange(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    float_image = tf.image.per_image_standardization(distorted_image)

    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)

def inputs(eval_data, data_dir, batch_size):
    """
    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
        data_dir: Path to the CIFAR-10 data directory
        batch_size: Number of images per batch

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size
        labels: Labels. 1D tensor of [batch_size] size
    """
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # generate a queue contains files to read
    filename_queue = tf.train.string_input_producer(filenames)
    
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)

    float_image = tf.image.per_image_standardization(resized_image)

    # why we need to reset the shape of the tensor here?
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=False)

