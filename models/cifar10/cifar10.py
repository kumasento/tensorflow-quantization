import re
import tensorflow as tf
import cifar10_input

NUM_CLASSES = cifar10_input.NUM_CLASSES
FLAGS = tf.app.flags.FLAGS

TOWER_NAME = 'tower'

# Helper functions
def _get_dtype():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = _get_dtype()
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = _get_dtype()
    var = _variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _conv2d(name, input_tensor, shape):
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=shape,
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv)
    return conv

def _local(scope_name, input_tensor, shape, is_reshape=False):
    with tf.variable_scope(scope_name) as scope:
        if is_reshape:
            reshape = tf.reshape(input_tensor, [FLAGS.batch_size, -1])
            input_tensor = reshape
    
        dim = input_tensor.get_shape()[1].value
        weights = _variable_with_weight_decay('weights',
                                              shape=[dim, shape],
                                              stddev=0.04,
                                              wd=0.004)
        biases = _variable_on_cpu('biases', shape, tf.constant_initializer(0.1))
        local = tf.nn.relu(tf.matmul(input_tensor, weights) + biases, name=scope.name)
        _activation_summary(local)

    return local

# Tensor constructors
def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('please supply a data_dir')
    images, labels = cifar10_input.distorted_inputs(data_dir=FLAGS.data_dir,
                                                    batch_size=FLAGS.batch_size)
    return images, labels

def inference(images):
    conv1 = _conv2d('conv1', images, [5, 5, 3, 64])

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    conv2 = _conv2d('conv2', norm1, [5, 5, 64, 64])

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')

    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')

    local3 = _local('local3', pool2, 384, is_reshape=True)

    local4 = _local('local4', local3, 192)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES], stddev=1.0/192, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear 


