import re
import tensorflow as tf
import cifar10_input

IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

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

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + '(raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op

def train(total_loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    loss_average_op = _add_loss_summaries(total_loss)
    
    with tf.control_dependencies([loss_average_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def eval(logits, labels):
    pass
