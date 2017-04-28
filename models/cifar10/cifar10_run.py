#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir',
                           '/mnt/data2/cifar-10-batches-bin',
                           """Directory of the CIFAR-10 dataset""")
tf.app.flags.DEFINE_string('log_dir',
                           '/tmp/cifar-10-log-dir',
                           """Directory of the CIFAR-10 training log""")
tf.app.flags.DEFINE_integer('batch_size',
                            128,
                            """Size of a batch""")
tf.app.flags.DEFINE_boolean('use_fp16',
                            False,
                            """Half float""")

import cifar10

def main(_):
    images, labels = cifar10.distorted_inputs()
    logits = cifar10.inference(images)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

if __name__ == '__main__':
    print('TensorFlow version %s' % tf.__version__)
    tf.app.run()
