#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time

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
tf.app.flags.DEFINE_integer('max_step',
                            20000,
                            """Max number of training steps""")
tf.app.flags.DEFINE_boolean('use_fp16',
                            False,
                            """Half float""")

import cifar10

def main(_):
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        images, labels = cifar10.distorted_inputs()

        logits = cifar10.inference(images)

        loss = cifar10.loss(logits, labels)

        train_op = cifar10.train(loss, global_step)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

            print('Training started')
            for i in range(FLAGS.max_step):
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time

                if i % 100 == 0:
                    print('Step %6d: loss = %.2f (%.3f sec)' %
                          (i, loss_value, duration))
                    summary_str = sess.run(summary)
                    writer.add_summary(summary_str, i)
                    writer.flush()

            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    print('TensorFlow version %s' % tf.__version__)
    tf.app.run()
