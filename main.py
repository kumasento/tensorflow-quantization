#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.tools.quantization.quantize_graph import GraphRewriter

tf.logging.set_verbosity(tf.logging.DEBUG)

import os
import sys
import time
import argparse

import models

FLAGS=None

MODEL_CLASSES = {
    "lenet": models.LeNetModel
}

def train_eval(
        sess,
        eval_correct,
        input_placeholder,
        label_placeholder,
        dataset):

    true_count = 0
    steps_per_epoch = dataset.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        image_feed, label_feed = dataset.next_batch(FLAGS.batch_size)

        feed_dict = {
            input_placeholder: image_feed,
            label_placeholder: label_feed
        }
        
        true_count += sess.run(eval_correct, feed_dict=feed_dict)

    precision = float(true_count) / num_examples
    print('\tNum examples: %10d  Num correct: %10d  Precision @ 1: %0.04f' %
            (num_examples, true_count, precision))

def train(model, log_dir, frozen_graph_name):
    """ Train the model and gives a frozen graph output """

    (train_data, test_data, validation_data) = model.load_dataset()
    (predictions, loss, train_op, eval_op) = model.build()

    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    with model.graph.as_default():
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir, model.graph)

        with tf.Session() as sess:
            sess.run(init_op)

            tf.train.write_graph(sess.graph_def, log_dir, 'graph.pb', as_text=False)

            for i in range(FLAGS.max_steps):
                start_time = time.time()

                image_feed, label_feed = train_data.next_batch(FLAGS.batch_size)
                feed_dict = {
                    model.input_placeholder: image_feed,
                    model.label_placeholder: label_feed
                }

                _, loss_value = sess.run(
                    [train_op, loss],
                    feed_dict=feed_dict)

                duration = time.time() - start_time

                if i % 100 == 0:
                    print('Step %6d: loss = %.2f (%.3f sec)' %
                            (i, loss_value, duration * 100))
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, i)
                    summary_writer.flush()
                    
                if (i + 1) % 1000 == 0:
                    checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=(i+1))
                    train_eval(sess, eval_op, model.input_placeholder, model.label_placeholder,train_data)
                    train_eval(sess, eval_op, model.input_placeholder, model.label_placeholder,test_data)
                    train_eval(sess, eval_op, model.input_placeholder, model.label_placeholder,validation_data)

            with tf.gfile.GFile(os.path.join(log_dir, 'saver.pb'), 'wb') as f:
                f.write(saver.to_proto().SerializeToString())

            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                model.graph.as_graph_def(),
                [eval_op.name.split(':')[0]])

            with tf.gfile.GFile(frozen_graph_name, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            
def eval_model(model, graph_name):
    """
    Evaluate the model based on a given graph
    """

    (_, test_data, _) = model.load_dataset()
    (_, _, _, eval_op) = model.build()

    with tf.gfile.GFile(graph_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

        input_placeholder = graph.get_tensor_by_name('import/' + model.input_placeholder.name)
        label_placeholder = graph.get_tensor_by_name('import/' + model.label_placeholder.name)
        imported_eval_op = graph.get_tensor_by_name('import/' + eval_op.name)

        with tf.Session() as sess:
            train_eval(
                sess,
                imported_eval_op,
                input_placeholder,
                label_placeholder,
                test_data)

def quantize(frozen_graph_name, quantized_graph_name):

def main(_):
    if not FLAGS.model_name:
        print("Please specify 'model name' through --model-name")
        return -1

    if FLAGS.model_name not in MODEL_CLASSES:
        print("Model name '" + FLAGS.model_name + "' does not exist!")
        return -1
    
    graph = tf.Graph()
    model = MODEL_CLASSES[FLAGS.model_name](graph=graph)

    if FLAGS.training:
        train(model, log_dir=FLAGS.log_dir, output_graph_name=FLAGS.output_graph_name)
    else:
        eval_model(model, frozen_graph_name=frozen_graph_name)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-name',
        type=str,
        default='',
        help='The name of the target model')

    parser.add_argument(
        '--quantized-graph-name',
        type=str,
        default='',
        help='The quantized graph of the target model')

    parser.add_argument(
        '--frozen-graph-name',
        type=str,
        default='',
        help='The name of frozen output graph')

    parser.add_argument(
        '--log-dir',
        type=str,
        default='',
        help='The directory that stores checkpoints and summary')

    parser.add_argument(
        '--batch-size',
        type=int,
        default=200,
        help='Size of the batch')

    parser.add_argument(
        '--max-steps',
        type=int,
        default=20000,
        help='Max number of steps')

    parser.add_argument(
        '--training',
        default=False,
        action='store_true',
        help='Train the model or not')

    return parser.parse_known_args()
    

if __name__ == '__main__':
    print(tf.__version__)
    FLAGS, _ = parse_args()
    tf.app.run()
