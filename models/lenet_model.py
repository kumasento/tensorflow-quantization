#!/usr/bin/env python
"""
This file provides the definition of the LeNet model, based on the TF layers API.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python import SKCompat
from tensorflow.contrib.learn.python import RunConfig 
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.logging.set_verbosity(tf.logging.INFO)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

IMAGE_SIZE = 28
NUM_CHANNELS = 1

DEFAULT_MNIST_DATA_DIR='/tmp/MNIST'

class LeNetModel(object):
    def __init__(
            self,
            graph=tf.Graph(),
            data_dir=DEFAULT_MNIST_DATA_DIR):
        """
        Args:
            graph: The TensorFlow graph to hold the built model
            data_dir: The directory to store the dataset
        """
        self.data_dir = data_dir
        self.graph = graph

    def load_dataset(self):
        mnist = read_data_sets(self.data_dir)
        train_data = mnist.train
        test_data = mnist.test
        validation_data = mnist.validation

        return (train_data, test_data, validation_data)

    def build(self, mode=learn.ModeKeys.TRAIN):
        with self.graph.as_default():
            self.input_placeholder = tf.placeholder(tf.float32, [None, IMAGE_SIZE * IMAGE_SIZE])
            self.label_placeholder = tf.placeholder(tf.int32, [None])

            input_layer = tf.reshape(
                self.input_placeholder,
                [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
            
            conv1 = tf.contrib.layers.conv2d(
                inputs=input_layer,
                num_outputs=32,
                kernel_size=[5, 5],
                padding='same',
                activation_fn=tf.nn.relu)

            pool1 = tf.contrib.layers.max_pool2d(
                inputs=conv1,
                kernel_size=[2, 2],
                stride=2)

            conv2 = tf.contrib.layers.conv2d(
                inputs=pool1,
                num_outputs=64,
                kernel_size=[5, 5],
                padding='same',
                activation_fn=tf.nn.relu)

            pool2 = tf.contrib.layers.max_pool2d(
                inputs=conv2,
                kernel_size=[2, 2],
                stride=2)

            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

            fc1 = tf.contrib.layers.fully_connected(
                inputs=pool2_flat,
                num_outputs=1024,
                activation_fn=tf.nn.relu)

            dropout = tf.contrib.layers.dropout(
                inputs=fc1,
                keep_prob=0.4,
                is_training=(mode == learn.ModeKeys.TRAIN))

            fc2 = tf.contrib.layers.fully_connected(
                inputs=dropout,
                num_outputs=10)

            train_op = None
            eval_op = None
            loss = None

            if mode != learn.ModeKeys.INFER:
                onehot_labels=tf.one_hot(
                    indices=tf.cast(self.label_placeholder, tf.int32),
                    depth=10)

                loss = tf.losses.softmax_cross_entropy(
                    onehot_labels=onehot_labels,
                    logits=fc2)

                eval_op = tf.reduce_sum(
                    tf.cast(
                        tf.nn.in_top_k(
                            fc2,
                            self.label_placeholder,
                            1), tf.int32))

            if mode == learn.ModeKeys.TRAIN:
                train_op = tf.contrib.layers.optimize_loss(
                    loss=loss,
                    global_step=tf.contrib.framework.get_global_step(),
                    learning_rate=0.001,
                    optimizer='Adam')

            predictions = {
                "classes": tf.argmax(input=fc2, axis=1),
                "probabilities": tf.nn.softmax(
                    fc2,
                    name="softmax_tensor")
            }

        self.eval_op = eval_op

        return (predictions, loss, train_op, eval_op)

def lenet_model_fn(features, labels, mode):
    # input layer
    input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2)

    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(
            inputs=pool2_flat,
            units=1024,
            activation=tf.nn.relu)

    dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.4,
            training=(mode == learn.ModeKeys.TRAIN))

    logits = tf.layers.dense(inputs=dropout, units=10)

    loss = None
    train_op = None

    if mode != learn.ModeKeys.INFER:
        onehot_labels=tf.one_hot(
                indices=tf.cast(labels, tf.int32),
                depth=10)

        loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels,
                logits=logits)

    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer='SGD')

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(
            logits,
            name="softmax_tensor")
    }

    return model_fn_lib.ModelFnOps(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)

def main(_):
    mnist = learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = learn.Estimator(
                config=RunConfig(gpu_memory_fraction=1./3),
                model_fn=lenet_model_fn,
                model_dir='/tmp/lenet_model')

    tensors_to_log = {} #{"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log,
            every_n_iter=50)

    SKCompat(mnist_classifier).fit(
            x=train_data,
            y=train_labels,
            batch_size=50,
            steps=100,
            monitors=[logging_hook])

    metrics = {
        "accuracy": learn.MetricSpec(
            metric_fn=tf.metrics.accuracy,
            prediction_key="classes"),
    }
    eval_results = SKCompat(mnist_classifier).score(
            x=eval_data,
            y=eval_labels,
            metrics=metrics)
    print(eval_results)

    mnist_classifier.save('/tmp/lenet_model')

if __name__ == '__main__':
    tf.app.run()
