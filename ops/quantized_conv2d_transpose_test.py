#!/usr/bin/env python

import tensorflow as tf
import numpy as np

class QuantizedConv2DTransposeTest(tf.test.TestCase):
    def test(self):
        module = tf.load_op_library('quantized_conv2d_transpose.so')
        
        input_tensor = tf.Variable(np.ones((1, 2, 2, 1)), dtype=tf.float32)
        filter_tensor = tf.Variable(np.ones((3, 3, 1, 1)), dtype=tf.float32)
        output_sizes = tf.Variable(np.array([1, 4, 4, 1]), dtype=tf.int32)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            min_input = 0
            max_input = 0
            min_filter = 0
            max_filter = 0
            out_type = tf.float32
            strides = [1, 1, 1, 1]
            padding = 'SAME'

            result = module.quantized_conv2d_transpose(input_tensor,
                                                       filter_tensor,
                                                       output_sizes,
                                                       min_input,
                                                       max_input,
                                                       min_filter,
                                                       max_filter,
                                                       out_type,
                                                       strides,
                                                       padding)
            print(sess.run(result))
            print(sess.run(
                tf.nn.conv2d_transpose(input_tensor, filter_tensor, output_sizes, strides, padding='VALID')))

if __name__ == '__main__':
    tf.test.main()
