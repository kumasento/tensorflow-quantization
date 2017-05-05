#!/bin/sh

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

g++ -std=c++11 -shared quantized_conv2d_transpose.cc -o quantized_conv2d_transpose.so -fPIC -I $TF_INC -O2
