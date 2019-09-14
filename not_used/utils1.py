""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

# NOTE : reuse, scope not used in the activation function of FC model
def normalize(inp, activation, reuse, scope):
    if activation is not None:
        return activation(inp)
    else:
        return inp

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1]) # shape = [-1] flattens the entire tensor to a list [x1...xn]
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred - label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / update_batch_size
