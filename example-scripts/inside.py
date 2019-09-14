from __future__ import print_function

import sys
import os
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

seq_len=10

x_data = np.random.uniform(0,1,(10,1))
y_data = x_data*0.8 + 0.5

x=tf.placeholder('float',[seq_len,1])
y=tf.placeholder('float',[seq_len,1])


with tf.variable_scope('network_cell'):
   W = tf.get_variable('W', [10] )
   b = tf.get_variable('b', [1])

def give_output(inputs):

  output=[]
  #cell = tf.nn.rnn_cell.BasicRNNCell(5)
  #rnn_outputs, _ = tf.nn.static_rnn(cell, inputs, dtype = tf.float32, initial_state=None)

  with tf.variable_scope('network_cell', reuse=True):
    W = tf.get_variable('W', [10] )
    b = tf.get_variable('b', [1])
    output = inputs*W + b

  return output

def train_network():
  prediction= give_output(x)
  loss=tf.nn.l2_loss(prediction-y)

  optimizer = tf.train.AdamOptimizer().minimize(loss)

  
  s=tf.Session()
  init=tf.global_variables_initializer()
  s.run(init)

  _, loss_value=s.run([optimizer,loss], feed_dict={x:x_data,y: y_data})

  return loss_value
