"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function

import sys
import os
import time
import csv
import math
import numpy as np
import tensorflow as tf

Hn = [5, 7]

initW1 = np.random.normal(0, 0.1, [1, Hn[0]])
initW2 = np.random.normal(0, 0.1, [Hn[0], Hn[1]])
initW3 = np.random.normal(0, 0.1, [Hn[1], 1])
initb1 = np.random.normal(0, 0.1, [1, Hn[0]])
initb2 = np.random.normal(0, 0.1, [1, Hn[1]])
initb3 = np.random.normal(0, 0.1, [1, 1])

for j in range(2):
    print('j:', j)
    # Need this to re-run the graph afresh
    tf.reset_default_graph()

    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    W1 = tf.get_variable('W1', initializer=tf.constant(initW1, dtype=tf.float32))
    W2 = tf.get_variable('W2', initializer=tf.constant(initW2, dtype=tf.float32))
    W3 = tf.get_variable('W3', initializer=tf.constant(initW3, dtype=tf.float32))
    b1 = tf.get_variable('b1', initializer=tf.constant(initb1, dtype=tf.float32))
    b2 = tf.get_variable('b2', initializer=tf.constant(initb2, dtype=tf.float32))
    b3 = tf.get_variable('b3', initializer=tf.constant(initb3, dtype=tf.float32))
    
    print('xs graph:', xs.graph)
    print('W1 graph:', W1.graph)
    
    H1 = tf.matmul(xs, W1) + b1
    A1 = tf.nn.tanh(H1)
    H2 = tf.matmul(A1, W2) + b2
    A2 = tf.nn.tanh(H2)
    y_hat = tf.matmul(A2, W3) + b3

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - y_hat), axis=0))
    train_step = tf.train.RMSPropOptimizer(0.005, 0.9).minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    







