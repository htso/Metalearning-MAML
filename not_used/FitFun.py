"""
FitFun.py -- fit two-layer fc MLP on some functional shape. The shapes tested on take the form of 
      product of sinusoidal functions. For example,
      
           f(x) = sin(wx + phase) * cos(wx + phase)

June 30, 2018
(c) Horace W Tso

Purpose : To verfiy the conclusion in Finn et al (2017) that MAML meta-learning model trained on few shots
generalize well on unseen data.

RESULT OF EXPERIMENTS:

Starting from random initial values for the weight matrices and biases, each functional shape is fitted
with a 2-layer neural net with 40 hidden units each. I ran up to 100k iterations with 200 points of training
data. 

(The outputs of this training round are in 
'MLPparam_Dist_Loss_TrainData-iter200k-randomStart.pkl'
'Fit_Loss_WDist_allFun-iter100k-randomStart.pdf')

The overall average of these trained weights and biases is computed, ie. the column means of the 'param' matrix. 
This is what I call the Mean-Start. I then use this as the starting point for a second round of training. 
In this round, I train each model for 5,000 iterations, far fewer than the previous round. 
The thinking is, instead of starting from total ignorance, I start from the "mid point" which has learned 
something from among all the functional shapes. This should yield faster convergence.

Although these function shapes vary widely in terms of smoothness, they share certain common characteristics 
that could be exploited in training. This is the fundamental reason that meta-learning could work on few-shot
datasets. 

(The results of the 2nd run validated this thinking, see
MLPparam_Dist_Loss_TrainData-iter5k-start-from-lastParamMean.pkl
Fit_Loss_WDist_allFun-iter5k-start-from-lastParamMean.pdf)

Looking at the results, I have these observations. 

First, it takes fewer iterations to get a comparable loss from the Mean-Start initialization, as expected. 
This makes sense because we are already closer to some local minimum than a random starting point. 

Second, with the same iteration budget, the Mean-Start runs produce lower training loss, except for one function.
That is, of the 16 functional shapes, 15 have lower training loss compared with the random init run. This supports
the MAML hypothesis that using the average point as initialization is a good strategy.

However, it's less clear whether the Mean-Start strategy produces better generalization. Comparing the validation
loss, only 10 out of the 16 runs have better validation loss with Mean-Start init, the other 6 runs have worse
loss, sometimes a lot worse than the random init run

"""
from __future__ import print_function

import sys
import os
import time
import csv
import math
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
from FunGenerator import FunGenerator
from Fun_utils import *
from numpy import linalg as LA

home = '/home/rspace/dropb/TSOTHOUGHTS/Tensorflow-CODE/maml_ht'

gnm = './FitFunShapes/Fit_Loss_WDist_allFun.pdf'
#initfnm = './FitFunShapes/InitW_and_initBias.pkl'
initfnm = './FitFunShapes/InitParam-LastAve.pkl'
initfnm1 = './FitFunShapes/InitParam.pkl'
initfnm2 = './FitFunShapes/HC_k4_Mean_Matrix.pkl'
wout_nm = './FitFunShapes/MLPparam_Dist_Loss_TrainData.pkl'

wout_nm1 = './FitFunShapes/Param_HC_k=1_iter10K.pkl'
gnm1 = './FitFunShapes/FitGraphs_HC_k=1_iter10K.pdf'


def LearningPath(InitParam, Npts, Loss_threshold, LR_init, dim_input=1, dim_output=1, Lambda=0.0):
    # the initializations are provided in InitParam
    initW1, initW2, initW3, initb1, initb2, initb3 = InitParam

    FG = FunGenerator(num_pts=0, batch_size=0, randomize=False)

    tf.reset_default_graph()
    xs = tf.placeholder(tf.float32, [None, dim_input])
    ys = tf.placeholder(tf.float32, [None, dim_output])
    # TF symbolic variables
    # NOTE : use the *same* initial values for every model in this subset of functional shapes
    lr = tf.get_variable('lr', initializer=tf.constant(LR_init, dtype=tf.float32))
    W1 = tf.get_variable('W1', initializer=tf.constant(initW1, dtype=tf.float32))
    W2 = tf.get_variable('W2', initializer=tf.constant(initW2, dtype=tf.float32))
    W3 = tf.get_variable('W3', initializer=tf.constant(initW3, dtype=tf.float32))
    b1 = tf.get_variable('b1', initializer=tf.constant(initb1, dtype=tf.float32))
    b2 = tf.get_variable('b2', initializer=tf.constant(initb2, dtype=tf.float32))
    b3 = tf.get_variable('b3', initializer=tf.constant(initb3, dtype=tf.float32))
    # TF Graph (MLP internals)
    H1 = tf.matmul(xs, W1) + b1
    A1 = tf.nn.tanh(H1)
    H2 = tf.matmul(A1, W2) + b2
    A2 = tf.nn.tanh(H2)
    y_hat = tf.matmul(A2, W3) + b3
    # Loss is MSE with L2 regularization
    loss0 = tf.reduce_mean(tf.reduce_sum(tf.square(ys - y_hat), axis=0))
    # L2-regularization on weights only, no reg on biases
    reg_terms = Lambda*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3))
    loss = loss0 + reg_terms
    # choices of optimizers : Adam, but does it really matter?
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    #train_step = tf.train.RMSPropOptimizer(0.005, 0.9).minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # save the initial parameters (ie. random starts) before entering the optimizer
    param_start = sess.run([W1, W2, W3, b1, b2, b3])

    # list of values, but length varies depending on how soon it converges
    loss_path = []
    i = 0
    learning_sch = LR_init
    i_last = i 
    while True:
        Xtr, Ytr, _, _ = FG.generate(num_pts=Npts, batch_size=1, x_range=[0,10], randomize=False, train=False, add_noise=True, noise_sd=0.1)
        sess.run(train_step, feed_dict={xs:Xtr[0], ys:Ytr[0], lr:learning_sch})
        lss = sess.run(loss, feed_dict={xs:Xtr[0], ys:Ytr[0], lr:learning_sch})
        loss_path.append(lss)
        # make learning rate a function of trailing losses
        ma_all = np.mean(loss_path)
        ma = np.mean(loss_path[-10000:])
        std_ma = np.std(loss_path[-5000:])
        
        # SCHEDULE : lower learning rate if loss gets smaller, otherwise increase it
        if i > 10000 and (i - i_last) > 20000:
            if lss < ma:
                learning_sch = 0.95*learning_sch
            else:
                learning_sch = 1.15*learning_sch
            i_last = i

        if i % 1000 == 0:
            print('iter : %d, loss : %3.5f lr : %1.5f' %(i, lss, learning_sch))
        if lss < Loss_threshold:
            break    
        i = i + 1

    # Pull the final parameters
    param_final = sess.run([W1, W2, W3, b1, b2, b3])
    sess.close()

    return param_final, param_start, loss_path, i

def PerturbGradPaths(N_perturb, Perturb_sigma, InitParam, Npts, Loss_threshold, dim_input=1, dim_output=1, Learning_rate=1e-3, Lambda=0.0):
    iW1, iW2, iW3, ib1, ib2, ib3 = InitParam
    p0_base = np.concatenate((iW1.flatten(), iW2.flatten(), iW3.flatten(), ib1.flatten(), ib2.flatten(), ib3.flatten()), axis=0)
    print('==== Base Path ================')
    param_final_base, param_start_base, loss_path_base, niter_base = GradientPath(InitParam=InitParam, Npts=Npts, \
                                                     dim_input=dim_input, dim_output=dim_output, \
                                                     Loss_threshold=Loss_threshold, \
                                                     Learning_rate=Learning_rate, \
                                                     Lambda=Lambda)
    print('.... done.')
    fW1, fW2, fW3, fb1, fb2, fb3 = param_final_base
    p1_base = np.concatenate((fW1.flatten(), fW2.flatten(), fW3.flatten(), fb1.flatten(), fb2.flatten(), fb3.flatten()), axis=0)
    path_dist_base = LA.norm(p1_base - p0_base, ord=1)
    print('Base path len :', path_dist_base)

    INIT, PARAM, LOSS_PATH, ITER, PATH_LEN, DIST_BTWN_INIT, DIST_BTWN_FINAL = [], [], [], [], [], [], []
    for i in range(N_perturb):
        print('===== Perturbation No ', i, ' =============== ')
        iW1, iW2, iW3, ib1, ib2, ib3 = InitParam
        # perturb the initial parameters by a small normal disturbance
        ep1 = np.random.normal(0, Perturb_sigma, [iW1.shape[0], iW1.shape[1]])
        iW1 = iW1 + ep1
        ep2 = np.random.normal(0, Perturb_sigma, [iW2.shape[0], iW2.shape[1]])
        iW2 = iW2 + ep2
        ep3 = np.random.normal(0, Perturb_sigma, [iW3.shape[0], iW3.shape[1]])
        iW3 = iW3 + ep3
        ep4 = np.random.normal(0, Perturb_sigma, [ib1.shape[0], ib1.shape[1]])
        ib1 = ib1 + ep4
        ep5 = np.random.normal(0, Perturb_sigma, [ib2.shape[0], ib2.shape[1]])
        ib2 = ib2 + ep5
        ep6 = np.random.normal(0, Perturb_sigma, [ib3.shape[0], ib3.shape[1]])
        ib3 = ib3 + ep6
        PerturbParam = [iW1, iW2, iW3, ib1, ib2, ib3]
        p0_i = np.concatenate((iW1.flatten(), iW2.flatten(), iW3.flatten(), ib1.flatten(), ib2.flatten(), ib3.flatten()), axis=0)
        
        param_final, param_start, loss_path, niter = GradientPath(InitParam=PerturbParam, Npts=Npts, \
                                                     dim_input=dim_input, dim_output=dim_output, \
                                                     Loss_threshold=Loss_threshold, \
                                                     Learning_rate=Learning_rate, \
                                                     Lambda=Lambda)
        print('.... done.')
        fW1, fW2, fW3, fb1, fb2, fb3 = param_final
        p1_i = np.concatenate((fW1.flatten(), fW2.flatten(), fW3.flatten(), fb1.flatten(), fb2.flatten(), fb3.flatten()), axis=0)

        path_len_i = LA.norm(p1_i - p0_i, ord=1)
        d0_i = LA.norm(p0_i - p0_base, ord=1)
        d1_i = LA.norm(p1_i - p1_base, ord=1)
        print('path len : ', path_len_i)
        print('d0 : ', d0_i)
        print('d1 : ', d1_i)

        INIT.append([iW1, iW2, iW3, ib1, ib2, ib3])
        PARAM.append(param_final)
        LOSS_PATH.append(loss_path)
        ITER.append(niter)
        PATH_LEN.append(path_len_i)
        DIST_BTWN_INIT.append(d0_i)
        DIST_BTWN_FINAL.append(d1_i)

    return INIT, PARAM, LOSS_PATH, ITER, PATH_LEN, DIST_BTWN_INIT, DIST_BTWN_FINAL



def GradientPath(InitParam, Npts, Loss_threshold, dim_input=1, dim_output=1, Learning_rate=1e-4, Lambda=0.0):
    # the initializations are provided in InitParam
    initW1, initW2, initW3, initb1, initb2, initb3 = InitParam

    FG = FunGenerator(num_pts=0, batch_size=0, randomize=False)

    # Need this call to re-run the graph afresh
    tf.reset_default_graph()
    xs = tf.placeholder(tf.float32, [None, dim_input])
    ys = tf.placeholder(tf.float32, [None, dim_output])
    # TF symbolic variables
    # NOTE : use the *same* initial values for every model in this subset of functional shapes
    W1 = tf.get_variable('W1', initializer=tf.constant(initW1, dtype=tf.float32))
    W2 = tf.get_variable('W2', initializer=tf.constant(initW2, dtype=tf.float32))
    W3 = tf.get_variable('W3', initializer=tf.constant(initW3, dtype=tf.float32))
    b1 = tf.get_variable('b1', initializer=tf.constant(initb1, dtype=tf.float32))
    b2 = tf.get_variable('b2', initializer=tf.constant(initb2, dtype=tf.float32))
    b3 = tf.get_variable('b3', initializer=tf.constant(initb3, dtype=tf.float32))
    # TF Graph (MLP internals)
    H1 = tf.matmul(xs, W1) + b1
    A1 = tf.nn.tanh(H1)
    H2 = tf.matmul(A1, W2) + b2
    A2 = tf.nn.tanh(H2)
    y_hat = tf.matmul(A2, W3) + b3
    # Loss is MSE with L2 regularization
    loss0 = tf.reduce_mean(tf.reduce_sum(tf.square(ys - y_hat), axis=0))
    # L2-regularization on weights only, no reg on biases
    reg_terms = Lambda*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3))
    loss = loss0 + reg_terms
    # choices of optimizers : Adam, but does it really matter?
    train_step = tf.train.AdamOptimizer(Learning_rate).minimize(loss)
    #train_step = tf.train.RMSPropOptimizer(0.005, 0.9).minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # save the initial parameters (ie. random starts) before entering the optimizer
    param_start = sess.run([W1, W2, W3, b1, b2, b3])

    # list of values, but length varies depending on how soon it converges
    loss_path = []
    i = 0
    while True:
        Xtr, Ytr, _, _ = FG.generate(num_pts=Npts, batch_size=1, x_range=[0,10], randomize=False, train=False, add_noise=True, noise_sd=0.1)
        sess.run(train_step, feed_dict={xs:Xtr[0], ys:Ytr[0]})
        lss = sess.run(loss, feed_dict={xs:Xtr[0], ys:Ytr[0]})
        loss_path.append(lss)
        if i % 10000 == 0:
            print('iter : %d, loss : %3.5f ' %(i, lss))
        if lss < Loss_threshold:
            break    
        i = i + 1

    # Pull the final parameters
    param_final = sess.run([W1, W2, W3, b1, b2, b3])
    sess.close()

    return param_final, param_start, loss_path, i

# ============================================================================
# TrainAllFun : Train all functional shapes starting from a given initialization 
# ============================================================================
# Args
#   InitParam : list of arrays
#   Ntrain, Nval : number of training and validation data points
#   loss_threshold : threshold below which training will end
#   max_iter, min_iter : maximum and minimum numbers of iterations
#   Lambda : regularization param
#
# Return :
#    param : matrix where rows corresponds to the M functional shapes, columns the flatten W and bparameters
#    dist_from_start : list of length M, each entry is the distant of the final parameter vector from the start 
#    loss_train : list of length M, training loss at convergence/termination
#    loss_val : list of length M, validation loss at convergence/termination
def TrainAllFun(InitParam, Ntrain, Nval, loss_threshold, max_iter, Lambda, gnm):
    param, dist_from_start, loss_val, loss_train_cycle = [], [], [], []
    # loss_train_cycle is a list of lists
    pp = PdfPages(gnm)

    FG = FunGenerator(num_pts=0, batch_size=0, randomize=False)
    Grid = FG.Grid
    N_fun = Grid.shape[0]

    # initializations are provided in InitParam
    initW1, initW2, initW3, initb1, initb2, initb3 = InitParam
    
    for j in range(N_fun):
        # Need this call to re-run the graph afresh
        tf.reset_default_graph()
        xs = tf.placeholder(tf.float32, [None, 1])
        ys = tf.placeholder(tf.float32, [None, 1])
        # TF symbolic variables
        # NOTE : use the *same* initial values for every model in this subset of functional shapes
        W1 = tf.get_variable('W1', initializer=tf.constant(initW1, dtype=tf.float32))
        W2 = tf.get_variable('W2', initializer=tf.constant(initW2, dtype=tf.float32))
        W3 = tf.get_variable('W3', initializer=tf.constant(initW3, dtype=tf.float32))
        b1 = tf.get_variable('b1', initializer=tf.constant(initb1, dtype=tf.float32))
        b2 = tf.get_variable('b2', initializer=tf.constant(initb2, dtype=tf.float32))
        b3 = tf.get_variable('b3', initializer=tf.constant(initb3, dtype=tf.float32))
        # MLP internals
        H1 = tf.matmul(xs, W1) + b1
        A1 = tf.nn.tanh(H1)
        H2 = tf.matmul(A1, W2) + b2
        A2 = tf.nn.tanh(H2)
        y_hat = tf.matmul(A2, W3) + b3
        # Loss is MSE with L2 regularization
        loss0 = tf.reduce_mean(tf.reduce_sum(tf.square(ys - y_hat), axis=0))
        # L2-regularization on weights only, no reg on biases
        reg_terms = Lambda*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3))
        loss = loss0 + reg_terms
        # could use Adam, but doesn't really matter
        train_step = tf.train.RMSPropOptimizer(0.005, 0.9).minimize(loss)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        # save the initial parameters (ie. random starts) before entering the optimizer
        W1_start, W2_start, W3_start, b1_start, b2_start, b3_start = sess.run([W1, W2, W3, b1, b2, b3])
        W1_prev, W2_prev, W3_prev, b1_prev, b2_prev, b3_prev = W1_start, W2_start, W3_start, b1_start, b2_start, b3_start
        
        W1_angles, W2_angles, W3_angles, b1_angles, b2_angles = [], [], [], [], [] 
        W1_dist, W2_dist, W3_dist, b1_dist, b2_dist, b3_dist = [], [], [], [], [], []
        # list of values, but length varies depending on how soon it converges
        loss_train_fun, loss_val_fun = [], []

        i = 0
        while True:
            # shuffle the training points (train=True)
            Xtr, Ytr, _, _ = FG.generate(num_pts=Ntrain, batch_size=1, iFun=j, \
                                         x_range=[0,5], randomize=False, \
                                         train=True, add_noise=True, noise_sd=0.1)
            # ========================================================
            sess.run(train_step, feed_dict={xs: Xtr[0], ys: Ytr[0]})
            # ========================================================
            lss_train = sess.run(loss, feed_dict={xs: Xtr[0], ys: Ytr[0]})
            # .....but no need to shuffle the validation points (train=False)
            Xval, Yval, _, _ = FG.generate(num_pts=Nval, batch_size=1, iFun=j, \
                                         x_range=[0,5], randomize=False, \
                                         train=False, add_noise=True, noise_sd=0.1)
            lss_val = sess.run(loss, feed_dict={xs: Xval[0], ys: Yval[0]})
            # add one value to the list 'loss_train_fun'
            loss_train_fun.append(lss_train)
            loss_val_fun.append(lss_val)

            if i % 500 == 0:
                Wn_1, Wn_2, Wn_3, bn_1, bn_2, bn_3 = sess.run([W1, W2, W3, b1, b2, b3])
                # Angle betwn current param vector and the previous vector, 
                # which is recorded every 500 iterations.
                # A rough measure of cumulative directional change
                #print('Wn_1 :', Wn_1[:2,:2])
                #print('W1_prev :', W1_prev[:2,:2])
                W1_angles.append(Step_Angle(Wn_1, W1_prev))
                W2_angles.append(Step_Angle(Wn_2, W2_prev))
                W3_angles.append(Step_Angle(Wn_3, W3_prev))
                b1_angles.append(Step_Angle(bn_1, b1_prev))
                b2_angles.append(Step_Angle(bn_2, b2_prev))
                W1_prev = Wn_1
                W2_prev = Wn_2
                W3_prev = Wn_3
                b1_prev = bn_1
                b2_prev = bn_2
                b3_prev = bn_3

                # Distance of current param from the initial parameter values
                # I use this as a measure of learning, ie. how far do they move from the state of
                # complete ignorance. Euclidean distance is used. Not sure if distance metric
                # matters at all.
                W1_dist.append(Step_Distance(Wn_1, W1_start))
                W2_dist.append(Step_Distance(Wn_2, W2_start))
                W3_dist.append(Step_Distance(Wn_3, W3_start))
                b1_dist.append(Step_Distance(bn_1, b1_start))
                b2_dist.append(Step_Distance(bn_2, b2_start))
                b3_dist.append(Step_Distance(bn_3, b3_start))

            if i % 10000 == 0:
                print('j:', j, ' i:', i, ' train loss:', lss_train, ' val loss:', lss_val)
  
            ma50 = np.mean(loss_train_fun[-50:])
            if ma50 < loss_threshold or i > max_iter:
                break
            i = i + 1
        print('.... done after ', i, ' iterations.')  

        Xshow, Yshow, _, _ = FG.generate(num_pts=Ntrain, batch_size=1, iFun=j, \
                                         x_range=[0,5], randomize=False, \
                                         train=False, add_noise=True, noise_sd=0.1)
        Xtest, Ytest, _, _ = FG.generate(num_pts=Ntrain, batch_size=1, iFun=j, \
                                         x_range=[3,7], randomize=False, \
                                         train=False, add_noise=True, noise_sd=0.1)
        prediction_train = sess.run(y_hat, feed_dict={xs:Xshow[0]})
        prediction_test = sess.run(y_hat, feed_dict={xs:Xtest[0]})

        fig, ax = plt.subplots(2,3, sharey=False, figsize=(14,8.27))

        ax[0,0].plot(Xshow[0], Yshow[0], marker='o', linewidth=1, color='grey', ms=1)
        ax[0,0].scatter(Xshow[0], prediction_train, color='blue', s=25, marker='s', label='train predict')
        ax[0,0].set_title('train vs predict', fontsize=10)
        ax[0,0].legend(loc="upper left")
        ax[0,0].grid(True)
        
        ax[0,1].plot(Xtest[0], Ytest[0], marker='o', linewidth=1, color='grey', ms=1)
        ax[0,1].scatter(Xtest[0], prediction_test, color='red', s=25, marker='s', label="test predict")
        ax[0,1].set_title('test vs predict', fontsize=10)
        ax[0,1].legend(loc="upper left")
        ax[0,1].grid(True)

        mx = max(np.max(loss_val_fun), np.max(loss_train_fun))
        ax[1,0].semilogy(loss_train_fun, color='green', label="train loss")
        ax[1,0].semilogy(loss_val_fun, color='red', label="val loss")
        ax[1,0].set_title('train, val loss at epochs', fontsize=10)
        ax[1,0].legend(loc="upper right")
        ax[1,0].grid(True)
        ax[1,0].text(5, 0.8*mx, 'train L:%2.3f' % loss_train_fun[-1])
        ax[1,0].text(5, 0.7*mx, 'val L:%2.3f' % loss_val_fun[-1])

        ax[1,1].plot(W1_angles, color='green', label='W1')
        ax[1,1].plot(W2_angles, color='red', label='W2')
        ax[1,1].plot(W3_angles, color='blue', label='W3')
        ax[1,1].set_title('W_i angles at epoches', fontsize=10)
        ax[1,1].grid(True)
        ax[1,1].legend(loc="lower right")

        ax[1,2].plot(W1_dist, color='green', label='W1')
        ax[1,2].plot(W2_dist, color='red', label='W2')
        ax[1,2].plot(W3_dist, color='blue', label='W3')
        ax[1,2].set_title('W_i dist from start at epoches', fontsize=8)
        ax[1,2].grid(True)
        ax[1,2].legend(loc="lower right")

        ax[0,2].plot(b1_dist, color='blue', label='b1')
        ax[0,2].plot(b2_dist, color='red', label='b2')
        ax[0,2].set_title('b_i dist from start at epoches', fontsize=8)
        ax[0,2].grid(True)
        ax[0,2].legend(loc="upper left")

        pp.savefig(fig, orientation = 'landscape')

        # Pull the final parameters
        W1_final, W2_final, W3_final, b1_final, b2_final, b3_final = sess.run([W1, W2, W3, b1, b2, b3])
        sess.close()

        param_start = np.concatenate([W1_start.flatten(), W2_start.flatten(), W3_start.flatten(), b1_start.flatten(), b2_start.flatten(), b3_start.flatten()])
        param_final = np.concatenate([W1_final.flatten(), W2_final.flatten(), W3_final.flatten(), b1_final.flatten(), b2_final.flatten(), b3_final.flatten()])
        param.append(param_final)  
        dist_from_start.append(np.linalg.norm(param_start - param_final, axis=None))
        loss_val.append(lss_val) # picks up the last lss_val when the while loop ended
        # list of lists
        loss_train_cycle.append(loss_train_fun)

    pp.close()

    return param, dist_from_start, loss_train_cycle, loss_val

def run_TrainAllFun(): 
    os.chdir(home)
    # Hyperparameters ============================================
    Hn = [20, 20]
    Lambda = 0.02 # regularization parameter
    loss_threshold = 1.0
    Ntrain = 100
    Nval = 100
    Ntest = 500
    max_iter = 100000
    N_init = 2
    
    # Generate N_init sets of random weights and bias
    initW1, initW2, initW3, initb1, initb2, initb3 = GenearteMultipleInitParam(N_init, Hn, sigma=0.05, fnm=None)
    PARAM_START = [initW1, initW2, initW3, initb1, initb2, initb3]
    
    PARAM, DIST, LOSS_TRAIN, LOSSVALID = [], [], [], []
    
    for i in range(N_init):
        print('===== Random Init round :', i, " ===============================================")
        gnm = 'FigGraph_Init%d_maxIter%d.pdf' % (i, max_iter)
        iParam = [initW1[i], initW2[i], initW3[i], initb1[i], initb2[i], initb3[i]]
        param, dist_from_start, loss_train, loss_val = TrainAllFun(iParam, Ntrain, Nval, loss_threshold, max_iter, Lambda, gnm)
        PARAM.append(param)
        DIST.append(dist_from_start)
        LOSS_TRAIN.append(loss_train)
        LOSSVALID.append(loss_val)

    print('==== Computing parameter averages =================================================')
    colmean = np.zeros((N_init, np.array(PARAM[0]).shape[1])) 
    W1ave, W2ave, W3ave, B1ave, B2ave, B3ave = [], [], [], [], [], []    
    for i in range(N_init):
        mat = np.array(PARAM[i])
        tmp = np.mean(mat, axis=0)
        colmean[i,:] = tmp
        ww1, ww2, ww3, bb1, bb2, bb3 = ConvertParam_V2Mat(colmean[i,:], Hn)
        W1ave.append(ww1)
        W2ave.append(ww2)
        W3ave.append(ww3)
        B1ave.append(bb1)
        B2ave.append(bb2)
        B3ave.append(bb3)

    PARAMave, DISTave, LOSS_TRAINave, LOSSVALIDave = [], [], [], []
    for i in range(N_init):
        print('===== Mean Init Round :', i, ' ==============================================')
        gnm = 'FigGraph_MeanInitRound%d.pdf' % i
        iParam = [W1ave[i], W2ave[i], W3ave[i], B1ave[i], B2ave[i], B3ave[i]]
        param, dist_from_start, loss_train, loss_val = TrainAllFun(iParam, Ntrain, Nval, loss_threshold, max_iter, Lambda, gnm)
        PARAMave.append(param)
        DISTave.append(dist_from_start)
        LOSS_TRAINave.append(loss_train)
        LOSSVALIDave.append(loss_val)

    with open('TrainAllFun-%dStarts.pkl' % N_init, 'wb') as f:
        pickle.dump({'PARAM_START': PARAM_START}, f)
        pickle.dump({'PARAM': PARAM}, f)
        pickle.dump({'DIST': DIST}, f)
        pickle.dump({'LOSS_TRAIN': LOSS_TRAIN}, f)
        pickle.dump({'LOSSVALID': LOSSVALID}, f)
        pickle.dump({'PARAMave': PARAMave}, f)
        pickle.dump({'DISTave': DISTave}, f)
        pickle.dump({'LOSS_TRAINave': LOSS_TRAINave}, f)
        pickle.dump({'LOSSVALIDave': LOSSVALIDave}, f)
        pickle.dump({'colmean': colmean}, f)
        pickle.dump({'Hn': Hn}, f)
        pickle.dump({'Lambda': Lambda}, f)
        pickle.dump({'loss_threshold': loss_threshold}, f)
        pickle.dump({'Ntrain': Ntrain}, f)
        pickle.dump({'Nval': Nval}, f)
        pickle.dump({'max_iter': max_iter}, f)


def run_MultipleGradPaths(): 
    os.chdir(home)

    # Hyperparameters ============================================
    Hn = [20, 20]
    Lambda = 0.0 # regularization parameter
    Ntrain = 50
    Niter = 1000000
    Ntrials = 1
    loss_thres = 3.0
    learning_rate = 1e-4
    
    # Generate N_init sets of random weight matrices and bias vectors ===================================
    initW1, initW2, initW3, initb1, initb2, initb3 = GenearteMultipleInitParam(N=Ntrials, Hn=Hn, dim_input=1, dim_output=1, sigma=0.1, fnm=None)
    iParam = [initW1[0], initW2[0], initW3[0], initb1[0], initb2[0], initb3[0]]

    PARAM, LOSS_PATH, ITER = [], [], []

    for i in range(Ntrials):
        print('trial : ', i)
        param_final, param_start, loss_path, niter = GradientPath(InitParam=iParam, Npts=Ntrain, \
                                                     dim_input=1, dim_output=1, \
                                                     Loss_threshold=loss_thres, \
                                                     Learning_rate=learning_rate, \
                                                     Lambda=Lambda)
        PARAM.append(param_final)
        LOSS_PATH.append(loss_path)
        ITER.append(niter)
    
    # Q: Are the learned param the same in all trials ?
    # W11, W21, W31, b11, b21, b31 = PARAM[0]
    # W12, W22, W32, b12, b22, b32 = PARAM[1]
    # W13, W23, W33, b13, b23, b33 = PARAM[2]

    # W1_del = np.sum(abs(W11.flatten() - W12.flatten()))
    # W2_del = np.sum(abs(W21.flatten() - W22.flatten()))
    # W3_del = np.sum(abs(W31.flatten() - W32.flatten()))
    # b1_del = np.sum(abs(b11.flatten() - b12.flatten()))
    # b2_del = np.sum(abs(b21.flatten() - b22.flatten()))
    # b3_del = np.sum(abs(b31.flatten() - b32.flatten()))

    # print('1 vs 2 :')
    # print('W1 :', W1_del)
    # print('W2 :', W2_del)
    # print('W3 :', W3_del)
    # print('b1 :', b1_del)
    # print('b2 :', b2_del)
    # print('b3 :', b3_del)

    # W1_del = np.sum(abs(W11.flatten() - W13.flatten()))
    # W2_del = np.sum(abs(W21.flatten() - W23.flatten()))
    # W3_del = np.sum(abs(W31.flatten() - W33.flatten()))
    # b1_del = np.sum(abs(b11.flatten() - b13.flatten()))
    # b2_del = np.sum(abs(b21.flatten() - b23.flatten()))
    # b3_del = np.sum(abs(b31.flatten() - b33.flatten()))
    
    # print('1 vs 3 :')
    # print('W1 :', W1_del)
    # print('W2 :', W2_del)
    # print('W3 :', W3_del)
    # print('b1 :', b1_del)
    # print('b2 :', b2_del)
    # print('b3 :', b3_del)

    # W1_del = np.sum(abs(W12.flatten() - W13.flatten()))
    # W2_del = np.sum(abs(W22.flatten() - W23.flatten()))
    # W3_del = np.sum(abs(W32.flatten() - W33.flatten()))
    # b1_del = np.sum(abs(b12.flatten() - b13.flatten()))
    # b2_del = np.sum(abs(b22.flatten() - b23.flatten()))
    # b3_del = np.sum(abs(b32.flatten() - b33.flatten()))
    
    # print('2 vs 3 :')
    # print('W1 :', W1_del)
    # print('W2 :', W2_del)
    # print('W3 :', W3_del)
    # print('b1 :', b1_del)
    # print('b2 :', b2_del)
    # print('b3 :', b3_del)

    with open('GradientPath_result.pkl', 'wb') as f:
        pickle.dump({'PARAM': PARAM}, f)
        pickle.dump({'LOSS_PATH': LOSS_PATH}, f)
        pickle.dump({'iParam': iParam}, f)
        pickle.dump({'ITER': ITER}, f)
        
def run_LearningPaths(): 
    os.chdir(home)

    # Hyperparameters ============================================
    Hn = [40, 40]
    Lambda = 0.0 # regularization parameter
    Ntrain = 50
    Niter = 1000000
    Ntrials = 1
    loss_thres = 5.0
    learning_rate_init = 1e-2
    
    # Generate N_init sets of random weight matrices and bias vectors ===================================
    initW1, initW2, initW3, initb1, initb2, initb3 = GenearteMultipleInitParam(N=Ntrials, Hn=Hn, dim_input=1, dim_output=1, sigma=0.1, fnm=None)
    iParam = [initW1[0], initW2[0], initW3[0], initb1[0], initb2[0], initb3[0]]

    PARAM, LOSS_PATH, ITER = [], [], []

    for i in range(Ntrials):
        print('trial : ', i)
        param_final, param_start, loss_path, niter = LearningPath(InitParam=iParam, Npts=Ntrain, \
                                                     LR_init=learning_rate_init,
                                                     dim_input=1, dim_output=1, \
                                                     Loss_threshold=loss_thres, \
                                                     Lambda=Lambda)
        PARAM.append(param_final)
        LOSS_PATH.append(loss_path)
        ITER.append(niter)

def run_PerturbGradPaths(): 
    os.chdir(home)

    # Hyperparameters ============================================
    Hn = [20, 20]
    Lambda = 0.0 # regularization parameter
    Ntrain = 50
    N_perturb = 10
    sigma = 0.5
    loss_thres = 5.0
    learning_rate = 1e-3
    
    # Generate N_init sets of random weight matrices and bias vectors ===================================
    print('....generating initial param.')
    initW1, initW2, initW3, initb1, initb2, initb3 = GenearteMultipleInitParam(N=1, Hn=Hn, dim_input=1, dim_output=1, sigma=0.1, fnm=None)
    iParam = [initW1[0], initW2[0], initW3[0], initb1[0], initb2[0], initb3[0]]

    INIT, PARAM, LOSS_PATH, ITER, PATH_LEN, DIST_BTWN_INIT, DIST_BTWN_FINAL = PerturbGradPaths(N_perturb=N_perturb, Perturb_sigma=sigma, \
        InitParam=iParam, Npts=Ntrain, Loss_threshold=loss_thres, dim_input=1, dim_output=1, \
        Learning_rate=learning_rate, Lambda=Lambda)
    
    with open('PerturbGradientPaths_sigma=%d_result.pkl' % (sigma), 'wb') as f:
        pickle.dump({'INIT': INIT}, f)
        pickle.dump({'PARAM': PARAM}, f)
        pickle.dump({'LOSS_PATH': LOSS_PATH}, f)
        pickle.dump({'ITER': ITER}, f)      
        pickle.dump({'PATH_LEN': PATH_LEN}, f)      
        pickle.dump({'DIST_BTWN_INIT': DIST_BTWN_INIT}, f)      
        pickle.dump({'DIST_BTWN_FINAL': DIST_BTWN_FINAL}, f)      

if __name__ == "__main__":
    #run_PerturbGradPaths()
    #run_MultipleGradPaths()
    #run_TrainAllFun()
    run_LearningPaths()
