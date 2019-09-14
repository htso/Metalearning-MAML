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
from numpy import linalg as LA
import pickle
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from FunGenerator import FunGenerator

home = '/mnt/WanChai/Dropbox/Tensorflow-Mostly/MetaLearning/maml_ht'

def Unit_Test():
    batch_size = 9
    num_pts = 100
    n_sd = 0.2
    split = [0.6, 0.2, 0.2]

    x_rng = [0, 2*np.pi]
    offset = [-1, 1]
    slp = [-2, 2]
    Type = "linear"
    param_rng = { 'x':x_rng, "lin_slp":slp, "lin_offset":offset, "innovation_sd":n_sd, "Type":Type }

    # shape parameters used in Finn et al 2017 >>>>>>>>>>>>>>>>
    # amp_rng = [1, 1]    
    # x_rng = [0, 2*np.pi]
    # w = [0.5, 0.6] # smaller ==> smoother
    # ph = [0, 0]
    # ftype = [1, 1]
    # Type = "periodic"
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    amp_rng = [0.5, 5.0]
    x_rng = [0, 10]
    w = [1/(2*np.pi), 1/(2*np.pi)] # f(x) = a*sin( 2*pi(1/2*pi) x + ph ) = a*sin(x + ph)
    ph = [np.pi, 2*np.pi]
    fType = [0, 0]
    Type = "periodic"
    param_rng = { 'x':x_rng, 'amp':amp_rng, 'freq':w, 'phase':ph, 'function':fType, 
                  "innovation_sd":n_sd, "Type":Type }

    # param_rng = { 'x':x_rng, 'amp':amp_rng, 'freq':w, 'phase':ph, 'function':ftype, 
                  # "lin_slp":slp, "lin_offset":offset, "innovation_sd":n_sd, "Type":Type }
    #param_rng = { 'x':x_rng, "Type":Type } 
                  
    FG = FunGenerator(num_pts=num_pts, batch_size=batch_size, param_range=param_rng, train_test_split=split, dim_input=1, dim_output=1)
    
    # generate a batch of (x,y) data
    res = FG.generate()
    
    x = res["x"]
    y = res["y"]

    x_train = res["x_train"]
    x_val = res["x_val"]
    x_test = res["x_test"]
    # print('x_train shape', x_train.shape)
    # print('x_val shape', x_val.shape)
    # print('x_test shape', x_test.shape)

    # print('x_train max', np.max(x_train[0,:,0]))
    # print('x_train min', np.min(x_train[0,:,0]))
    # print('x_val max', np.max(x_val[0,:,0]))
    # print('x_val min', np.min(x_val[0,:,0]))
    # print('x_test max', np.max(x_test[0,:,0]))
    # print('x_test min', np.min(x_test[0,:,0]))

    y_train = res["y_train"]
    y_val = res["y_val"]
    y_test = res["y_test"]

    x_eq = res["x_equal_spaced"]
    y_eq = res["y_equal_spaced"]
    
    ymax = np.max(y)
    ymin = np.min(y)
    xmax = np.max(x)
    xmin = np.min(x)

    n = math.sqrt(batch_size)
    fig = plt.figure(figsize=(16,9))
    if Type is "periodic":
        st = fig.suptitle("w = [%1.1f %1.1f], ph = [%1.1f %1.1f], amp = [%1.1f %1.1f], ftype = [%d %d]" %(w[0], w[1], ph[0], ph[1], amp_rng[0], amp_rng[1], fType[0], fType[1]), fontsize="x-large")
    elif Type is "linear":
        st = fig.suptitle("slp = [%1.1f %1.1f], offset = [%1.1f %1.1f]" % (slp[0], slp[1], offset[0], offset[1]), fontsize="x-large")
    k = 1
    for i in range(5):
        ax1 = fig.add_subplot(5,4,k)
        ax1.plot(x_eq[i,:,0], y_eq[i,:,0], linewidth=1)
        ax1.scatter(x[i,:,0], y[i,:,0], s=20, color='black', alpha=0.8)
        ax1.set_title('All data')
        ax1.set_ylim([ymin, ymax])
        ax1.set_xlim([xmin, xmax])
        k = k+1
        ax1 = fig.add_subplot(5,4,k)
        ax1.plot(x_eq[i,:,0], y_eq[i,:,0], linewidth=1)
        ax1.scatter(x_train[i,:,0], y_train[i,:,0], s=20, color='blue', alpha=0.8)
        ax1.set_title('Train')
        ax1.set_ylim([ymin, ymax])
        ax1.set_xlim([xmin, xmax])  
        k = k+1
        ax1 = fig.add_subplot(5,4,k)
        ax1.plot(x_eq[i,:,0], y_eq[i,:,0], linewidth=1)
        ax1.scatter(x_val[i,:,0], y_val[i,:,0], s=20, color='green', alpha=0.8)
        ax1.set_title('Validation')
        ax1.set_ylim([ymin, ymax])
        ax1.set_xlim([xmin, xmax])  
        k = k+1
        ax1 = fig.add_subplot(5,4,k)            
        ax1.plot(x_eq[i,:,0], y_eq[i,:,0], linewidth=1)
        ax1.scatter(x_test[i,:,0], y_test[i,:,0], s=20, color='red', alpha=0.8)
        ax1.set_title('Test')
        ax1.set_ylim([ymin, ymax])
        ax1.set_xlim([xmin, xmax])        
        k = k+1

    # #plt.show()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.90)
    plt.savefig('Fun_shape.png')
    return None

if __name__ == "__main__":
    Unit_Test()
