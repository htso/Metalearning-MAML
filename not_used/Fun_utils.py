from __future__ import print_function

import sys
import os
import time
import csv
import math
import numpy as np
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def Step_Angle(x, x_prev):
    dprod = np.dot(x.flatten(), x_prev.flatten())
    x_len = np.linalg.norm(x.flatten())
    prev_len = np.linalg.norm(x_prev.flatten())
    val = dprod / (x_len*prev_len)
    if val < -1.0:
        val = -1.0
    if val > 1.0:
        val = 1.0
    angle = math.acos(val)
    return angle

def Step_Distance(x, x0):
    d = np.linalg.norm(x0 - x, axis=None)
    return d

def ConvertParam_V2Mat(v, Hn): 
    h0 = Hn[0]
    h1 = Hn[0]*Hn[1] + h0
    h2 = h1 + Hn[1]
    h3 = h2 + Hn[0]
    h4 = h3 + Hn[1]
    h5 = h4 + 1
    iW1 = v[:h0]
    iW2 = v[h0:h1]
    iW3 = v[h1:h2]
    ib1 = v[h2:h3]
    ib2 = v[h3:h4]
    ib3 = v[h4:h5]
    iW1 = np.reshape(iW1, newshape=(1, Hn[0]))
    iW2 = np.reshape(iW2, newshape=(Hn[0], Hn[1]))
    iW3 = np.reshape(iW3, newshape=(Hn[1], 1))
    ib1 = np.reshape(ib1, newshape=(1, Hn[0]))
    ib2 = np.reshape(ib2, newshape=(1, Hn[1]))
    ib3 = np.reshape(ib3, newshape=(1, 1))
    return iW1, iW2, iW3, ib1, ib2, ib3,  

# N : number of sets of initialization to generate
# Hn : list that specifies the neural net hidden layers, each element is the # of hidden cells in that layer
# dim_input, dim_output : input and output dimension, e.g. equal 1 for functional shapes
# sigma : std deviation for the standard normal variates
# fnm : file name to write the result
# RETURN : six lists of matrices
def GenearteMultipleInitParam(N, Hn, dim_input=1, dim_output=1, sigma=0.1, fnm=None):
    W1, W2, W3, b1, b2, b3 = [], [], [], [], [], []
    for i in range(N):
        iW1, iW2, iW3, ib1, ib2, ib3 = InitParamRandom(Hn, dim_input=dim_input, dim_output=dim_output, sigma=sigma, fnm=None)
        W1.append(iW1)
        W2.append(iW2)
        W3.append(iW3)
        b1.append(ib1)
        b2.append(ib2)
        b3.append(ib3)
    if fnm is not None:
        with open(fnm, 'wb') as f:
            pickle.dump({'W1': W1}, f)
            pickle.dump({'W2': W2}, f)
            pickle.dump({'W3': W3}, f)
            pickle.dump({'b1': b1}, f)
            pickle.dump({'b2': b2}, f)
            pickle.dump({'b3': b3}, f)
    return W1, W2, W3, b1, b2, b3


# Hn : list that specifies the neural net hidden layers, each element is the # of hidden cells in that layer
# dim_input, dim_output : input and output dimension, e.g. equal 1 for functional shapes
# sigma : std deviation for the standard normal variates
# fnm : file name to write the result
# RETURN : three matrices and three vectors (W1, W2, W3, b1, b2, b3)
def InitParamRandom(Hn, dim_input=1, dim_output=1, sigma=0.1, fnm=None):
    initW1 = np.random.normal(0, sigma, [dim_input, Hn[0]])
    initW2 = np.random.normal(0, sigma, [Hn[0], Hn[1]])
    initW3 = np.random.normal(0, sigma, [Hn[1], dim_output])
    initb1 = np.random.normal(0, sigma, [1, Hn[0]])
    initb2 = np.random.normal(0, sigma, [1, Hn[1]])
    initb3 = np.random.normal(0, sigma, [dim_output, 1])
    if fnm is not None:
        with open(fnm, 'wb') as f:
	        pickle.dump({'initW1': initW1}, f)
	        pickle.dump({'initW2': initW2}, f)
	        pickle.dump({'initW3': initW3}, f)
	        pickle.dump({'initb1': initb1}, f)
	        pickle.dump({'initb2': initb2}, f)
	        pickle.dump({'initb3': initb3}, f)
        f.close()
    return initW1, initW2, initW3, initb1, initb2, initb3

def Init_from_file(initfnm):
    f = open(initfnm, 'rb')
    tmp = pickle.load(f)
    initW1 = tmp['initW1']
    tmp = pickle.load(f)
    initW2 = tmp['initW2']
    tmp = pickle.load(f)
    initW3 = tmp['initW3']
    tmp = pickle.load(f)
    initb1 = tmp['initb1']
    tmp = pickle.load(f)
    initb2 = tmp['initb2']
    tmp = pickle.load(f)
    initb3 = tmp['initb3']
    f.close()
    return initW1, initW2, initW3, initb1, initb2, initb3

def Init_from_mean_mat(fnm, Hn):
    f = open(fnm, 'rb')
    tmp = pickle.load(f)
    W_mean_mat = tmp['w_mean_mat']
    f.close()
    print('W_mean_mat shape:', W_mean_mat.shape)
    v = W_mean_mat[k_clust,:]
    h0 = Hn[0]
    h1 = Hn[0]*Hn[1] + h0
    h2 = h1 + Hn[1]
    h3 = h2 + Hn[0]
    h4 = h3 + Hn[1]
    h5 = h4 + 1
    initW1 = v[:h0]
    initW2 = v[h0:h1]
    initW3 = v[h1:h2]
    initb1 = v[h2:h3]
    initb2 = v[h3:h4]
    initb3 = v[h4:h5]
    initW1 = np.reshape(initW1, newshape=(1, Hn[0]))
    initW2 = np.reshape(initW2, newshape=(Hn[0], Hn[1]))
    initW3 = np.reshape(initW3, newshape=(Hn[1], 1))
    initb1 = np.reshape(initb1, newshape=(1, Hn[0]))
    initb2 = np.reshape(initb2, newshape=(1, Hn[1]))
    initb3 = np.reshape(initb3, newshape=(1, 1))
    return initW1, initW2, initW3, initb1, initb2, initb3    

# R's table() function
# x : list of integers
# return : frequency counts of the values in x
def RTable(x):
    cnt = pd.Series(x).value_counts()
    return cnt

def GetParam(fnm, path, verbose=True):
    f = open(path+fnm, 'rb')
    xx = pickle.load(f)
    param = xx['param']
    xx = pickle.load(f)
    dist = xx['dist_from_start']
    xx = pickle.load(f)
    loss_train = xx['loss_train']
    xx = pickle.load(f)
    loss_val = xx['loss_val']
    xx = pickle.load(f)
    yrange_train = xx['yrange_train']
    xx = pickle.load(f)
    Hn = xx['Hn']
    xx = pickle.load(f)
    Grid = xx['Grid']
    xx = pickle.load(f)
    funX = xx['funX']
    xx = pickle.load(f)
    funY = xx['funY']
    xx = pickle.load(f)
    Lambda = xx['Lambda']
    xx = pickle.load(f)
    loss_threshold = xx['loss_threshold']
    xx = pickle.load(f)
    Ntrain = xx['Ntrain']
    xx = pickle.load(f)
    Nval = xx['Nval']
    xx = pickle.load(f)
    Ntest = xx['Ntest']
    xx = pickle.load(f)
    max_iter = xx['max_iter']
    xx = pickle.load(f)
    min_iter = xx['min_iter']
    xx = pickle.load(f)
    xrng = xx['xrng']
    xx = pickle.load(f)
    use_init_file = xx['use_init_file']
    f.close()
    
    if verbose:
        print('param len:', len(param))
        print('param[0] len:', len(param[1]))
        print('dist len:', len(dist))
        print('loss_train len:', len(loss_train))
        print('loss_val len:', len(loss_val))
        print('yrange_train len:', len(yrange_train))
        print('Hn:', Hn)
        print('Grid shape:', Grid.shape)
        print('funX shape:', funX.shape)
        print('funY shape:', funY.shape)
        print('Lambda:', Lambda)
        print('loss_threshold:', loss_threshold)
        print('Ntrain:', Ntrain)
        print('Nval:', Nval)
        print('Ntest:', Ntest)
        print('max_iter:', max_iter)
        print('min_iter:', min_iter)
        print('xrng:', xrng)
        print('use_init_file:', use_init_file)

    return param, dist, loss_train, loss_val, yrange_train, Hn, Grid, funX, funY, Lambda, loss_threshold, \
           Ntrain, Nval, Ntest, max_iter, min_iter, xrng, use_init_file