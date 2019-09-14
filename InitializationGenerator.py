from __future__ import print_function

import sys
import os
import time
import csv
import math
import pickle
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

def GenerateInitializations(N, Hn, dim_input=1, dim_output=1, sigma=0.01, fnm=None):
    '''
    Generate multiple sets of initializations for a FC net 

    Args :
      N : number of initialization sets to generate
      Hn : specify the FC network, each element of this list is the number of hidden cells in each layer
           length of the list is the number of hidden layers in the neural network
      dim_input : input dimension, equal 1 for functional shapes
      dim_output : output dimension, equal 1 for functional shapes
      sigma : std deviation for the standard normal variates
      fnm : file name to save the initialized weights

    return : W_ls, list of dictionaries
    '''
    W_ls = []
    for i in range(N):
        w = Init_by_TruncatedNormal(Hn=Hn, dim_input=dim_input, dim_output=dim_output, sigma=sigma)
        W_ls.append(w)
    if fnm is not None:
        with open(fnm, 'wb') as f:
            pickle.dump({'W_ls': W_ls}, f)
        f.close()    
    return W_ls

def Init_by_TruncatedNormal(Hn, sigma=0.05, upper_b=1.5, lower_b=-1.5, dim_input=1, dim_output=1):
    '''
    Initialize parameters of a FC net by normal random variates
    
    Args :
      Hn : list that specifies the neural net hidden layers, each element is the # of hidden cells in that layer
      dim_input, dim_output : input and output dimension, e.g. equal 1 for functional shapes
      sigma : std deviation for the standard normal variates

    return : dictionary, len=2*(len(Hn)+1), whose elements consist of weight matrices and bias vectors
    '''
    W = {}
    W['w1'] = truncnorm.rvs(lower_b/sigma, upper_b/sigma, loc=0, scale=sigma, size=[dim_input, Hn[0]])
    W['b1'] = np.zeros(shape=[1, Hn[0]])
    for i in range(1, len(Hn)):
        W['w' + str(i+1)] = truncnorm.rvs(lower_b/sigma, upper_b/sigma, loc=0, scale=sigma, size=[Hn[i-1], Hn[i]]) 
        W['b' + str(i+1)] = np.zeros(shape=[1, Hn[i]])
    W['w' + str(len(Hn) + 1)] = truncnorm.rvs(lower_b/sigma, upper_b/sigma, loc=0, scale=sigma, size=[Hn[-1], dim_output]) 
    W['b' + str(len(Hn) + 1)] = np.zeros(shape=[1, dim_output])
    return W

def Read_init_from_file(fnm):
    '''
    Read back the list of parameter initializations from a pickle file
    '''
    f = open(fnm, 'rb')
    tmp = pickle.load(f)
    W = tmp['W_ls']
    f.close()
    return W


def Init_from_mean_mat(fnm, Hn, k_clust):
    '''
    Read the parameter mean from file and reshape it back to {W_i, b_i} format

    fnm : pickle file name
    Hn : list of size of hidden layers in a neural net
    k_clust : the kth initialization

    return : the weight matrices and bias vectors W1, W2, W3, b1, b2, b3
    '''
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


# ... need more work
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

