'''
Restart-Results.py -- Analyse the impact of different initializations on neural network fit. 

July 13, 2018
(c) Horace W Tso

Clustering reference :
https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

'''
from __future__ import print_function
import sys
import os
import time
import csv
import math
import random
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
from FunGenerator import FunGenerator
from sklearn.neighbors import DistanceMetric
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from Fun_utils import *

home = '/home/rspace/dropb/TSOTHOUGHTS/Tensorflow-CODE/maml_ht'
os.chdir(home)

path = '/media/rspace/WanChai/Dropbox/TSOTHOUGHTS/Tensorflow-CODE/maml_ht/FitFunShapes/'

fnm = [None]*20
# Trained MLP parameters of the four sets of runs, trained for 5000 iterations
fnm[0] = 'Param_HC_k=1_iter5K.pkl'
fnm[1] = 'Param_HC_k=2_iter5K.pkl'
fnm[2] = 'Param_HC_k=3_iter5K.pkl'
fnm[3] = 'Param_HC_k=4_iter5K.pkl'
# Trained MLP parameters of the runs from random initialization
fnm[4] = 'Param_Dist_Loss_TrainData-iter200k-randomStart.pkl'
# Mean parameters of the four HC clusters
fnm[5] = 'HC_k4_Mean_Matrix.pkl'
# Trained MLP parameters of the four sets of runs, trained for 10,000 iterations
fnm[6] = 'Param_HC_k=1_iter10K.pkl'
fnm[7] = 'Param_HC_k=2_iter10K.pkl'
fnm[8] = 'Param_HC_k=3_iter10K.pkl'
fnm[9] = 'Param_HC_k=4_iter10K.pkl'
fnm[10] = 'Param-4starts-and-ave.pkl'


# Get the membership info for each functional shape that came out of clustering
f = open(path + fnm[5], 'rb')
xx = pickle.load(f)
w_mean_mat = xx['w_mean_mat']
xx = pickle.load(f)
memb_ix = xx['memb_ix']
f.close()
print('memb_ix : \n', memb_ix)

# Get the results of the four sets of runs using the cluster mean as the initial parameter values
LossTrain_mat = np.zeros((192,4))
LossVal_mat = np.zeros((192,4))
for i in range(6, 10, 1):
    param, dist, loss_train, loss_val, yrange_train, Hn, Grid, funX, funY, Lambda, loss_threshold, \
    	Ntrain, Nval, Ntest, max_iter, min_iter, xrng, use_init_file = GetParam(fnm[i], path, verbose=False)
    loss_train_last = [ll[-1] for ll in loss_train]
    loss_val_last = [ll[-1] for ll in loss_val]
    LossTrain_mat[:,i-6] = loss_train_last
    LossVal_mat[:,i-6] = loss_val_last

# Find the minimum loss among the four runs for each functional shape
min_col = np.argmin(LossTrain_mat, axis=1) + 1
print('train min_col :\n', min_col)

# Do they agree with the membership ID?
print('which equal to memb_ix ?\n', min_col == memb_ix)

# What percent matches the membership info?
print('what percent is consistent with membership info ?\n', 100.0*np.sum(min_col == memb_ix) / 192.0)

min_col = np.argmin(LossVal_mat, axis=1) + 1
#print('val min_col :\n', min_col)

print('LossTrain_mat : \n', LossTrain_mat[:10,:])
#print('LossTrain_mat : \n', LossTrain_mat[-10:,:])












