'''
ParamPlay.py -- calculate the various statistics on the trained MLP weight-bias parameters from MLP-Fit-Fun.py
July 13, 2018
(c) Horace W Tso

reference :
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

path = '/media/rspace/WanChai/Dropbox/TSOTHOUGHTS/Tensorflow-CODE/maml_ht/FitFunShapes/'
home = '/home/rspace/dropb/TSOTHOUGHTS/Tensorflow-CODE/maml_ht'
os.chdir(home)

fnm1 = 'MLPparam_Dist_Loss_TrainData-iter5k-start-from-lastParamMean.pkl'
fnm2 = 'MLPparam_Dist_Loss_TrainData-iter200k-randomStart.pkl'
fnm3 = 'InitW_and_initBias.pkl'

# fg = FunGenerator(num_pts=0, batch_size=0, randomize=False)
# Nrows = fg.nrow
# Grid = fg.Grid
# print('Grid shape :', Grid.shape)
# Note : set inplace=True
# Grid.sort_values(['w1', 'w2', 'ph1', 'ph2'], ascending=[True,True,True,True], inplace=True) 
# print(Grid.iloc[:40])

# funX, funY, _, _ = fg.generate(num_pts=150, batch_size=Nrows, randomize=False, train=False)
# print('funX shape :', funX.shape)
# print('funY shape :', funY.shape)

# Read the result from fit-Per-Fun run
param, dist, loss_train, loss_val, yrange_train, Hn, Grid, funX, funY, Lambda, loss_threshold, \
    Ntrain, Nval, Ntest, max_iter, min_iter, xrng, use_init_file = GetParam(fnm2, path)

param_mat = np.array(param)
print('param shape:', param_mat.shape)

# Calculate the average of the parameters at the *end* of iteration for each fit
# then use these as the initial value to refit. This is the thinking behind MAML
# where the meta-learning finds the best starting point in the parameter space so
# that only one or few gradient steps are needed to learn a good model.
colmean = np.mean(param_mat[1:,:], axis=0)
# location indices on the flat param vector 'colmean'
h0 = Hn[0]
h1 = Hn[0]*Hn[1] + h0
h2 = h1 + Hn[1]
h3 = h2 + Hn[0]
h4 = h3 + Hn[1]
h5 = h4 + 1

initW1 = colmean[:h0]
initW2 = colmean[h0:h1]
initW3 = colmean[h1:h2]
initb1 = colmean[h2:h3]
initb2 = colmean[h3:h4]
initb3 = colmean[h4:h5]

print('initW1 len:', len(initW1))
print('initW2 len:', len(initW2))
print('initW3 len:', len(initW3))
print('initb1 len:', len(initb1))
print('initb2 len:', len(initb2))
print('initb3 len:', len(initb3))

initW1 = np.reshape(initW1, newshape=(1, Hn[0]))
initW2 = np.reshape(initW2, newshape=(Hn[0], Hn[1]))
initW3 = np.reshape(initW3, newshape=(Hn[1], 1))
initb1 = np.reshape(initb1, newshape=(1, Hn[0]))
initb2 = np.reshape(initb2, newshape=(1, Hn[1]))
initb3 = np.reshape(initb3, newshape=(1, 1))

print('initW1 shape:', initW1.shape)
print('initW2 shape:', initW2.shape)
print('initW3 shape:', initW3.shape)
print('initb1 shape:', initb1.shape)
print('initb2 shape:', initb2.shape)
print('initb3 shape:', initb3.shape)



#sys.exit()

# initfnm1 = './FitFunShapes/InitParam-LastAve.pkl'
# with open(initfnm1, 'wb') as f:
#     pickle.dump({'initW1': initW1}, f)
#     pickle.dump({'initW2': initW2}, f)
#     pickle.dump({'initW3': initW3}, f)
#     pickle.dump({'initb1': initb1}, f)
#     pickle.dump({'initb2': initb2}, f)
#     pickle.dump({'initb3': initb3}, f)
# f.close()

# sys.exit()



# Note : loss_train is a list of lists, where each element has the sequence of losses of every iteration
#        until convergence/cutoff. I'm only interested in the last value, thus....
loss_last = [ll[-1] for ll in loss_train]
loss_val_last = [ll[-1] for ll in loss_val]
print('loss_last len', len(loss_last))
print(loss_last[:10])

# pp = PdfPages('loss_distrib.pdf')

# fig, ax = plt.subplots(1,2, figsize=(14,8.27))

# ax[0].hist(loss_last, 30, normed=1, facecolor='blue', alpha=0.75)
# ax[0].set_title('Training Loss Distribution')
# ax[0].set_xlabel('Loss')
# ax[0].set_ylabel('Probability')
# ax[0].grid(True)

# ax[1].hist(loss_val_last, 30, normed=1, facecolor='blue', alpha=0.75)
# ax[1].set_title('Validation Loss Distribution')
# ax[1].set_xlabel('Loss')
# ax[1].set_ylabel('Probability')
# ax[1].grid(True)

# pp.savefig(fig, orientation = 'landscape')

# sys.exit()

# ===== Explaining and Interpreting Training Loss ===================================================
# Intuition is the farther the parameters (weights and biases) have to travel from the starting
# point (initWi, initbi), the harder it's to fit the function. So, how much could param distance 
# explain the variance in training loss?

# # Model 1. loss ~ dist
# X = np.array(dist).reshape(-1,1)
# y = np.array(loss_last).reshape(-1,1)
# mod = linear_model.LinearRegression()
# mod.fit(X, y)
# y_hat = mod.predict(X)
# coef = mod.coef_
# Mse = mean_squared_error(y_hat, y)
# r2 = r2_score(y, y_hat)

# plt.scatter(X, y,  color='black', marker='o', s=25)
# plt.plot(X, y_hat, color='blue', linewidth=1)
# plt.xlabel('Distance from start')
# plt.ylabel('Training Loss')
# plt.grid(True)
# plt.text(30, 45, 'R2:%2.3f' % r2)
# plt.text(30, 40, 'MSE:%2.3f' % Mse)
# plt.title('larger loss <=> param farther from init values')
# plt.show()

# Q : can training loss be explained by the generator parameter of the functional shapes ?
#     Judging from the last two graphs in Fit_Loss_WDist_allFun.pdf, there seems to be some
#     relationship betwn frequency of the product terms and training loss. Makes sense 
#     since the more widgling the shape, the harder it is for NN to fit. 

# Model 2. loss ~ w1 + w2
# X = Grid.iloc[:,:2]
# X = np.array(X)
# print(type(X))
# print(X.shape)
# y = np.array(loss_last).reshape(-1,1)
# mod = linear_model.LinearRegression()
# mod.fit(X, y)
# y_hat = mod.predict(X)
# coef = mod.coef_
# Mse = mean_squared_error(y_hat, y)
# r2 = r2_score(y, y_hat)

# print('coef :', coef)
# print('MSE :', Mse)
# print('R2 :', r2)

# fig, ax = plt.subplots(1,2, sharey=False, figsize=(14,8.27))
# ax[0].scatter(X[:,0], y, color='black', marker='o', s=25)
# ax[0].scatter(X[:,0]+0.05, y_hat, color='red', marker='s', s=40)
# ax[0].set_xlabel('w1')
# ax[0].set_ylabel('last loss')
# ax[0].grid(True)
# ax[0].set_title('Higher w1 leads to larger loss')

# ax[1].scatter(X[:,1], y, color='black', marker='o', s=25)
# ax[1].scatter(X[:,1]+0.05, y_hat, color='red', marker='s', s=40)
# ax[1].set_xlabel('w2')
# ax[1].set_ylabel('last loss')
# ax[1].grid(True)
# ax[1].set_title('...same with w2')
# plt.show()

# Conclusion : This kind of regression won't get very far in explaining the loss variance
#              because how well NN fits a functional shape depends on the starting parameter
#              values.  
# 
# ===============================================================================================

param_mat0 = param_mat[1:,:] # the first row is the random init values
# rid = np.reshape(range(param_mat0.shape[0]), (param_mat0.shape[0],1)) # row index
#print('rid:\n', rid[:10])
# param_mat1 = np.hstack((rid, param_mat0))
#print('param_mat1 shape:', param_mat1.shape)
#print(param_mat1[:5,-5:])

allW = param_mat0[:,:h2]
print(allW.shape)

# Since some of the runs didn't converge, get ride of those with large training losses
# ix = np.where(np.reshape(loss_last) < 3.5)
# ...better use list comprehension
ix = [i for i,v in enumerate(loss_last) if v < 3.5]
param_mat2 = param_mat0[ix,:]
print('param_mat2 shape :', param_mat2.shape)

# Distance matrix
# Dist = DistanceMetric.get_metric('euclidean')
# Dmat = Dist.pairwise(param_mat2[:,1:])
# print('Dmat :', Dmat.shape)
#print(Dmat[:20,:20])
# the closest 5 functional shapes in terms of NN weight/bias parameters
# k = 5
# ixa = [np.argpartition(x, k)[:k] for x in Dmat]
#print(len(ixa))
#print(ixa)

# ==== Hierchical clustering (scipy) ==============================================================
# Ref : https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
Z1 = linkage(allW, method='ward', metric='euclidean')
fig1 = plt.figure(figsize=(50, 10))
dn1 = dendrogram(
    Z1,
    #truncate_mode='lastp',  # show only the last p merged clusters
    #p=12,  # show only the last p merged clusters
    show_leaf_counts=True,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=10.,
    show_contracted=True  # to get a distribution impression in truncated branches
)
pp = PdfPages('allW_dendrog-ward-euclidean1.pdf')
pp.savefig(fig1, orientation='landscape')

Z2 = linkage(param_mat0, method='ward', metric='euclidean')
fig2 = plt.figure(figsize=(50, 10))
dn2 = dendrogram(
    Z2,
    #truncate_mode='lastp',  # show only the last p merged clusters
    #p=12,  # show only the last p merged clusters
    show_leaf_counts=True,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=10.,
    show_contracted=True  # to get a distribution impression in truncated branches
)
pp = PdfPages('Param_dendrog-ward-euclidean1.pdf')
pp.savefig(fig2, orientation='landscape')

#plt.show()

# max_d = 27
#i_clus = fcluster(Z, max_d, criterion='distance')
#print('clusters :\n', clusters)

# pick k, number of clusters 
k = 4
memb_ix = fcluster(Z2, k, criterion='maxclust')
# fclsuter returns a list of integers, each is the member id of the group it belongs
print('type :', type(memb_ix))
print('memb_ix len :', len(memb_ix))
print('clusters :\n', memb_ix)
# Q : How many members in each group ?
# RTable is python equivalent of R's table function
cnt = RTable(memb_ix)
print('cluster counts :\n', cnt)

# Calculate the mean parameters in each cluster
w_mean_mat = np.zeros((k, param_mat0.shape[1]))
for i in range(k):
    ix = np.where(memb_ix == i+1)
    #print('ix[0] type', type(ix[0]))
    #print('members of cluster :', ix)
    pp = param_mat0[ix[0].tolist(),:]
    # print('pp shape:', pp.shape)
    tmp = np.mean(pp, axis=0)
    w_mean_mat[i,:] = tmp
    print('col mean :', w_mean_mat[i,:10])



outfnm = './FitFunShapes/HC_k%i_Mean_Matrix.pkl' % k
with open(outfnm, 'wb') as f:
    pickle.dump({'w_mean_mat': w_mean_mat}, f)
    pickle.dump({'memb_ix': memb_ix}, f)
    pickle.dump({'Z2': Z2}, f)
    pickle.dump({'Z1': Z1}, f)
f.close()

# outfnm = './FitFunShapes/Param_Distance_Matrix.pkl'
# with open(outfnm, 'wb') as f:
#     pickle.dump({'Dmat': Dmat}, f)
# f.close()



# ==== this graph doesn't have much insight ===============================
# fig, ax = plt.subplots(2,2, sharey=False, figsize=(14,8.27))
# ax[0,0].scatter(Grid.iloc[:,0], Grid.iloc[:,1], c=i_clus[1:], cmap='prism')  # plot points with cluster dependent colors
# ax[0,0].set_xlabel('w1')
# ax[0,0].set_ylabel('w2')
# ax[0,1].scatter(Grid.iloc[:,0], Grid.iloc[:,2], c=i_clus[1:], cmap='prism')  # plot points with cluster dependent colors
# ax[0,1].set_xlabel('w1')
# ax[0,1].set_ylabel('ph1')
# ax[1,0].scatter(Grid.iloc[:,1], Grid.iloc[:,2], c=i_clus[1:], cmap='prism')  # plot points with cluster dependent colors
# ax[1,0].set_xlabel('w2')
# ax[1,0].set_ylabel('ph1')
# ax[1,1].scatter(Grid.iloc[:,1], Grid.iloc[:,3], c=i_clus[1:], cmap='prism')  # plot points with cluster dependent colors
# ax[1,1].set_xlabel('w2')
# ax[1,1].set_ylabel('ph2')
# plt.show()















