
import numpy as np
import os
import random
from itertools import product
import pandas as pd
#import tensorflow as tf
#from tensorflow.python.platform import flags


# python equivalent of R's expand.grid() 
def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())], columns=dictionary.keys())

class FunGenerator(object):
    """
    Generator to generate periodic functional shapes, such as product of sine and cosine, 
    where each component has different frequency and phase.
    """
    def __init__(self, num_pts=100, batch_size=1, dim_input=1, dim_output=1, randomize=True, config={}):
        """
        num_pts: number of points on the function to generate 
        batch_size: size of meta batch size (e.g. number of functions)
        randomize: whether to shuffle the rows of Grid, if False, sort in ascending order by column
        """
        self.batch_size = batch_size
        self.num_pts = num_pts
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.generate = self.genSineProduct
        self.w1 = np.array([1.0])
        self.w2 = np.array([3.0])
        #self.w1 = np.linspace(1.0, 3.0, num=3)
        #self.w2 = np.linspace(2.0, 5.0, num=4)
        self.ph1 = np.linspace(0.1*np.pi, 0.6*np.pi, num=4)
        self.ph2 = np.linspace(0.4*np.pi, 0.9*np.pi, num=4)
        self.grid_cols = {'w1':self.w1, 'w2':self.w2, 'ph1':self.ph1, 'ph2':self.ph2}
        # python equivalent of R's expand.grid() 
        self.G1 = expand_grid(self.grid_cols)
        # get ride of the rows where w1 is same as w2
        #self.ix = np.where(self.G1['w1'] != self.G1['w2'])
        #self.Grid = self.G1.iloc[self.ix]
        self.Grid = self.G1
        # shuffle the entries
        if randomize is True:
          ix1 = np.random.choice(self.Grid.shape[0], self.Grid.shape[0], replace=False)
          self.Grid = self.Grid.iloc[ix1]
        else:
          self.Grid = self.Grid.sort_values(['w1', 'w2', 'ph1', 'ph2'], ascending=[True,True,True,True])  
        self.nrow = self.Grid.shape[0]

    # To generate the desired number of cycles inside the x range, just specify x_range.
    # Example :
    #    w1 = 1, w2 = 3 
    #    x_range = [0, 1] ==> generate one complete cycle of sine, three complete cycles of cosine
    #    x_range = [0, 2] ==> generate two complete cycle of sine, six complete cycles of cosine
    def genSineProduct(self, num_pts=None, batch_size=None, iFun=0, x_range=None, randomize=True, train=True, add_noise=True, noise_sd=0.1):
        if num_pts is None:
            num_pts = self.num_pts
        if batch_size is None:
            batch_size = self.batch_size
        if x_range is None:
            xmin = 0.0
            xmax = 1.0
        else:
            xmin = x_range[0]
            xmax = x_range[1]
        yy = np.zeros([batch_size, num_pts, self.dim_output])
        xx = np.zeros([batch_size, num_pts, self.dim_input])
        if randomize is True and batch_size < self.nrow:
            ix = np.random.choice(self.Grid.shape[0], batch_size, replace=False)
            Subgrid = self.Grid.iloc[ix]
        else:
            Subgrid = self.Grid

        for i in range(batch_size):
            if train:
                xx[i] = np.random.uniform(xmin, xmax, [num_pts, self.dim_input])
            else:
                xx[i] = np.reshape(np.linspace(xmin, xmax, num=num_pts), newshape=(num_pts,1))
            
            if add_noise:
                ep = np.random.normal(loc=0.0, scale=noise_sd, size=[num_pts, self.dim_input])
            else:
                ep = 0.0

            # sin(w1*x+ph1) *cos(w2*x+ph2) 
            yy[i] = np.sin(2*np.pi*Subgrid.iloc[iFun,0]*xx[i] + Subgrid.iloc[iFun,2]) * np.cos(2*np.pi*Subgrid.iloc[iFun,1]*xx[i] + Subgrid.iloc[iFun,3]) + ep
        return xx, yy, None, None

    

