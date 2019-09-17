
import numpy as np
import os
import random
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt

class FunGenerator(object):
    """
    Generate periodic functional shapes slightly more complex than a sine or cosine. The current implementation generates a product of one sine and cosine.
    Just modify generate_period_fun for other possibility.
    """
    def __init__(self, train_test_split=[5, 5, 20], batch_size=1, param_range=None, dim_input=1, dim_output=1):
        """
        train_test_split : list of three integers, first is # training data, second # of validation, 
                           last is the # of test data, which is the last chunk in the sequence. 
                           So that the test set measures how well the model does on unseen, out-of-sample prediction.
        batch_size: size of meta batch size (e.g. number of functions)
        param_range : dictionary, where
            key : {x, amp, freq, phase, function, lin_slp, lin_offset}, 
            value : list of two values specifying the minimum and maximum of the parameter for that function
            For example, key = {"x": [0, 2*pi], amp": [0.5, 1.0], "freq": [1.0, 3.0], "phase": [pi, 2*pi], "function":[0, 2]}           
        noise_sd : variance of the normally distributed noise    
        dim_input : default to 1
        dim_output : default to 1
        type : 1=periodic function, 2=linear

        NOTE : It may be desirable to have certain number of cycles to occur in the data. Use x_range to
        control this. For ex,
            given w1 = 1, w2 = 3,
            set x_range = [0, 1] ==> generate one complete cycle of sine, three complete cycles of cosine
            set x_range = [0, 2] ==> generate two complete cycle of sine, six complete cycles of cosine
        """
        self.Type = param_range.get("Type", "periodic")
        self.batch_size = batch_size
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.x_rng = param_range.get("x", [0,10])
        self.amp = param_range.get("amp", [1,1])
        self.freq = param_range.get("freq", [1/(2*np.pi), 1/(2*np.pi)])
        self.phase = param_range.get("phase", [0,np.pi])
        self.function = param_range.get("function", [0,0])
        self.lin_offset = param_range.get("lin_offset", [0,0])
        self.lin_slp = param_range.get("lin_slp", [1,-1])
        self.innovation_sd = param_range.get("innovation_sd", 0)
        if self.Type == "periodic":
            self.generate = self.generate_periodic_fun
        elif self.Type == "linear":
            self.generate = self.generate_linear
        elif self.Type == "Finn":
            # Reproduce section 5.1 on Regression in Finn et al 2017 >>>>>>>>>>>>
            # These parameters produce the sine function used in Finn
            self.x_rng = [-5, 5]
            self.amp = [0.1, 5.0]
            self.freq = [1/(2*np.pi), 1/(2*np.pi)]
            self.phase = [0, np.pi]
            self.function = [0,0]
            self.innovation_sd = 0.0
            self.generate = self.generate_periodic_fun
        else:    
            raise ValueError("i don't know this Type?")

        self.n_train = train_test_split[0]
        self.n_val = train_test_split[1]
        self.n_test = train_test_split[2] 
        self.num_pts = sum(train_test_split)
        
    def train_val_test_split(self, x, y):
        '''
        Split sequence data into a training, validation, and a test set according to the following methodology.
        The first part, defined by train_pct + val_pct will be used to train model. The remaining part, the 
        last segment of a sequence, is for out-of-sample modeling test. 

                    train & val                             test
            |--------------------------------------------|--------|
        
        The train and validation set are mixed together. For example, some validation points
        may be in between training points, and vice versa. Goal is make the model learn the periodicity of the 
        function. Then test it on an out-of-sample area in the range of x that the model has never
        seen before.

        x, y shape : 

            (batch_size, number of data points, 1)

        '''
        x1 = x[:,:(self.n_train + self.n_val),:]
        y1 = y[:,:(self.n_train + self.n_val),:]
        # randomly mix the first (n_train + n_val) points
        ix = np.arange(self.n_train + self.n_val)
        random.shuffle(ix) # this function applies in place
        ix_train = ix[:self.n_train] # indices to the first n_train points
        ix_val = ix[-self.n_val:] # validation is the last n_val points
        # x
        x_train = x1[:,ix_train, :]
        x_val   = x1[:,ix_val,   :]
        x_test  =  x[:,-self.n_test:, :]
        # y = f(x)
        y_train = y1[:,ix_train, :]
        y_val   = y1[:,ix_val,   :]
        y_test  =  y[:,-self.n_test:, :]

        return x_train, y_train, x_val, y_val, x_test, y_test

    def linear_fun(self, x, offset, slp, ep):
        '''
        linear function :    y = a + b*x + noise
        '''
        y = offset + slp*x + ep
        return y

    def generate_linear(self):
        '''
        Generator of linear pair data of form (x, y)
        '''
        if np.absolute(self.lin_offset[0] - self.lin_offset[1]) < 1e-5:
            offset = [self.lin_offset[0]]*self.batch_size
        else :    
            offset = np.random.uniform(self.lin_offset[0], self.lin_offset[1], size=self.batch_size)
        if np.absolute(self.lin_slp[0] - self.lin_slp[1]) < 1e-5:
            slp = [self.lin_slp[0]]*self.batch_size
        else :    
            slp = np.random.uniform(self.lin_slp[0], self.lin_slp[1], size=self.batch_size)

        x_equal_spaced = np.zeros([self.batch_size, 100, self.dim_input])
        y_equal_spaced = np.zeros([self.batch_size, 100, self.dim_output])
        x = np.zeros([self.batch_size, self.num_pts, self.dim_input])
        y = np.zeros([self.batch_size, self.num_pts, self.dim_output])
        for i in range(self.batch_size):
            ep = np.random.normal(loc=0.0, scale=self.innovation_sd, size=self.num_pts)
            ep_equal_spaced = np.random.normal(loc=0.0, scale=self.innovation_sd, size=100)
            x_equal_spaced[i,:,0] = np.linspace(self.x_rng[0], self.x_rng[1], num=100)
            if self.x_rng[0] == self.x_rng[1]:
                x[i,:,0] = np.linspace(0, self.x_rng[1], num=self.num_pts)
            else:
                x[i,:,0] = np.sort(np.random.uniform(self.x_rng[0], self.x_rng[1], size=self.num_pts))
            #print('x[i] shape ', x[i].shape)    
            y[i,:,0] = self.linear_fun(x[i,:,0], offset=offset[i], slp=slp[i], ep=ep)
            y_equal_spaced[i,:,0] = self.linear_fun(x_equal_spaced[i,:,0], offset=offset[i], slp=slp[i], ep=ep_equal_spaced)
            #plt.scatter(x[i,:,0], y[i,:,0], s=20, color='blue', alpha=0.8)
            #plt.show()    

        x_train, y_train, x_val, y_val, x_test, y_test = self.train_val_test_split(x, y)
        # All data objects have this shape :
        # x_train, y_train, x_val, y_val, x_test, y_test,
        #  
        #     (batch_size, number of data points, 1)
        #
        res = {"x_train" : x_train, 
                "x_val"  : x_val, 
                "x_test" : x_test,
                "x"      : x, 
                "y_train": y_train,
                "y_val"  : y_val, 
                "y_test" : y_test,
                "y"      : y,
                "offset" : offset,
                "slp"    : slp,
                "innovation_sd" : self.innovation_sd,
                "x_equal_spaced":x_equal_spaced,
                "y_equal_spaced":y_equal_spaced}
        return res

    def periodic_fun(self, x, ftype, a1, a2, w1, w2, ph1, ph2, ep):
        '''
        library of simple functional forms consisting of sum or product of sine and cosine 
              ftype          function
            ------------------------------
                0             sin  
                1           sin + cos  
                2           sin * cos
        '''
        if ftype == 0:
            y = a1*np.sin(2*np.pi*w1*x + ph1) + ep
        elif ftype == 1:
            y = a1*np.sin(2*np.pi*w1*x + ph1) + a2*np.cos(2*np.pi*w2*x + ph2) + ep
        elif ftype == 2:
            y = a1 * np.sin(2*np.pi*w1*x + ph1) * np.cos(2*np.pi*w2*x + ph2) + ep
        else:
            raise ValueError('i don''t know this function type!')
        return y    
        
    def generate_periodic_fun(self):
        '''
        Function generator to generate a batch of functional shapes.  
        Need the following class variables.
            x_rng, amp, freq, phase, function
        Each of these is a list of two values, a lower bound and an upper bound. They are used
        to randomly pick num_pts points in between. For example, amp[0]=1, amp[1]=2, then
        num_pts random values would be picked between 1 and 2 for amp1 and amp2.

        If the lower is same as upper bound, the treatment is the follow,
            x_rng : 
                fixed spacing betwn two adjacent points,
                Ex :
                    x_rng[0] = x_rng[1] = 9 ==> x = (0, 1.0, 2.0, 3, ..., 9.0)
            freq :
                w1 = freq[0], w2 = w1 + small noise
            phase :
                phase1 = phase2 = self.phase[0]
        '''

        #print('freq[0] - freq[1] = ', np.absolute(self.freq[0] - self.freq[1])        )
        if np.absolute(self.freq[0] - self.freq[1]) < 1e-5:
            w1 = [self.freq[0]]*self.batch_size
            w2 = [self.freq[0] +  np.random.uniform(low=0.01, high=0.1, size=1)]*self.batch_size
            #print('w1 shape ', len(w1))
            #print('w2 shape ', len(w2))
        else :    
            w1 = np.random.uniform(self.freq[0], self.freq[1], size=self.batch_size)
            w2 = np.random.uniform(self.freq[0], self.freq[1], size=self.batch_size)

        if self.function[0] == self.function[1]:
            ftype = [self.function[0]]
        else:
            ftype = np.random.randint(low=self.function[0], high=self.function[1], size=self.batch_size)   
        #print('ftype :', ftype) 

        amp1 = np.random.uniform(self.amp[0], self.amp[1], size=self.batch_size)
        amp2 = np.random.uniform(self.amp[0], self.amp[1], size=self.batch_size)    
        phase1 = np.random.uniform(self.phase[0], self.phase[1], size=self.batch_size)
        phase2 = np.random.uniform(self.phase[0], self.phase[1], size=self.batch_size)
        # print('phase1 ', phase1)
        # print('phase2 ', phase2)
        
        x_equal_spaced = np.zeros([self.batch_size, 100, self.dim_input])
        y_equal_spaced = np.zeros([self.batch_size, 100, self.dim_output])
        x = np.zeros([self.batch_size, self.num_pts, self.dim_input])
        y = np.zeros([self.batch_size, self.num_pts, self.dim_output])
        for i in range(self.batch_size):
            ep = np.random.normal(loc=0.0, scale=self.innovation_sd, size=self.num_pts)
            #print('ep ', ep.shape)
            x_equal_spaced[i,:,0] = np.linspace(self.x_rng[0], self.x_rng[1], num=100)
            if self.x_rng[0] == self.x_rng[1]:
                x[i,:,0] = np.linspace(0, self.x_rng[1], num=self.num_pts)
            else:
                x[i,:,0] = np.sort(np.random.uniform(self.x_rng[0], self.x_rng[1], size=self.num_pts))
            #print('x[i] shape ', x[i].shape)    
            if len(ftype) > 1:
                fun_type = ftype[i]
            else:
                fun_type = ftype[0]
            y[i,:,0] = self.periodic_fun(x[i,:,0], ftype=fun_type, \
                                a1=amp1[i], a2=amp2[i], \
                                w1=w1[i], w2=w2[i], \
                                ph1=phase1[i], ph2=phase2[i], \
                                ep=ep )
            y_equal_spaced[i,:,0] = self.periodic_fun(x_equal_spaced[i,:,0], ftype=fun_type, \
                                a1=amp1[i], a2=amp2[i], \
                                w1=w1[i], w2=w2[i], \
                                ph1=phase1[i], ph2=phase2[i], \
                                ep=0 )
            #plt.scatter(x[i,:,0], y[i,:,0], s=20, color='blue', alpha=0.8)
            #plt.show()

        
        # Split the {x, y} into a training, a validation, and a test set 
        #
        #            train & val                             test
        #    |--------------------------------------------|--------|
        #
        # The train and validation set are mixed together. For example, some validation points
        # may be in between training points, and vice versa. Goal is make the model learn the periodicity of the 
        # function. Then test it on a truly out-of-sample data in the range of x that the model has never
        # seen before.
        #
        x1 = x[:,:(self.n_train + self.n_val),:]
        y1 = y[:,:(self.n_train + self.n_val),:]
        # randomly mix the first (n_train + n_val) points
        ix = np.arange(self.n_train + self.n_val)
        random.shuffle(ix) # this function applies in place
        ix_train = ix[:self.n_train] # indices to the first n_train points
        ix_val = ix[-self.n_val:] # validation is the last n_val points
        # x
        x_train = x1[:,ix_train, :]
        x_val   = x1[:,ix_val,   :]
        x_test  =  x[:,-self.n_test:, :]
        # y = f(x)
        y_train = y1[:,ix_train, :]
        y_val   = y1[:,ix_val,   :]
        y_test  =  y[:,-self.n_test:, :]

        # All data objects have this shape :
        # x_train, y_train, x_val, y_val, x_test, y_test,
        #  
        #     (batch_size, number of data points, 1)
        #
        res = {"x_train" : x_train, 
                "x_val"  : x_val, 
                "x_test" : x_test,
                "x"      : x, 
                "y_train": y_train,
                "y_val"  : y_val, 
                "y_test" : y_test,
                "y"      : y,
                "amp1"   : amp1,
                "amp2"   : amp2,
                "w1"     : w1, 
                "w2"     : w2, 
                "phase1" : phase1, 
                "phase2" : phase2, 
                "innovation_sd" : self.innovation_sd,
                "ftype"  : ftype,
                "x_equal_spaced":x_equal_spaced,
                "y_equal_spaced":y_equal_spaced}
        return res

    



