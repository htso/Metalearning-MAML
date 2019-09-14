from __future__ import print_function
import numpy as np
import pandas as pd
import os
import random
from itertools import product

def R_expand_grid(d):
	'''
    python equivalent of R's expand.grid() 	

    Args :
        d : dictionary
    return : data frame     
	'''
    return pd.DataFrame([row for row in product(*d.values())], columns=d.keys())

def R_table(x):
	'''
    python version of R's table() function	

    Args :
        x : list of integers
    return : frequency counts of the values in x    
	'''
    cnt = pd.Series(x).value_counts()
    return cnt


