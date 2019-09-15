from __future__ import print_function

import sys
import os
import time
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def Step_CosineSimilarity(x, x_prev):
    '''
	Euclidean angle between two vectors
    '''
    dprod = np.dot(x.flatten(), x_prev.flatten())
    x_len = max(1e-6, np.linalg.norm(x.flatten()))
    prev_len = max(1e-6, np.linalg.norm(x_prev.flatten()))
    # make sure denominators are not zero
    similarity = dprod / (x_len*prev_len)
    similarity = max(min(similarity, 1.0), -1.0)
    angle = math.acos(similarity)
    return angle, similarity

def Step_L2_Distance(x, x0):
    '''
	L2 distance betwn two vectors
    '''
    return np.linalg.norm(x0 - x, axis=None)


