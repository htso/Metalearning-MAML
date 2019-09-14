from __future__ import print_function

import sys
import os
import time
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def Step_Angle(x, x_prev):
    '''
	calculate angle between two vectors
    '''
    dprod = np.dot(x.flatten(), x_prev.flatten())
    x_len = max(1e-6, np.linalg.norm(x.flatten()))
    prev_len = max(1e-6, np.linalg.norm(x_prev.flatten()))
    val = dprod / (x_len*prev_len)
    val = max(min(val, 1.0), -1.0)
    angle = math.acos(val)
    return angle

def Step_Distance(x, x0):
    '''
	calculate the distance betwn two vectors
    '''
    return np.linalg.norm(x0 - x, axis=None)


