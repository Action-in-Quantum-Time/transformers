# Support functions
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for quantum time series analysis
# Date: 2023

import os
import matplotlib.pyplot as plt
import numpy as np
import pylab
import math
from IPython.display import clear_output


##### Defines several target functions for testing

### Sample Target functions
### - Each target defines its X range and y range needs to be [0,+1]
### - Assume X is either a scalar, a list or a vector
### - Returns y which is either a scalar, a list or a vector

# Common target class
class Target:
    name = "Target"
    
    # Initialises the target
    def __init__(self):
        self.xmin = -2*np.pi
        self.xmax = +2*np.pi
        self.ymin = 0.0
        self.ymax = 1.0
        self.epsilon = 0.1
        
    # Returns the X range of the function
    def xrange(self):
        return (self.xmin, self.xmax)
    
    # Returns the y range of the function
    def yrange(self):
        return (self.ymin, self.ymax)
    
    # Returns the epsilon (error to be generated)
    def eps(self):
        return self.epsilon
    
    # Returns the main function
    def fun(self, x):
        return x / (2.0*np.pi)
    
    # Plots target data in its natural range
    def plot(self, sample_no=20, color='blue', marker='None', linestyle='solid', 
             xlim=None, ylim=None, title=None, save_plot=None):
        if not title:
            title = 'Function "'+self.name+'"'
        sample_x = [self.xmin+i*(self.xmax-self.xmin)/sample_no
                    for i in range(sample_no)]
        sample_y = [self.fun(self.xmin+i*(self.xmax-self.xmin)/sample_no) 
                  for i in range(sample_no)]
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.title(title)
        plt.xlabel(f'Range ({sample_no} points)')
        plt.ylabel("Target value")
        if xlim is not None: plt.xlim(xlim)
        if ylim is not None: plt.ylim(ylim)
        plt.xlim(self.xrange())
        plt.plot(sample_x, sample_y, color=color, marker=marker, linestyle=linestyle)
        if save_plot is not None:
            os.makedirs(os.path.dirname(save_plot), exist_ok=True)
            plt.savefig(save_plot, format='eps')
        plt.show()

# Simple trig function
class Target_sin(Target):
    name = "Target_sin"
    
    def fun(self, x):
        return np.sin(x) / 2.0 + 0.5

# Complex trig function
class Target_2_sins(Target):
    name = "Target_2_sins"
    
    def fun(self, x):
        return (np.sin(5.0 * x) + 0.5*np.sin(8.0 * x)) / 4 + 0.5

# Complex trig function
class Target_3_sins(Target):
    name = "Target_3_sins"
    
    def fun(self, x):
        return (0.3 * np.sin(0.7 * x) + (np.sin(5.0 * x)/3 + 0.5*np.sin(8.0 * x)) / 5 + 0.5) * 80 / (x-100)+1

# Complex poly function
class Target_poly(Target):
    name = "Target_poly"

    def __init__(self):
        super().__init__()
        self.xmin = -0.9*np.pi
        self.xmax = +1.1*np.pi
        self.epsilon = 0.1
        
    def fun(self, x):
        return -(8*x-4*x**2+0.2*x**3-0.1*x**5)/70+0.1

# Complex poly function
class Target_poly_3(Target):
    name = "Target_poly"

    def __init__(self):
        super().__init__()
        self.xmin = -0.5
        self.xmax = +1
        self.epsilon = 0.1
        
    def fun(self, x):
        return 0.3-0.5*x-x**2+2*x**3

# Complex line function
class Target_line(Target):
    name = "Target_line"

    def __init__(self, slope=0.1, intercept=0.5, xmin=-2.0, xmax=+2.0):
        super().__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.slope = slope
        self.intercept = intercept
        self.epsilon = 0.1
        
    def fun(self, x):
        return self.intercept+self.slope*x

# Complex trig with trend function
class Target_trig_trend(Target):
    name = "Target_trig_trend"

    def __init__(self):
        super().__init__()
        self.xmin = -4.0
        self.xmax = +4.0
        self.epsilon = 0.1
        
    def fun(self, x):
        return 0.5+0.09*x+0.09*np.sin(3*x)+0.15*np.cos(6*x)

# Broken jitter
class Target_jitter(Target):
    name = "Target_jitter"

    def __init__(self):
        super().__init__()
        self.xmin = -6.0
        self.xmax = +6.0
        self.epsilon = 0.1
        self.point_no = 60 # 300
        self.breaks = [-3, 0, 3]
        self.scales = [0.2, 0.8, 0.4, 0.7]
        
    def fun_point(self, x):
        if (x < self.xmin):
            return 0.0
        elif (x < self.breaks[0]):
            return self.scales[0]+self.epsilon*np.random.random()
        elif (x < self.breaks[1]):
            return self.scales[1]+self.epsilon*np.random.random()
        elif (x < self.breaks[2]):
            return self.scales[2]+self.epsilon*np.random.random()
        elif (x < self.xmax):
            return self.scales[3]+self.epsilon*np.random.random()
        else:
            return 0.0
    
    def fun(self, x):
        if type(x) is int or type(x) is float:
            return self.fun_point(x)
        else:
            return np.array([self.fun_point(xi) for xi in x])

# Normalised beer sales data
class Target_beer(Target):
    name = "Target_beer"
    beer_data = [  \
       0.097, 0.0, 0.033, 0.124, 0.113, 0.042, 0.088, 0.08, 0.078, 0.055, \
       0.077, 0.138, 0.148, 0.135, 0.302, 0.165, 0.203, 0.187, 0.203, 0.242, \
       0.353, 0.269, 0.281, 0.357, 0.359, 0.376, 0.631, 0.213, 0.275, 0.281, \
       0.287, 0.291, 0.288, 0.337, 0.426, 0.382, 0.179, 0.165, 0.174, 0.218, \
       0.196, 0.225, 0.22, 0.272, 0.22, 0.273, 0.463, 0.205, 0.274, 0.309, \
       0.541, 0.581, 0.41, 0.095, 0.163, 0.194, 0.325, 0.301, 0.234, 0.147, \
       0.138, 0.132, 0.192, 0.178, 0.295, 0.173, 0.235, 0.299, 0.244, 0.212, \
       0.311, 0.296, 0.531, 0.51, 0.379, 0.447, 0.414, 0.471, 0.776, 0.344, \
       0.389, 0.353, 0.366, 0.411, 0.435, 0.393, 0.453, 0.404, 0.327, 0.338, \
       0.255, 0.269, 0.217, 0.219, 0.252, 0.278, 0.197, 0.207, 0.337, 0.561, \
       0.223, 0.312, 0.53, 0.652, 0.493, 0.131, 0.16, 0.343, 0.264, 0.178, \
       0.205, 0.221, 0.222, 0.179, 0.206, 0.237, 0.251, 0.24, 0.293, 0.555, \
       0.3, 0.282, 0.332, 0.396, 0.603, 0.515, 0.379, 0.476, 0.433, 0.536, \
       1.0, 0.474, 0.459, 0.471, 0.458, 0.448, 0.465, 0.484, 0.65, 0.494, \
       0.37, 0.358, 0.313, 0.303, 0.29, 0.245, 0.235, 0.322, 0.208, 0.226, \
       0.383, 0.679, 0.231, 0.35, 0.518, 0.806, 0.655, 0.177, 0.238, 0.229, \
       0.431, 0.338, 0.228, 0.219, 0.231, 0.246, 0.285, 0.307, 0.253, 0.347, \
       0.468, 0.331, 0.383, 0.369, 0.379, 0.481, 0.446, 0.685, 0.585, 0.474, \
       0.548, 0.498, 0.907, 0.606, 0.469, 0.462, 0.447, 0.493, 0.51, 0.472, \
       0.467, 0.669, 0.591, 0.396, 0.294, 0.342, 0.39, 0.353, 0.359, 0.368, \
       0.251, 0.32, 0.419, 0.683, 0.23, 0.36, 0.535, 0.819, 0.752, 0.193, \
       0.235, 0.297, 0.259, 0.465, 0.359, 0.209, 0.21, 0.23, 0.264, 0.34, \
       0.451, 0.266, 0.293, 0.346, 0.312, 0.299, 0.311, 0.41, 0.414, 0.692, \
       0.577, 0.487, 0.545, 0.622, 0.95, 0.782, 0.51, 0.532, 0.566, 0.61, \
       0.581, 0.553, 0.558, 0.68, 0.552, 0.35, 0.331, 0.376, 0.434, 0.412, \
       0.343, 0.311, 0.335, 0.318, 0.458, 0.821, 0.315, 0.341, 0.487, 0.954, \
       0.719, 0.293, 0.282, 0.243, 0.291, 0.576, 0.449, 0.25, 0.267, 0.267, \
       0.303, 0.36, 0.279, 0.311, 0.288, 0.425, 0.246, 0.272, 0.297, 0.33, \
       0.339, 0.641, 0.562, 0.4, 0.503, 0.506, 0.667, 0.73, 0.411, 0.418, \
       0.422, 0.471, 0.468, 0.471, 0.449, 0.567, 0.484, 0.332, 0.292, 0.28, \
       0.337, 0.288, 0.275, 0.278, 0.275, 0.294, 0.442, 0.668, 0.179, 0.275, \
       0.377, 0.716]
    
    def __init__(self, pt_from=None, pt_to=None):
        super().__init__()
        self.ts_data = self.beer_data.copy()
        pt_from = 0 if pt_from == None else pt_from
        pt_to = len(self.ts_data)-1 if pt_to == None else pt_to
        self.ts_data = self.ts_data[pt_from:pt_to]
        minv, maxv = min(self.ts_data), max(self.ts_data)
        self.ts_len = len(self.ts_data)
        self.xmin = 0
        self.xmax = self.ts_len-1
        self.ymin = min(self.ts_data)
        self.ymax = max(self.ts_data)
        self.epsilon = 0.1
        
    def fun_point(self, x):
        # print(x)
        if (x < self.xmin):
            return 0.0
        elif (x > self.xmax):
            return 0.0
        elif (int(x) == self.xmax):
            return self.ts_data[-1]
        else:
            lx = int(x)
            ux = lx+1
            ly = self.ts_data[lx]
            uy = self.ts_data[ux]
            return ly+(x-lx)*(uy-ly)/(ux-lx)

    def fun(self, x):
        if type(x) is int or type(x) is float or type(x) is np.float64:
            return self.fun_point(x)
        else:
            return np.array([self.fun_point(xi) for xi in x])

### Normalised data from CSV file
from utils.Files import *

class Target_csv_file(Target):
    name = "Target_csv_file"
    header = []

    def __init__(self, fpath, pt_from=None, pt_to=None, col=1, norm=True):
        super().__init__()
        
        self.header, self.ts_data = read_csv_file(fpath)
        self.ts_data = self.ts_data[:, col]
        self.ymin = min(self.ts_data)
        self.ymax = max(self.ts_data)
        self.ts_data = np.array([(y-self.ymin)/(self.ymax - self.ymin) for y in self.ts_data])

        pt_from = 0 if pt_from == None else pt_from
        pt_to = len(self.ts_data)-1 if pt_to == None else pt_to
        self.ts_data = self.ts_data[pt_from:pt_to]
        minv, maxv = min(self.ts_data), max(self.ts_data)
        self.ts_len = len(self.ts_data)
        self.xmin = 0
        self.xmax = self.ts_len-1
        self.ymin = min(self.ts_data)
        self.ymax = max(self.ts_data)
        self.epsilon = 0.1
        
    def fun_point(self, x):
        # print(x)
        if (x < self.xmin):
            return 0.0
        elif (x > self.xmax):
            return 0.0
        elif (int(x) == self.xmax):
            return self.ts_data[-1]
        else:
            lx = int(x)
            ux = lx+1
            ly = self.ts_data[lx]
            uy = self.ts_data[ux]
            return ly+(x-lx)*(uy-ly)/(ux-lx)

    def fun(self, x):
        if type(x) is int or type(x) is float or type(x) is np.float64:
            return self.fun_point(x)
        else:
            return np.array([self.fun_point(xi) for xi in x])

### Define a target function params
def target_split(f, samples, noise=0, train_pc=0.7, seed=None):
    
    if seed is not None: np.random.seed(seed)
    samples_train = int(samples * train_pc)
    samples_valid = samples-samples_train
        
    lb, ub = f.xrange()
    lb_train, ub_train = lb, lb+train_pc*(ub - lb)
    lb_valid, ub_valid = lb+train_pc*(ub - lb), ub
    T = (ub - lb)
    
    ### Prepare all X and y data
    X_all = np.linspace(lb, ub, num=samples)
    y_all = f.fun(X_all)
    
    ### Some of these are legacy
    X_train = (ub_train - lb_train) * np.random.random(samples_train) + lb_train
    X_train = np.sort(X_train, axis = 0)
    y_train = f.fun(X_train) + noise * (np.random.random(samples_train) - 0.5)
    X_valid = (ub_valid - lb_valid) * np.random.random(samples_valid) + lb_valid
    X_valid = np.sort(X_valid, axis = 0)
    y_valid = f.fun(X_valid) + noise * (np.random.random(samples_valid) - 0.5)
    
    ### Reshape Xs for fitting, scoring and prediction
    X_all = X_all.reshape(samples, 1)
    X_train = X_train.reshape(samples_train, 1)
    X_valid = X_valid.reshape(samples_valid, 1)

    return X_all, y_all, X_train, y_train, X_valid, y_valid