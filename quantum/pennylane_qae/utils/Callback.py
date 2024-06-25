# Support functions for QTSA workshop
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for quantum time series analysis
# Date: 2023

import matplotlib.pyplot as plt
import numpy as np
import pylab
import math
from IPython.display import clear_output


##### Callback functions for model training

### Callback function that draws a live plot when the .fit() method is called
### - Could change to record weights (TF coefficients?)

# Training callback
class Callback:
    name = "Regr_callback"
    
    # Initialises the callback
    def __init__(self, log_interval=50):
        self.objfun_vals = []
        self.log_interval = log_interval

    # Initialise callback lists
    # - For some reason [] defaults not always work (bug?)
    def reset(self, obfun=[]):
        self.objfun_vals = obfun

    # Find the first minimum objective fun value
    def min_obj(self):
        if self.objfun_vals == []:
            return (-1, 0)
        else:
            minval = min(self.objfun_vals)
            minvals = [(i, v) for i, v in enumerate(self.objfun_vals) if v == minval]
            return minvals[0]

    # Creates a simple plot of the objective functionm
    # - Can be used iteratively to make animated plot
    def plot(self):
        clear_output(wait=True)
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.title("Objective function")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(range(len(self.objfun_vals)), self.objfun_vals, color="blue")
        plt.show()

    # Callback function to store objective function values and plot
    def graph(self, weights, obj_func_eval):
        self.objfun_vals.append(obj_func_eval)
        self.plot()
            
    # Callback function to store objective function values but not plot
    def collect(self, weights, obj_func_eval):
        self.objfun_vals.append(obj_func_eval)
        current_batch_idx = len(self.objfun_vals)
        if current_batch_idx % self.log_interval == 0:
            prev_batch_idx = current_batch_idx-self.log_interval
            last_batch_min = np.min(self.objfun_vals[prev_batch_idx:current_batch_idx])
            print('Regr callback(', prev_batch_idx, ', ', current_batch_idx,') = ', last_batch_min)
            
