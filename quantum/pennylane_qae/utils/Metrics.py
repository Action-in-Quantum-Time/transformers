# Support functions for TS metrics
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for quantum time series analysis
# Date: 2024

import sys
sys.path.append('.')
sys.path.append('..')
sys.path

import numpy as np
import math

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


### Various metrics calculated between two TS window sets given as dictionaries

### Merge a window set given as a dictionary into a list of its entries
def merged_tswind(wind_dict, trim_left=0, trim_right=0):    
    wind_list = []
    sorted_keys = sorted(wind_dict.keys())
    if (trim_left+trim_right) <= len(wind_dict[sorted_keys[0]]):
        for sel_wind in sorted_keys:
            wind = wind_dict[sel_wind]
            wind = wind[trim_left:]
            wind = wind[:-trim_right] if trim_right>0 else wind
            wind_list.extend(wind)
    return wind_list

### Perform RMS on dictionaries
def rms_tswin(wind_exp, wind_pred, trim_left=0, trim_right=0):
    exp = merged_tswind(wind_exp, trim_left=trim_left, trim_right=trim_right)
    pred = merged_tswind(wind_pred, trim_left=trim_left, trim_right=trim_right)
    return np.sqrt(mean_squared_error(exp, pred))    

### Perform MAE on dictionaries
def mae_tswin(wind_exp, wind_pred, trim_left=0, trim_right=0):
    exp = merged_tswind(wind_exp, trim_left=trim_left, trim_right=trim_right)
    pred = merged_tswind(wind_pred, trim_left=trim_left, trim_right=trim_right)
    return mean_absolute_error(exp, pred)    

### Perform MAPE on dictionaries
def mape_tswin(wind_exp, wind_pred, trim_left=0, trim_right=0):
    exp = merged_tswind(wind_exp, trim_left=trim_left, trim_right=trim_right)
    pred = merged_tswind(wind_pred, trim_left=trim_left, trim_right=trim_right)
    return mean_absolute_percentage_error(exp, pred)    

### Calculate R score on dictionaries
def r2_tswin(wind_exp, wind_pred, trim_left=0, trim_right=0):
    exp = merged_tswind(wind_exp, trim_left=trim_left, trim_right=trim_right)
    pred = merged_tswind(wind_pred, trim_left=trim_left, trim_right=trim_right)
    return r2_score(exp, pred)    

### Merge a flat TS window
def merged_flat_tswind(wind_flat, trim_left=0, trim_right=0):    
    wind_list = []
    if (trim_left+trim_right) <= len(wind_flat[0]):
        for sel_wind in range(wind_flat.shape[0]):
            wind = wind_flat[sel_wind]
            wind = wind[trim_left:]
            wind = wind[:-trim_right] if trim_right>0 else wind
            wind_list.extend(wind)
    return np.array(wind_list)

### Perform RMS on TSs
def rms_flat_tswin(wind_exp, wind_pred, trim_left=0, trim_right=0):
    exp = merged_flat_tswind(wind_exp, trim_left=trim_left, trim_right=trim_right)
    pred = merged_flat_tswind(wind_pred, trim_left=trim_left, trim_right=trim_right)
    return np.sqrt(mean_squared_error(exp, pred))    

### Perform MAE on TSs
def mae_flat_tswin(wind_exp, wind_pred, trim_left=0, trim_right=0):
    exp = merged_flat_tswind(wind_exp, trim_left=trim_left, trim_right=trim_right)
    pred = merged_flat_tswind(wind_pred, trim_left=trim_left, trim_right=trim_right)
    return mean_absolute_error(exp, pred)    

### Perform MAPE on TSs
def mape_flat_tswin(wind_exp, wind_pred, trim_left=0, trim_right=0):
    exp = merged_flat_tswind(wind_exp, trim_left=trim_left, trim_right=trim_right)
    pred = merged_flat_tswind(wind_pred, trim_left=trim_left, trim_right=trim_right)
    return mean_absolute_percentage_error(exp, pred)    

### Calculate R score on TSs
def r2_flat_tswin(wind_exp, wind_pred, trim_left=0, trim_right=0):
    exp = merged_flat_tswind(wind_exp, trim_left=trim_left, trim_right=trim_right)
    pred = merged_flat_tswind(wind_pred, trim_left=trim_left, trim_right=trim_right)
    return r2_score(exp, pred)    
