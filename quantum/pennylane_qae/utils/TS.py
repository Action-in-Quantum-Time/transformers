# Support functions for TS manipulation
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for quantum time series analysis
# Date: 2023

import sys
sys.path.append('.')
sys.path.append('..')
sys.path

import numpy as np
import math
from utils.Window import *

### Generate TS data given a generator function
#   f: Generator function
#   samples: Number of samples to be gen erated
#   scale: Scaling to be applied to y values
#   pan: Panning up or down to be applied to y values
def gen_ts(f, samples, scale=1, pan=0):
    lb, ub = f.xrange()

    X_all = np.linspace(lb, ub, num=samples)
    y_all = [(y*scale+pan) for y in f.fun(X_all)]
    X_all = [x for x in range(len(X_all))]

    return X_all, y_all

### Add noise to TS
#   X_ts: TS X axis
#   y_ts: TS y axis
#   noise: Noise level 
#   noise_type: If 'abs' the added noise is +-noise*[min(y), max(y)], if 'rel' it is +-noise*y
def gen_noisy_ts(X_ts, y_ts, noise=0.0, noise_type='abs'):
    #print('Min and max:', min(y_ts), max(y_ts))
    if noise == 0:
        return X_ts, y_ts
    else:
        rng = np.random.default_rng()
        # noise_vec = (2*rng.random(len(y_ts))-1)*noise
        noise_vec = np.random.uniform(-1, 1, len(y_ts))*noise

        #print('\nNoise vector:\n', noise_vec, '\n')
        y_noisy = [y+e for (y, e) in zip(y_ts, noise_vec)]
        return X_ts, np.array(y_noisy)

### Create differenced TS data
#   X_ts: TS x axis data points coords
#   y_ts: TS y axis data points coords
def gen_diff_ts(X_ts, y_ts):
    y_deltas = []
    prev = y_ts[0]
    for next in range(len(y_ts)):
        y_deltas.append(y_ts[next]-prev)
        prev = y_ts[next]
    return X_ts, y_deltas

### Scale TS values
#   y_ts: TS values
#   from_y=-1: New range min value
#   to_y: +1: New range max value
def gen_scale_ts(y_ts, from_y=-1.0, to_y=1.0):
    if from_y > to_y:
        from_y, to_y = to_y, from_y
    y_min = min(y_ts)
    y_max = max(y_ts)
    y_scaled_ts = [(y - y_min) / (y_max - y_min) * (to_y - from_y) + from_y for y in y_ts]
    return y_scaled_ts, y_min, y_max

### Generate data given a generator function
#   f: Generator function
#   samples: Number of samples to be gen erated
#   train_pc: Percent of data to be used for training, e.g. 0.7
#   wind_size: Window size
#   wind_step: Windows step
#   scale=1: Optional scaling to be applied to y values
#   pan=0: Optional panning up or down to be applied to y values
#   differencing=True: Optional flag to request differencing of y values
#   noise=0: Optional level of noise to be added to TS y values
def gen_ts_windows(f, samples, train_pc, wind_size, wind_step, scale=None, pan=0, differencing=True, 
                   noise=0, noise_type='abs', debug=False, title='Selected Data'):

    ### Define a target function params
    samples_train = int(np.round(samples * train_pc, 0))
    samples_valid = samples - samples_train
    
    ### Prepare all X and y data
    X_all, y_all = gen_ts(f, samples)
    if noise != 0:
        _, y_all = gen_noisy_ts(X_all, y_all, noise=noise, noise_type=noise_type)
    if differencing:
        _, y_all = gen_diff_ts(X_all, y_all)
    if scale != None:
        y_all, old_miny, old_maxy = gen_scale_ts(y_all, scale[0], scale[1])
    
    ### Split data into windows
    # Create windowed time series, ignore X units
    # However, as the task is not predictive, so the horizon is ignored
    X_train_ts, y_train_ts, X_valid_ts, y_valid_ts = ts_wind_split(
        ts_wind_make(X_all, wind_size, wind_step), 
        ts_wind_make(y_all, wind_size, wind_step), 
        train_pc)

    ### Show TS parameters
    if debug:
        print(f'{title}\n')
        print(f'Function: {f.name}, Eps: {f.eps()}')
        print(f'Samples: {samples}, Split: {train_pc}, Train Samples: {samples_train}, Valid Samples: {samples_valid}')
        print(f'Window Size: {wind_size}, Step: {wind_step}, Horizon: {0}')
        print(f'Differencing: {differencing}, noise type="{noise_type}", noise added={noise}')
        print(f'Training Windows: {X_train_ts.shape[0]}, Validation Windows: {X_valid_ts.shape[0]}')
    
    return X_train_ts, y_train_ts, X_valid_ts, y_valid_ts

### Calculates the total noise as a percentage of value range
def gen_calculated_noise(windows_org, windows_noisy):
    noise_sum = 0
    points_no = 0
    hi_val = 0.0
    low_val = 0.0
    for wind_no in range(len(windows_org)):
        wind_org = windows_org[wind_no]
        wind_noisy = windows_noisy[wind_no]
        for val_no in range(len(wind_org)):
            points_no += 1
            noise_sum += np.abs(wind_noisy[val_no]-wind_org[val_no])
            hi_val = max(wind_noisy[val_no], hi_val)
            low_val = min(wind_noisy[val_no], low_val)
    noise = (noise_sum / points_no)/(hi_val - low_val)
    return noise, noise_sum, points_no