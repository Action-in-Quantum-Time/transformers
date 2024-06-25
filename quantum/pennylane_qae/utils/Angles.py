# Support functions for TS angle encoding
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for quantum time series analysis
# Date: 2024

import sys
sys.path.append('.')
sys.path.append('..')
sys.path

import numpy as np
import math

# Angle encoding
# Deltas between consecutive time series values have been angle encoded. 
# In the context of a quibit representation (see the Figure), the encoding assumes zero to be encoded as H state, 
# negative values to be rotations up, while positive values as rotations down. 
#
# This encoding allows cumulative sequence calculations and easy value decoding upong the qubit measurements. 
# Should there be huge voilatility in data, additional scaling has been added to shrink the region of valid angular qubit positions.

### Angle encoding of a TS value relative to the previous value
#   val: value in [-1..+1] range to be encoded
#   optional scaler=np.pi/2: Number scaler 
#   optional err_range=0/0.05: allows range scaling to cater for accumulating errors
#   returns: Encoding of the next value relative to the previous value
def ts_relang_encode_val(val, scaler=np.pi/2, err_range=0):
    return val * scaler * (1 - 2 * err_range)

### Decoding
def ts_relang_decode_val(val, scaler=np.pi/2, err_range=0):
    return val / (scaler * (1 - 2 * err_range))

### Normalises the value to 1 (not required with angle encoding)
def ts_relang_norm_val(next_code):
    norm_code = next_code
    return norm_code

### Print encoding and decoding for testing
def print_ts_relang_encode_val(n):
    val = round(ts_relang_encode_val(n), 3)
    if val == 0:
        print(f'{(n)} -> {val} (π*{0.0})')
    else:
        print(f'{(n)} -> {val} (π/{round(np.pi / val, 3)})')
    
def print_ts_relang_decode_val(n):
    if n == 0:
        print(f'{round(n, 3)} (π*{0.0}) -> {ts_relang_decode_val(n)}')
    else:
        print(f'{round(n, 3)} (π/{round(np.pi / n, 3)}) -> {round(ts_relang_decode_val(n), 3)}')

def print_ts_relang_norm_val(p):
    print(f'{round(p, 3)} -> {round(ts_relang_norm_val(p), 3)}')

##### Encoding/decoding of the time series

### Encoding the entire data set
def ts_relang_encode(wind_set, scaler=np.pi):
    encoded_set = []
    for wind_idx in range(wind_set.shape[0]):
        wind = wind_set[wind_idx]
        encoded_wind = []
        for val_idx in range(wind.shape[0]):
            val = wind[val_idx]
            encoded_val = ts_relang_encode_val(val, scaler)
            encoded_wind.append(encoded_val)
        encoded_set.append(encoded_wind)
    org_wind_start = np.array([w[0] for w in wind_set])
    return np.array(encoded_set), org_wind_start

### Encoding the entire data set
def ts_relang_decode(org_wind_start, encoded_wind_set, scaler=np.pi):
    decoded_set = []
    for wind_idx in range(encoded_wind_set.shape[0]):
        wind = encoded_wind_set[wind_idx]
        decoded_wind = []
        for val_idx in range(wind.shape[0]):
            encoded_val = wind[val_idx]
            decoded_val = ts_relang_decode_val(encoded_val, scaler)
            decoded_wind.append(decoded_val)
            prev = decoded_val
        decoded_set.append(decoded_wind)
    return np.array(decoded_set)

### In agle encoding, encoded data is already normalised
def ts_relang_norm(encoded_wind_set):
    return np.array(encoded_wind_set)
