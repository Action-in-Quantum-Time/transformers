# Support functions for TS plotting
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for quantum time series analysis
# Date: 2024

import sys
sys.path.append('.')
sys.path.append('..')
sys.path

import os
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import clear_output

### Plot the target function and data
def plot_train_and_test_data(
    X_org, y_org, X_train, y_train, X_valid, y_valid,
    y_train_hat=None, y_valid_hat=None,
    xlim=None, ylim=None, rcParams=(12, 6),
    legend_cols=3,
    labels=['Target function', 'Training data', 'Test data', 'Fitted model', 'Model predictions'],
    colors=['lightblue', 'lightblue', 'pink', 'blue', 'red'],
    linestyles=['dashed', 'solid', 'solid', 'dashed', 'dashed'],
    xlabel='Range', ylabel='Target value',
    title='Target function with noisy data',
    save_plot=None):

    # Parameter values
    if rcParams is not None:
        plt.rcParams["figure.figsize"] = rcParams
        
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Plot target function
    if linestyles[0] != 'none':
        plt.plot(X_org, y_org, color=colors[0], linestyle=linestyles[0], label=labels[0])
    plt.plot(X_train, y_train, color=colors[1], linestyle=linestyles[1], label=labels[1])
    plt.plot(X_valid, y_valid, color=colors[2], linestyle=linestyles[2], label=labels[2])
    
    # Plot fitted line
    if y_train_hat is not None:
        plt.plot(X_train, y_train_hat, color=colors[3], linestyle=linestyles[3], label=labels[3])
        plt.plot(X_train, y_train_hat, color=colors[3], marker='o', linestyle='None')
    else:
        plt.plot(X_train, y_train, color=colors[3], marker='o', linestyle='None')

    # Plot prediction
    if y_valid_hat is not None:
        plt.plot(X_valid, y_valid_hat, color=colors[4], linestyle=linestyles[4], label=labels[4])
        plt.plot(X_valid, y_valid_hat, color=colors[4], marker='o', linestyle='None')
    else:
        plt.plot(X_valid, y_valid, color=colors[4], marker='o', linestyle='None')

    plt.legend(loc='best', ncol=legend_cols)
    
    if save_plot is not None:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, format='eps')

    plt.axvline(x = (X_train[-1][0]+X_valid[0][0])/2, color = 'lightgray', linestyle='dashed')

    plt.show()    


### Plot source data
def qae_plot_source_data(
    X_train, y_train, X_valid, y_valid,
    xlim=None, ylim=None, rcParams=(12, 6),
    add_markers=True,
    label_suffix=['', '', ''],
    xlabel='Range', ylabel='Target value (deltas)',
    title=f'Differenced TS Windows for Training and Validation',
    sel_wind=None, save_plot=None):

    # Parameter values
    if rcParams is not None:
        plt.rcParams["figure.figsize"] = rcParams
        
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    # Plot the original time series
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Plot target function
    plt.plot([xt[0] for xt in X_train], [y[0] for y in y_train], color='blue', label='Training')
    plt.plot([xv[0] for xv in X_valid], [y[0] for y in y_valid], color='red', label='Validation')
    if sel_wind != None:
        plt.plot([xt[0] for xt in X_train][sel_wind:sel_wind+2], [y[0] for y in y_train][sel_wind:sel_wind+2], color='magenta', label='Selected Window')
    if add_markers:
        plt.plot([xt[0] for xt in X_train], [y[0] for y in y_train], marker='o', color='lightblue', linestyle='None')
        plt.plot([xv[0] for xv in X_valid], [y[0] for y in y_valid], marker='o', color='pink', linestyle='None')
    plt.legend(loc='lower right', ncol=3)
    if save_plot is not None:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, format='eps')
    plt.show()


### Plot differenced TS
def qae_plot_source_and_noisy_data_unfolded(
    y_train, y_valid, y_train_noisy, y_valid_noisy, wind_step,
    label_suffix=['Training data (with noise)', 'Training data (pure)', 
                  'Validation data (with noise)', 'Validation data (pure)'],
    xlabel='Range', ylabel='Target value (deltas)', xlim=None, ylim=None,
    title=f'TS Windows for Training and Validation',
    save_plot=None, sel_wind=None):

    y_train_set = qae_winds_integ_1(qae_wind_to_dict(y_train, 0, wind_step))
    y_valid_set = qae_winds_integ_1(qae_wind_to_dict(y_valid, (y_train.shape[0]+2)*wind_step, wind_step))
    qae_seq_1_plot(y_train_org_set | y_valid_org_set, ylim=(0, 0.95), 
                   title=f'Pure time series (unfolded)', xlabel='Range', ylabel='Target value', save_plot=f'{FIGURES_PATH}/org_ts_windows.eps')
    return
    

### Plot differenced TS
def qae_plot_source_and_noisy_data(
    y_train, y_valid, y_train_noisy, y_valid_noisy,
    label_suffix=['Training data (with noise)', 'Training data (pure)', 
                  'Validation data (with noise)', 'Validation data (pure)'],
    xlabel='Range', ylabel='Target value (deltas)', xlim=None, ylim=None,
    title=f'TS Windows for Training and Validation',
    save_plot=None, sel_wind=None):
    
    sel_train_range=range(len(y_train))
    # sel_valid_range=range(len(y_train)+len(y_train[0]), len(y_train)+len(y_train[0])+len(y_valid))
    sel_valid_range=range(len(y_train)+3, len(y_train)+len(y_valid)+3)
    
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim: plt.xlim(xlim[0], xlim[1])
    if ylim: plt.ylim(ylim[0], ylim[1])
    
    plt.plot(list(sel_train_range), [y[0] for y in y_train_noisy], color='lightblue', label=label_suffix[0])
    plt.plot(list(sel_train_range), [y[0] for y in y_train], color='blue', marker='.', linestyle='none', label=label_suffix[1])
    plt.plot(list(sel_valid_range), [y[0] for y in y_valid_noisy], color='pink', label=label_suffix[2])
    plt.plot(list(sel_valid_range), [y[0] for y in y_valid], color='red', marker='.', linestyle='none', label=label_suffix[3])
    plt.axvline(x = len(y_train)+1, color = 'lightgray', linestyle='dashed')
    plt.legend(loc='lower center', ncol=4)
    if save_plot:
        plt.savefig(save_plot, format='eps')
    plt.show()


###
### Performance curves with extra details
###

import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

from IPython.display import clear_output

# Exponential Moving Target used to smooth the lines 
def smooth_movtarg(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value        
    return smoothed

### Plot cost
def cost_plot(objective_func_vals, rcParams=(8, 4), yscale='linear', log_interv=1,
                  backplot=False, back_color='linen', smooth_weight=0.9, save_plot=None):
    min_cost = min(objective_func_vals)
    iter = len(objective_func_vals)
    smooth_fn = smooth_movtarg(objective_func_vals, smooth_weight)
    x_of_min = np.argmin(objective_func_vals)
    clear_output(wait=True)
    plt.rcParams["figure.figsize"] = rcParams
    plt.title(f'Cost vs iteration '+('with smoothing ' if smooth_weight>0 else ' '))
    plt.xlabel(f'Iteration (min cost={np.round(min_cost, 4)} @ iter# {x_of_min*log_interv})')
    plt.ylabel("Cost function value")
    plt.axvline(x=x_of_min*log_interv, color="lightgray", linestyle='--')
    plt.yscale(yscale)
    if backplot:
        plt.plot([x*log_interv for x in range(len(objective_func_vals))], objective_func_vals, color=back_color) # lightgray
    plt.plot([x*log_interv for x in range(len(smooth_fn))], smooth_fn, color='black')
    if save_plot is not None:
        plt.savefig(save_plot, format='eps')
    plt.show()

### Plot MAE
def mae_plot(mae_train_vals, mae_valid_vals, rcParams=(8, 4), yscale='linear', 
                  backplot=False, back_color='linen', smooth_weight=0.9, save_plot=None):
    # clear_output(wait=True)
    min_train_mae = min(mae_train_vals)
    min_valid_mae = min(mae_valid_vals)
    smooth_train = smooth_movtarg(mae_train_vals, smooth_weight)
    smooth_valid = smooth_movtarg(mae_valid_vals, smooth_weight)
    x_of_min = np.argmin(mae_valid_vals)
    iter = len(mae_train_vals)
    plt.rcParams["figure.figsize"] = rcParams
    plt.title(f'MAE vs iteration '+('with smoothing ' if smooth_weight>0 else ' ')+
              f'(iter# {iter}, '+
              f'min train MAE={np.round(min_train_mae, 4)}, '+
              f'valid MAE={np.round(min_valid_mae, 4)})')
    plt.xlabel(f'Iteration (min valid MAE @ iter# {x_of_min})')
    plt.ylabel("MAE")
    plt.axvline(x=x_of_min, color="lightgray", linestyle='--')
    plt.yscale(yscale)
    if backplot:
        plt.plot(range(len(mae_train_vals)), mae_train_vals, color=back_color) # powderblue
        plt.plot(range(len(mae_valid_vals)), mae_valid_vals, color=back_color) # mistyrose
    plt.plot(smooth_train, label='Training', color='blue')
    plt.plot(smooth_valid, label='Validation', color='red')
    plt.legend(loc='upper right', ncol=1)
    if save_plot is not None:
        plt.savefig(save_plot, format='eps')
    plt.show()

### Plot and compare various performance plots
#   If log_interv is a number, it applies to all curves
#   If it is a list, each number applies to its curve
def multi_perform_plot(pvals, rcParams=(8, 4), yscale='linear', 
                  backplot=False, smooth_weight=0.9, save_plot=None,
                  title='Performance vs iteration', meas_type='Cost', 
                  meas_min=True, labels=[], back_color='linen',
                  line_styles=None, line_cols=None, col_cycle_rep=5,
                  xlim=None, ylim=None, log_interv=1):
    
    if not pvals: # Empty list of curves
        return
    if type(log_interv) is int:
        log_int_list = [log_interv]*len(pvals)
    elif type(log_interv) is list:
        log_int_list = log_interv
    else:
        print('*** log_interv must be an interger or a list')
        return

    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_cols = prop_cycle.by_key()['color']
    if line_cols is None: line_cols = default_cols*col_cycle_rep
    if line_styles is None: line_styles = ['solid']*len(line_cols)

    iter = max([len(p) for p in pvals])*max(log_int_list)
    plt.rcParams["figure.figsize"] = rcParams
    plt.title(f'{title} '+('with smoothing ' if smooth_weight>0 else ' ')+
              f'(iter# {iter})')
    plt.xlabel(f'Iteration')
    plt.ylabel(meas_type)
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    plt.yscale(yscale)
    for i in range(len(pvals)):
        if backplot:
            plt.plot([x*log_int_list[i] for x in range(len(pvals[i]))], pvals[i], color=back_color)
    for i in range(len(pvals)):
        if meas_min:
            lim = 'min'
            sel_val = np.round(min(pvals[i]), 3)
            sel_x = np.argmin(pvals[i])
        else:
            lim = 'max'
            sel_val = np.round(max(pvals[i]), 3)
            sel_x = np.argmax(pvals[i])
        smooth_vals = smooth_movtarg(pvals[i], smooth_weight)
        sel_lab = labels[i] if labels else f'{i}'
        plt.plot([x*log_int_list[i] for x in range(len(pvals[i]))], smooth_vals, 
                 linestyle=line_styles[i], color=line_cols[i],
                 label=f'{sel_lab}  ({lim} {meas_type}={sel_val} @ iter# {sel_x*log_int_list[i]})')
    plt.legend(loc='best', ncol=1)
    if save_plot is not None:
        plt.savefig(save_plot, format='eps')
    plt.show()

### Plot and compare various performance plots
# def multi_perform_plot(pvals, rcParams=(8, 4), yscale='linear', 
#                   backplot=False, smooth_weight=0.9, save_plot=None,
#                   title='Performance vs iteration', meas_type='Cost', 
#                   meas_min=True, labels=[], back_color='linen',
#                   xlim=None, ylim=None, log_interv=1):
#     if not pvals: # Empty list of curves
#         return
#     iter = max([len(p) for p in pvals])
#     plt.rcParams["figure.figsize"] = rcParams
#     plt.title(f'{title} '+('with smoothing ' if smooth_weight>0 else ' ')+
#               f'(iter# {iter*log_interv})')
#     plt.xlabel(f'Iteration')
#     plt.ylabel(meas_type)
#     if xlim is not None: plt.xlim(xlim)
#     if ylim is not None: plt.ylim(ylim)
#     plt.yscale(yscale)
#     for i in range(len(pvals)):
#         if backplot:
#             plt.plot([x*log_interv for x in range(len(pvals[i]))], pvals[i], color=back_color)
#     for i in range(len(pvals)):
#         if meas_min:
#             lim = 'min'
#             sel_val = np.round(min(pvals[i]), 3)
#             sel_x = np.argmin(pvals[i])
#         else:
#             lim = 'max'
#             sel_val = np.round(max(pvals[i]), 3)
#             sel_x = np.argmax(pvals[i])
#         smooth_vals = smooth_movtarg(pvals[i], smooth_weight)
#         sel_lab = labels[i] if labels else f'{i}'
#         plt.plot([x*log_interv for x in range(len(pvals[i]))], smooth_vals, label=f'{sel_lab}  ({lim} {meas_type}={sel_val} @ iter# {sel_x*log_interv})')
#     plt.legend(loc='best', ncol=1)
#     if save_plot is not None:
#         plt.savefig(save_plot, format='eps')
#     plt.show()

