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
from matplotlib.lines import Line2D
from IPython.display import clear_output

from utils.TS import *


#####
##### Windows integration
#####


### Convert a TS to its dictionary form
def itg_wind_to_dict(wind_ts, start, step):
    wind_set = {}
    for wind_idx in range(len(wind_ts)):
        wind_x = wind_idx * step
        wind_set[wind_x+start] = wind_ts[wind_idx]
    return wind_set


### Integrates QAE results stored in a single set of TS widows into a single sequence
#   Note: When windows overap their average values will be returned
#         When windows are too far apart, separate sub-sequences will be returned
#   wind_set: The selected set of TS windows
#   trim_left: The number of values to be trimmed from the left edge of each window
#   trim_right:  The number of values to be trimmed from the right edge of each window

def itg_winds_integ_1(wind_set, trim_left=0, trim_right=0):

    # Collect all overalapping values into lists attached to individual data points
    vals = {}
    for k in sorted(wind_set.keys()):
        val_list = wind_set[k]
        list_len = len(val_list)
        for i in range(list_len):
            if (i < trim_left) or (i >= list_len - trim_right):
                None # Skip these trimmed values
            else:
                val_idx = k+i
                if val_idx in vals:
                    vals[val_idx].append(val_list[i])
                else:
                    vals[val_idx] = [val_list[i]]

    # Collapse all consecutive values into subsequences
    # All values apart start new subsequences
    seq = {}
    next_key = -3
    prev_key = 0
    for k in sorted(vals.keys()):
        next_key += 1
        if k == next_key:
            seq[prev_key].append(np.average(vals[k]))
        else:
            prev_key = k
            next_key = k
            seq[k] = [np.average(vals[k])]
            
    return seq


### Integrates QAE results stored as TS widows into a single sequence
#   Note: When windows overap their average values will be returned
#         When windows are too far apart, separate sub-sequences will be returned
#   in_org_set: The selected set of original TS windows
#   in_meas_set: The set of measurements of the original values in TS windows
#   out_recons_set: The set of QAE reconstructions for each original TS window
#   trim_left: The number of values to be trimmed from the left edge of each window
#   trim_right:  The number of values to be trimmed from the right edge of each window

def itg_winds_integ(in_org_set, in_meas_set, out_recons_set, trim_left=0, trim_right=0):
    in_org_seq = itg_winds_integ_1(in_org_set, trim_left, trim_right)
    in_meas_seq = itg_winds_integ_1(in_meas_set, trim_left, trim_right)
    out_recons_seq = itg_winds_integ_1(out_recons_set, trim_left, trim_right)
    return in_org_seq, in_meas_seq, out_recons_seq

### Test window integration
# in_org_seq, in_meas_seq, out_recons_seq = itg_winds_integ(in_org_set, in_meas_set, out_reconstr_set, trim_left, trim_right)
# [print((k, out_reconstr_set[k])) for k in sorted(out_reconstr_set.keys())]
# print()
# vals = itg_winds_integ_1(out_recons_seq)
# [print((k, vals[k])) for k in sorted(vals.keys())]


#####
##### Integrated windows plotting
#####


### Returns the x range for the sequence set
def itg_seq_x_range(start, subseq):
    return range(start, start+len(subseq))


### Plots a single sequences (may consist of separate subsets)
def itg_seq_1_plot(in_org_seq, marker_points=None, markers_on_top=True, vert_line=True, line_offset=2,
                 labels=['Training', 'Validation', 'Error', '', '', ''],
                 xlim=None, ylim=None, rcParams=(12, 6), save_plot=None,
                 cols = ['royalblue', 'red', 'orange', 'green', 'brown', 'black'],
                 marker_cols = ['cornflowerblue', 'pink', 'lightorange', 'lightgreen', 'lightbrown', 'gray'],
                 xlabel='Selected data points', ylabel='Inter-point differences',
                 title=f'Original and measured input vs reconstructed data'):
    
    sorted_keys = sorted(in_org_seq.keys())
    
    # Plot prepared data
    plt.rcParams["figure.figsize"] = rcParams
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.xlim(lb, ub)

    sub_seq = -1
    for sel_wind in sorted_keys:
        
        sub_seq += 1
        org_in = in_org_seq[sel_wind]

        if sub_seq > 0 and vert_line:
            plt.axvline(x = sel_wind-line_offset, color = 'lightgray', linestyle='dashed')
        
        # Plot markers first
        if marker_points is not None and not markers_on_top:
            markers = marker_points[sel_wind]
            plt.plot(itg_seq_x_range(sel_wind, markers), markers, marker='.', color=marker_cols[sub_seq], linestyle='None')
        
        # Plot target function
        plt.plot(itg_seq_x_range(sel_wind, org_in), org_in, color=cols[sub_seq], label=labels[sub_seq])

        # Plot markers last
        if marker_points is not None and markers_on_top:
            markers = marker_points[sel_wind]
            plt.plot(itg_seq_x_range(sel_wind, markers), markers, marker='.', color=marker_cols[sub_seq], linestyle='None')
        
    plt.legend(loc='lower center', ncol=2) # , bbox_to_anchor=(0.5, -0.2)handles=handles, fancybox=True, shadow=True
    if save_plot is not None:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, format='eps')
    plt.show()

    print('\n')
    return

### Plot differenced TS
def itg_seq_n_plot(
    y_train, y_valid, y_train_noisy_ls, y_valid_noisy_ls,
    labels=['Training', 'Validation'],
    xlabel='Range', ylabel='Target value (deltas)', xlim=None, ylim=None,
    title=f'TS Windows for Training and Validation',
    cols = {'train_pure':'blue', 'train_noisy':'lightblue', 'valid_pure':'red', 'valid_noisy':'pink'},
    line_styles={'train_pure':'solid', 'train_noisy':'none', 'valid_pure':'solid', 'valid_noisy':'none'},
    marker_line_cols=['cornflowerblue', 'orange', 'tomato'], marker_style='.',
    save_plot=None, sel_wind=None):
    
    sel_train_range=range(len(y_train))
    sel_valid_range=range(len(y_train)+3, len(y_train)+len(y_valid)+3)
  
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim: plt.xlim(xlim[0], xlim[1])
    if ylim: plt.ylim(ylim[0], ylim[1])
    
    for i in range(len(y_train_noisy_ls)):
        plt.plot(list(sel_train_range), y_train_noisy_ls[i], color=cols['train_noisy'], marker=marker_style, linestyle=line_styles['train_noisy'])
        
    for i in range(len(y_valid_noisy_ls)):
        plt.plot(list(sel_valid_range), y_valid_noisy_ls[i], color=cols['valid_noisy'], marker=marker_style, linestyle=line_styles['valid_noisy'])
    
    plt.plot(list(sel_train_range), y_train, color=cols['train_pure'], linestyle=line_styles['train_pure'], label=labels[0])
    plt.plot(list(sel_valid_range), y_valid, color=cols['valid_pure'], linestyle=line_styles['valid_pure'], label=labels[1])

    plt.axvline(x = len(y_train)+1, color = 'lightgray', linestyle='dashed')
    plt.legend(loc='lower left', ncol=4)
    if save_plot:
        plt.savefig(save_plot, format='eps')
    plt.show()

### Plots all three sequences (may consist of separate subsets)
def itg_seq_plot(in_org_seq, in_meas_seq, out_recons_seq, 
                 add_markers=False,
                 line_styles=['solid', 'solid', 'solid'], marker_line_style='None',
                 line_cols=['royalblue', 'darkorange', 'red'],
                 marker_line_cols=['cornflowerblue', 'orange', 'tomato'],
                 labels=['Original data', 'Measured data', 'Reconstructed data'],
                 label_suffix=['', '', ''],
                 xlim=None, ylim=None, rcParams=(12, 6), save_plot=None,
                 xlabel='Selected data points', ylabel='Inter-point differences',
                 title=f'Original and measured input vs reconstructed data'):
    
    sorted_keys = sorted(in_org_seq.keys())
    
    # Plot prepared data
    plt.rcParams["figure.figsize"] = (12, 6)
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
        
    for sel_wind in sorted_keys:

        org_in = in_org_seq[sel_wind]
        org_meas = in_meas_seq[sel_wind]
        out_meas = out_recons_seq[sel_wind]    
        
        # Plot target function
        plt.plot(itg_seq_x_range(sel_wind, org_in), org_in, linestyle=line_styles[0], color=line_cols[0])
        if add_markers:
            plt.plot(itg_seq_x_range(sel_wind, org_in), org_in, marker='o', color=marker_line_cols[0], linestyle=marker_line_style)
        plt.plot(itg_seq_x_range(sel_wind, org_meas), org_meas, linestyle=line_styles[1], color=line_cols[1])
        if add_markers:
            plt.plot(itg_seq_x_range(sel_wind, org_meas), org_meas, marker='o', color=marker_line_cols[1], linestyle=marker_line_style)
        plt.plot(itg_seq_x_range(sel_wind, out_meas), out_meas, linestyle=line_styles[2], color=line_cols[2])
        if add_markers:
            plt.plot(itg_seq_x_range(sel_wind, out_meas), out_meas, marker='o', color=marker_line_cols[2], linestyle=marker_line_style)
        
    # access legend objects automatically created from data
    handles, labs = plt.gca().get_legend_handles_labels()
    
    # create manual symbols for legend
    line_org = Line2D([0], [0], label=f'{labels[0]} {label_suffix[0]}', color=line_cols[0], linestyle=line_styles[0])
    line_meas = Line2D([0], [0], label=f'{labels[1]} {label_suffix[1]}', color=line_cols[1], linestyle=line_styles[1])
    line_recon = Line2D([0], [0], label=f'{labels[2]} {label_suffix[2]}', color=line_cols[2], linestyle=line_styles[2])
    
    # add manual symbols to auto legend
    handles.extend([line_org, line_meas, line_recon])
    
    # plt.legend(handles=handles)    
    # plt.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.2),
    #            ncol=3, fancybox=True, shadow=True)
    plt.legend(handles=handles, loc='best', ncol=3)
    if save_plot is not None:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, format='eps')
    plt.show()

    print('\n')
    return

### Plot differenced and augmented windows - pure and noisy, plus repeated noisy curves
def itg_diffaug_plot(y_train_enc, y_valid_enc, y_train_noisy_enc, y_valid_noisy_enc, wind_step, noise, noise_reps,
                     labels=['Training', 'Validation'], xlabel='Range', ylabel='Target value (deltas)', 
                     line_styles={'train_pure':'solid', 'train_noisy':'none', 'valid_pure':'solid', 'valid_noisy':'none'},
                     cols = {'train_pure':'blue', 'train_noisy':'lightblue', 'valid_pure':'red', 'valid_noisy':'pink'},
                     marker_style='.', xlim=None, ylim=None, title=None, save_plot=None, sel_wind=None):
    
    y_train_diff_set = itg_winds_integ_1(itg_wind_to_dict(y_train_enc, 0, wind_step))
    y_valid_diff_set = itg_winds_integ_1(itg_wind_to_dict(y_valid_enc, (y_train_enc.shape[0]+2)*wind_step, wind_step))
    
    #train_pure_start = sorted(y_train_diff_set.keys())[0]
    train_pure_vals = y_train_diff_set[sorted(y_train_diff_set.keys())[0]]
    
    #valid_pure_start = sorted(y_valid_diff_set.keys())[0]
    valid_pure_vals = y_valid_diff_set[sorted(y_valid_diff_set.keys())[0]]
    
    ### Generate noisy sets for differenced TSs
    
    y_train_noisy_ts = []; y_valid_noisy_ts = []
    y_train_noisy_flats = []; y_valid_noisy_flats = []
    
    ### Collect flattened TSs
    for ni in range(noise_reps):
        y_train_diff_noisy_set = itg_winds_integ_1(itg_wind_to_dict(y_train_noisy_enc[ni], 0, wind_step))
        y_valid_diff_noisy_set = itg_winds_integ_1(itg_wind_to_dict(y_valid_noisy_enc[ni], (y_train_noisy_enc[ni].shape[0]+2)*wind_step, wind_step))
    
        train_noisy_vals = np.array(y_train_diff_noisy_set[sorted(y_train_diff_noisy_set.keys())[0]])
        y_train_noisy_flats.append(train_noisy_vals)
        
        valid_noisy_vals = np.array(y_valid_diff_noisy_set[sorted(y_valid_diff_noisy_set.keys())[0]])
        y_valid_noisy_flats.append(valid_noisy_vals)
    
    
    # Calculate amount of noise in differenced TSs (and last noise)
    calc_noise, noise_sum, points_no = gen_calculated_noise(y_train_enc, y_train_noisy_enc[0]) 
    print(f'\n\nTS noise = {np.round(noise*100, 2)}%, calculated noise = {np.round(calc_noise*100, 2)}%, '+
          f'total noise = {np.round(noise_sum, 4)}, over the points = {points_no}\n')

    if title is None: 
        title=f'Diff and augmented TSs (flattened), plus {noise_reps} noisy series with estimated noise of {np.round(calc_noise*100, 2)}% (dots)'
    
    # Plot all TS with noise
    itg_seq_n_plot(train_pure_vals, valid_pure_vals, y_train_noisy_flats, y_valid_noisy_flats,
                   title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, labels=labels,
                   cols=cols, line_styles=line_styles, marker_style=marker_style, save_plot=save_plot)

### Plot differenced and augmented windows - pure and noisy, plus repeated noisy curves
def itg_integrated_plot(y_train_enc, y_valid_enc, y_train_noisy_enc, y_valid_noisy_enc, wind_step, noise,
                     labels=['Training', 'Validation'], xlabel='Range', ylabel='Target value (deltas)', 
                     line_styles={'train_pure':'solid', 'train_noisy':'none', 'valid_pure':'solid', 'valid_noisy':'none'},
                     cols = {'train_pure':'blue', 'train_noisy':'lightblue', 'valid_pure':'red', 'valid_noisy':'pink'},
                     marker_style='.', xlim=None, ylim=None, title=None, save_plot=None, sel_wind=None):
    itg_diffaug_plot(y_train_enc, y_valid_enc, [y_train_noisy_enc], [y_valid_noisy_enc], wind_step, noise, 1,
         labels=labels, xlabel=xlabel, ylabel=ylabel, line_styles=line_styles, cols = cols,
         marker_style=marker_style, xlim=xlim, ylim=ylim, title=title, save_plot=save_plot)   
