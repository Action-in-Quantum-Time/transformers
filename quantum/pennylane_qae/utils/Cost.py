# Support functions for TS metrics
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for quantum time series analysis
# Date: 2024

### Cost function
#   This cost function can support several different cost types (and observables).
#   As the model is being optimised the cost plot will be produced continuously.
#   If model testing is initialised, the training and validation MAE are also calculated and plotted.
#   Model training with simultaneous testing is slow!
#
#   Note:
#   - As all intermediate parameters will be saved, the performance metrics can be calculated later.
#
#   To do:
#   - Modify to allow cost+parameters to be saved at some intervals only.

import sys
sys.path.append('.')
sys.path.append('..')
sys.path

import numpy as np
import math
import random
import time
import datetime
import warnings
import matplotlib.pyplot as plt
from IPython.display import clear_output


##### Utilities

### Converts numbers to their binary representation
def digit_list(a, n_bits):
    return np.array([int(i) for i in f'{a:0{n_bits}b}'])

### Calculates an inner join on binary representation
def inner_join(a, b, n_bits):
    inna = digit_list(a, n_bits)
    innb = digit_list(b, n_bits)
    return np.inner(inna, innb)

### Makes an array indexed by number and 
#   returns the number of binary ones in its representation
def make_1s_count(bin_digs):
    dig_arr = [0]*bin_digs
    for i in range(bin_digs):
        dig_arr[i] = sum(digit_list(i, bin_digs))
    return np.array(dig_arr)


##### Manipulation of measurement probabilities

### Takes a list of probabilities (add to 1) and presents them in tabular form
def report_c2p(probs):
    qubit_no = int(np.log2(len(probs)))
    qa = np.array([[int(bit) for bit in format(qn, f'#0{qubit_no+2}b')[2:]] for qn in range(len(probs))])
    qr = np.zeros((qa.shape[0],qa.shape[1]+1))
    qr[:,:-1] = qa
    qr[:,qa.shape[1]:qa.shape[1]+1] = np.array([probs]).transpose()
    return qr

### Converts results of circuit state_vector measurements to 
#   individual qubit measurements
#   probs: a list of circuit measurement probabilities (must be of length 2^n)
#   returns: all qubit Z projective measurements onto |0>
def cprobs_to_qprobs(probs):
    qubit_no = int(np.log2(len(probs)))
    qpos = [[int(bit) for bit in format(qn, f'#0{qubit_no+2}b')[2:]] for qn in range(len(probs))]
    # qarr = np.array(qpos)
    qprobs = np.zeros(qubit_no)
    for qp in range(len(qpos)):
        for q in range(qubit_no):
            qprobs[q] += qpos[qp][q]*probs[qp]
    return qprobs

### Converts results of circuit state vector measurements to qubit angular positions
#   probs: a list of circuit measurement probabilities (must be of length 2^n)
#   returns 0: a list of qubit angular positions
#           1: a list of individual probabilities of qubit cast to |0> (towards positive values)
def cprobs_to_qangles(probs):
    qubit_0_probs = cprobs_to_qprobs(probs)
    qubit_0_probs = [1-x for x in qubit_0_probs] # compatibility with single_qubit_angle_meas

    all_q_meas = []
    for q in range(len(qubit_0_probs)):
        p0 = qubit_0_probs[q]
        p1 = 1 - p0
        amp0 = np.sqrt(p0)
        amp1 = np.sqrt(p1)
        meas_angle = 2*np.arccos(amp0)-np.pi/2
        all_q_meas.append(meas_angle)
    return all_q_meas, qubit_0_probs

### Single qubit measurement in terms of its angular position
#   Recall the figure explaining encoding
#   *** Should use encoding / decoding from Angles

from qiskit_aer.backends import AerSimulator
from qiskit_aer import Aer

def single_qubit_angle_meas(qc, backend, shots=10000):
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    
    counts0 = counts['0'] if '0' in counts.keys() else 0
    counts1 = counts['1'] if '1' in counts.keys() else 0
    p0 = counts0/(counts0+counts1)
    p1 = counts1/(counts0+counts1)
    amp0 = np.sqrt(p0)
    amp1 = np.sqrt(p1)

    meas_angle = 2*np.arccos(amp0)-np.pi/2
    return meas_angle, p0

### Multi-quibit measurement of individual qubits in terms of their angular position
#   Assumes a statevector backend and the state saved in the circuit
#   Recall the figure explaining encoding
#   qc: Circuit to be measured
#   backend: Backend to be used for measuring the circuit
#   shots (optional): The number of shots in execution
#   returns 0: a list of qubit angular positions
#           1: a list of individual probabilities of qubit cast to |0> (towards positive values)
#   *** Should use encoding / decoding from Angles

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector

def multi_qubit_angle_meas(qc, backend, shots=10000):
    result = backend.run(qc, shots=shots).result()
    qc_state = result.get_statevector()
    probs = Statevector(qc_state).reverse_qargs().probabilities(qargs=range(qc.num_qubits))
    qubit_0_probs = cprobs_to_qprobs(probs)
    qubit_0_probs = [1-x for x in qubit_0_probs] # compatibility with single_qubit_angle_meas

    all_q_meas = []
    for q in range(len(qubit_0_probs)):
        p0 = qubit_0_probs[q]
        p1 = 1 - p0
        amp0 = np.sqrt(p0)
        amp1 = np.sqrt(p1)
        meas_angle = 2*np.arccos(amp0)-np.pi/2
        all_q_meas.append(meas_angle)
    return all_q_meas, qubit_0_probs


##### Objective functions

### Calculates cost which is the probability of all measurements to be zero (measurement of 1)
#   If the repeated measurement of SWAP test returns 1 then the two states are identical (or highly similar). 
#   If the repeated measurement returns 0.5 then the two states are orthogonal (or highly dissimilar).
def cost_swap(pvals, probs):
    recs = probs.shape[0]
    return np.sum(probs[:, 1]) / recs

### Calculates cost which is 1 - the probability of measuring all qubits to be zero (|0>^n)
def cost_zero(pvals, probs):
    recs = probs.shape[0]
    return 1.0 - np.sum(probs[:, 0]) / recs

### Calculates cost which is 1 - the probability of measuring all qubits to be zero (|0>^n)
def cost_neg_zero(pvals, probs):
    recs = probs.shape[0]
    return -(1.0 - np.sum(probs[:, 0]) / recs)

### Calculates cost which is 1 - the sum of log(probability) of measuring qubits to be zero (|0>^n)
def cost_zero_log(pvals, probs):
    small_val = 0.0001
    recs = probs.shape[0]
    return -np.sum([(np.log(x) if x > small_val else small_val) for x in probs[:, 0]]) / recs

### Calculates the cost which is the weighted number of 1s per measurement
def cost_min1s(pvals, probs):
    digit_no = int(np.log2(probs.shape[1]+1))
    recs = probs.shape[0]
    digits_wsum = 0
    for rec in range(recs):
        for i in range(digit_no):
            digits_wsum += sum(digit_list(i, digit_no))*probs[rec, i]
    return digits_wsum


##### Higher-level objective functions (compare expected output with the predicted as calculated from probabilities)

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

### Collect all expected and predicted values
def collect_objfun_vals(expected_vals, probs):
    pred_vals = []
    exp_vals = []
    vals_no = 0
    for i in range(len(probs)):
        pred_vals += list(cprobs_to_qangles(probs[i])[0])
        exp_vals += list(expected_vals[i])
        vals_no += len(expected_vals[i])
    return np.array(exp_vals), np.array(pred_vals), vals_no

### Find the score between expected values and predicted QAE values as calculated from the results probabilities

def obj_fun_mae(expected_vals, probs):
    exp_vals, pred_vals, vals_no = collect_objfun_vals(expected_vals, probs)
    return mean_absolute_error(exp_vals, pred_vals)

def obj_fun_mse(expected_vals, probs):
    exp_vals, pred_vals, vals_no = collect_objfun_vals(expected_vals, probs)
    return mean_squared_error(exp_vals, pred_vals)

def obj_fun_rmse(expected_vals, probs):
    exp_vals, pred_vals, vals_no = collect_objfun_vals(expected_vals, probs)
    return np.sqrt(mean_squared_error(exp_vals, pred_vals))

def obj_fun_r2(expected_vals, probs):
    exp_vals, pred_vals, vals_no = collect_objfun_vals(expected_vals, probs)
    return 1-r2_score(exp_vals, pred_vals)



# make the plot nicer
plt.rcParams["figure.figsize"] = (12, 6) 


##### Cost classes - Cost and Cost_Flex

### Class Cost
class Cost:

    # Class constants
    name = "Cost class"

    type_min1s = 'min1s'
    type_swap = 'swap'
    type_zeros = 'zeros'
    type_negzeros = 'negzeros'
    type_zerolog = 'zerolog'
    
    feedback_plot = 'plot'
    feedback_print = 'print'
    feedback_none = 'none'
    
    yscale_linear = 'linear'
    yscale_log = 'log'
    yscale_asinh = 'asinh'
    yscale_logit = 'logit' # + function, functionlog, symlog
            
    # Initialises the costs
    def __init__(self, train_set, qnn, optimizer, init_vals, 
                 epochs=None, shuffle_interv=0, log_interv=1,
                 feedback='plot', cost_type='swap', yscale='linear', 
                 rand=None, prompt=None, print_page=20
                ):

        # All important variables
        self.train_set = np.copy(train_set)
        self.shuffle_interv = shuffle_interv
        self.shuffled = '*'
        self.log_interv = log_interv
        self.objective_func_vals = []
        self.params = []
        self.qnn = qnn
        self.opt = optimizer
        self.init_vals = init_vals
        self.iter = -1
        self.rand = rand
        self.init_time = time.time()
        self.elapsed_time = 0
        self.feedback = feedback
        self.cost_type = cost_type
        self.yscale = yscale
        self.perform_tests=False
        self.epochs = epochs
        self.print_page = print_page
        self.prompt = '' if prompt is None else prompt+' '

        if self.rand is None:
            self.rand = int(self.init_time) % 10000
        np.random.seed(self.rand)

        if self.shuffle_interv>0:
            np.random.shuffle(self.train_set)            
            self.shuffled = '*'
        else:
            self.shuffled = ' '

    ### Reset objective function values
    def reset(self):
        self.iter = -1
        self.objective_func_vals = []
        self.params = []
        self.elapsed_time = 0
        self.init_time = time.time()

    ### Plot cost
    def cost_plot(self, col='black', title=None, xlabel='Iteration', ylabel='Cost function value', rcParams=(12, 6)):
        min_cost = min(self.objective_func_vals)
        min_x = np.argmin(self.objective_func_vals)*self.log_interv
        clear_output(wait=True)
        plt.rcParams["figure.figsize"] = rcParams
        if title:
            plt.title(title)
        else:
            time_str = str(datetime.timedelta(seconds=int(self.elapsed_time)))
            info = f'Cost vs iteration ({self.prompt}iter# {self.shuffled}{self.iter}, '+ \
                   ('' if self.epochs is None or self.epochs == 0 else f' ({int(self.iter / self.epochs * 100)}%), ')+ \
                   f'min cost={np.round(min_cost, 4)} @ iter {min_x}, '+ \
                   f'time={time_str})'
            plt.title(info)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale(self.yscale)
        plt.plot(range(0, self.log_interv*len(self.objective_func_vals), self.log_interv), self.objective_func_vals, color=col)
        plt.show()

    ### Print cost
    def cost_print(self):
        if len(self.objective_func_vals) > 0:
            if self.iter % self.print_page == 0:
                clear_output(wait=True)
            min_cost = min(self.objective_func_vals)
            min_x = np.argmin(self.objective_func_vals)*self.log_interv
            curr_cost = self.objective_func_vals[-1]
            comp_symb = '<' if min_cost < curr_cost else '='
            time_str = str(datetime.timedelta(seconds=int(self.elapsed_time)))
            info = f'{self.prompt}iter# {self.shuffled}{self.iter}'+ \
                   ('' if self.epochs is None or self.epochs == 0 else f' ({int(self.iter / self.epochs * 100)}%)')+ \
                   f', time:{time_str}'+ \
                   f', min cost {np.round(min_cost, 4)} @ iter {min_x} {comp_symb} cost {np.round(curr_cost, 4)} '
            print(info)

    ### Cost function used in training
    def cost_fun(self, params_values, *args):
        self.iter = self.iter+1

        # Shuffle the data set if needed
        if (self.shuffle_interv>0) and (self.iter % self.shuffle_interv == 0):
            np.random.shuffle(self.train_set)
            self.shuffled = '*'
        else:
            self.shuffled = ' '

        # Perform one step forward
        probs = self.qnn.forward(self.train_set, params_values)

        # Calculate the cost of the improved model
        if self.cost_type == Cost.type_swap:
            cost = cost_swap(None, probs)
        elif self.cost_type == Cost.type_min1s:
            cost = cost_min1s(None, probs)
        elif self.cost_type == Cost.type_zerolog:
            cost = cost_zero_log(None, probs)
        elif self.cost_type == Cost.type_negzeros:
            cost = cost_neg_zero(None, probs)
        # elif self.cost_type == Cost.type_zeros:
        #     cost = cost_zero(None, probs)
        else: # it is Cost.type_zeros
            cost = 1.0 - np.sum(probs[:, 0]) / probs.shape[0]

        # Calculate elapsed time so far
        self.elapsed_time = time.time() - self.init_time
    
        # Save and report the cost and parameters as needed
        if (self.iter % self.log_interv == 0):
            self.objective_func_vals.append(cost)
            self.params.append(params_values)
            
            # Feedback on the model performance
            if self.feedback == Cost.feedback_plot:
                self.cost_plot()
            elif self.feedback == Cost.feedback_print:
                self.cost_print()
            # Else it is Cost.feedback_none

        return cost

    def optimize(self):
        return self.opt.minimize(fun=self.cost_fun, x0=self.init_vals)


from sklearn.utils import shuffle

### Flexible Cost class 
class Cost_Flex:

    # Class constants
    name = "Cost_Flex class"

    type_swap = 'swap'
    type_zeros = 'zeros'
    
    feedback_plot = 'plot'
    feedback_print = 'print'
    feedback_none = 'none'
    
    yscale_linear = 'linear'
    yscale_log = 'log'
    yscale_asinh = 'asinh'
    yscale_logit = 'logit' # + function, functionlog, symlog
            
    # Initialises the costs
    def __init__(self, train_X, train_Y, qnn, optimizer, init_vals, 
                 epochs=None, shuffle_interv=0, log_interv=1,
                 feedback='plot', yscale='linear', rand=None, prompt=None, 
                 print_page=20, cost_type='zeros', obj_fun=None
                ):

        # All important variables
        self.train_X = train_X                  # Training set features, e.g. noisy data
        self.train_Y = train_Y                  # Training set output, e.g. pure data
        self.shuffle_interv = shuffle_interv    # Number of iterations between shuffles, none when 0
        self.shuffled = '*'                     # Shuffling indicator on output
        self.log_interv = log_interv            # Number of iterations between results logging - cost and parameters
        self.obj_fun_vals = []                  # A list of objective function values to be saved
        self.params = []                        # A list of model parameters to be saved
        self.qnn = qnn                          # QNN model to be used
        self.opt = optimizer                    # Optimiser to be used
        self.init_vals = init_vals              # Initial parameter values for the optimiser
        self.iter = -1                          # Iteration counter, starts with -1
        self.rand = rand                        # Seed for the random number generator
        self.init_time = time.time()            # Timer
        self.elapsed_time = 0                   # Optimisation elapsed time
        self.feedback = feedback                # Feedback prompt to be displayed during the optimisation
        self.cost_type = cost_type              # Legacy "hard-wired" cost type as a string ("swap" and "zeros" built in)
        self.obj_fun = obj_fun                  # Objective function, takes expect vals and probs of different outcomes
        self.yscale = yscale                    # Type of y-scale for plotting, e.g. linear or log
        self.epochs = epochs                    # Number of optimisation iterations tobe executed
        self.print_page = print_page            # Number of iterations between print feedback to be given
        self.prompt = ''                        # Short prompt leading any form of feedback given

        # Default prompt is empty string
        self.prompt = '' if prompt is None else prompt+' '

        # Legacy default objective functions
        if obj_fun is None:
            if self.cost_type == Cost.type_swap:
                obj_fun = cost_swap
            elif self.cost_type == Cost.type_zeros:
                obj_fun = cost_zero

        if self.rand is None:
            self.rand = int(self.init_time) % 10000
        np.random.seed(self.rand)

        if self.shuffle_interv>0:
            self.train_X, self.train_Y = shuffle(self.train_X, self.train_Y)            
            self.shuffled = '*'
        else:
            self.shuffled = ' '

    ### Reset objective function values
    def reset(self):
        self.iter = -1
        self.obj_fun_vals = []
        self.params = []
        self.elapsed_time = 0
        self.init_time = time.time()

    ### Plot cost
    def cost_plot(self, col='black', title=None, xlabel='Iteration', ylabel='Cost function value', rcParams=(12, 6)):
        min_cost = min(self.obj_fun_vals)
        min_x = np.argmin(self.obj_fun_vals)*self.log_interv
        clear_output(wait=True)
        plt.rcParams["figure.figsize"] = rcParams
        if title:
            plt.title(title)
        else:
            time_str = str(datetime.timedelta(seconds=int(self.elapsed_time)))
            info = f'Cost vs iteration ({self.prompt}iter# {self.shuffled}{self.iter}, '+ \
                   ('' if self.epochs is None or self.epochs == 0 else f' ({int(self.iter / self.epochs * 100)}%), ')+ \
                   f'min cost={np.round(min_cost, 4)} @ iter {min_x}, '+ \
                   f'time={time_str})'
            plt.title(info)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale(self.yscale)
        plt.plot(range(0, self.log_interv*len(self.obj_fun_vals), self.log_interv), self.obj_fun_vals, color=col)
        plt.show()

    ### Print cost
    def cost_print(self):
        if len(self.obj_fun_vals) > 0:
            if self.iter % self.print_page == 0:
                clear_output(wait=True)
            min_cost = min(self.obj_fun_vals)
            min_x = np.argmin(self.obj_fun_vals)*self.log_interv
            curr_cost = self.obj_fun_vals[-1]
            comp_symb = '<' if min_cost < curr_cost else '='
            time_str = str(datetime.timedelta(seconds=int(self.elapsed_time)))
            info = f'{self.prompt}iter# {self.shuffled}{self.iter}'+ \
                   ('' if self.epochs is None or self.epochs == 0 else f' ({int(self.iter / self.epochs * 100)}%)')+ \
                   f', time:{time_str}'+ \
                   f', min cost {np.round(min_cost, 4)} @ iter {min_x} {comp_symb} cost {np.round(curr_cost, 4)} '
            print(info)

    ### Cost function used in training
    def cost_fun(self, params_values, *args):
        self.iter = self.iter+1

        # Shuffle the data set if needed
        if (self.shuffle_interv>0) and (self.iter % self.shuffle_interv == 0):
            self.train_X, self.train_Y = shuffle(self.train_X, self.train_Y)            
            self.shuffled = '*'
        else:
            self.shuffled = ' '

        # Perform one step forward
        probs = self.qnn.forward(self.train_X, params_values)

        # Calculate the cost of the improved model
        # By comparing the expected values vs predicted from probabilities
        cost = self.obj_fun(self.train_Y, probs)

        # Calculate elapsed time so far
        self.elapsed_time = time.time() - self.init_time
    
        # Save and report the cost and parameters as needed
        if (self.iter % self.log_interv == 0):
            self.obj_fun_vals.append(cost)
            self.params.append(params_values)
            
            # Feedback on the model performance
            if self.feedback == Cost.feedback_plot:
                self.cost_plot()
            elif self.feedback == Cost.feedback_print:
                self.cost_print()
            # Else it is Cost.feedback_none

        return cost

    def optimize(self):
        return self.opt.minimize(fun=self.cost_fun, x0=self.init_vals)

# Regressor callback - handling costs for NeuralNetworkRegressor
class Regr_callback:
    name = "Regr_callback"
    
    # Initialises the callback
    def __init__(self, log_interval=50):
        self.objfun_vals = []
        self.params_vals = []
        self.log_interval = log_interval

    # Initialise callback lists
    # - For some reason [] defaults not always work (bug?)
    def reset(self, obfun=[], params=[]):
        self.objfun_vals = obfun
        self.params_vals = params

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
        best_val = self.min_obj()
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.title(f'Objective function value (min: {np.round(best_val[1], 4)} @ {best_val[0]})')
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(range(len(self.objfun_vals)), self.objfun_vals, color="blue")
        plt.show()

    # Callback function to store objective function values and plot
    def graph(self, weights, obj_func_eval):
        self.objfun_vals.append(obj_func_eval)
        self.params_vals.append(weights)
        self.plot()
            
    # Callback function to store objective function values but not plot
    def collect(self, weights, obj_func_eval):
        self.objfun_vals.append(obj_func_eval)
        self.params_vals.append(weights)
        current_batch_idx = len(self.objfun_vals)
        if current_batch_idx % self.log_interval == 0:
            prev_batch_idx = current_batch_idx-self.log_interval
            last_batch_min = np.min(self.objfun_vals[prev_batch_idx:current_batch_idx])
            print('Regr callback(', prev_batch_idx, ', ', current_batch_idx,') = ', last_batch_min)
