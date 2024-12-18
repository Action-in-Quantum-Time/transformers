# Support functions for TS project
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for quantum time series analysis
# Date: 2024

##### Initial settings

import sys
sys.path.append('.')
sys.path.append('..')
sys.path

import matplotlib.pyplot as plt
import pylab
import math
from IPython.display import clear_output

from matplotlib import set_loglevel
set_loglevel("error")


##### Libraries used in QAE development

import pennylane as qml
from pennylane import numpy as np
import torch


##### Utilities

from utils.PennyLane import *

### Draw a circuit
#   Lots of styles apply, e.g. 'black_white', 'black_white_dark', 'sketch', 
#     'pennylane', 'pennylane_sketch', 'sketch_dark', 'solarized_light', 'solarized_dark', 
#     'default', we can even use 'rcParams' to redefine all attributes

def draw_circuit(circuit, fontsize=20, style='pennylane', 
                 scale=None, title=None, decimals=2, level=None):
    def _draw_circuit(*args, **kwargs):
        nonlocal circuit, fontsize, style, scale, title, level
        qml.drawer.use_style(style)
        fig, ax = qml.draw_mpl(circuit, decimals=decimals, level=level)(*args, **kwargs)
        if scale is not None:
            dpi = fig.get_dpi()
            fig.set_dpi(dpi*scale)
        if title is not None:
            fig.suptitle(title, fontsize=fontsize)
        plt.show()
    return _draw_circuit


##### Quantum models and facilities

### Creates a sequence encoder
#   wires: list/array of wires to be used
#   data: list/array of input values to be angle encoded (can be less than wires)
def sequence_encoder(wires, data):
    n_inputs = len(data)
    n_wires = len(wires)

    for w in wires:
        qml.Hadamard(wires=w)
        if w > n_inputs-1:
            qml.RY(0, wires=w)
        else:
            qml.RY(data[w], wires=w)
            
### Creates a number encoder
#   wires: list/array of wires to be used
#   numbers: a list with a single number to be basis encoded (can be less than wires)
def number_encoder(wires, num):
    n_wires = len(wires)
    digits = bin_int_to_list(num, n_wires)
    n_digits = len(digits)

    qml.BasisEmbedding(digits, wires)
            

##### Model components

### Creates an ansatz consisting of angle Ry weight encoding and entanglement blocks
#   wires: list/array of wires to create an ansatz on
#   weights: list/array of weights to be used for Ry weight blocks
def ansatz_y(wires, weights):
    qml.BasicEntanglerLayers(weights=weights, wires=wires, rotation=qml.RY)

### Ansatz size estimator
#   n_wires: number of wires
#   n_layers: number of weight+entanglment layers
def ansatz_y_shape(n_wires, n_layers):
    shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_wires)
    return shape

### Creates an ansatz consisting of angle Rx,Ry,Rz weight encoding and entanglement blocks
#   wires: list/array of wires to create an ansatz on
#   weights: list/array of weights to be used for Rx,Ry,Rz weight blocks
def ansatz_xyz(wires, weights):
    qml.StronglyEntanglingLayers(weights=weights, wires=wires)

### Ansatz size estimator
#   n_wires: number of wires
#   n_layers: number of weight+entanglment layers
def ansatz_xyz_shape(n_wires, n_layers):
    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
    return shape
    
### Swap test
#   data_wires: wires with data to be tested
#   comp_wires: wires with data to be compared against
#   test_wire: measured wire
def swap_test(data_wires, comp_wires, test_wire):
    n_wires = range(len(data_wires)+len(comp_wires) + 1)
    n_aux_wire = len(data_wires)+len(comp_wires)
    
    qml.Hadamard(test_wire)
    for i in range(len(data_wires)):
        qml.CSWAP([test_wire, data_wires[i], comp_wires[i]])    
    qml.Hadamard(test_wire)


##### QAE Models

### Creates half QAE with swap test
#   latent_wires: wires spanning the latent space
#   trash_wires: wires spanning the trash space (to be ignored)
#   comp_wires: wires with zeros to compare with trash space
#   test_wire: wire which determines if QAE is converging
#   data: list/array of input values to be angle encoded (can be less than wires)
#   weights: list/array of weights to be used in weight blocks
#   inverse: If True the ansatz is inverted
def half_qae_enc_with_swap(latent_wires, trash_wires, comp_wires, test_wire, data, weights, inverse=False):
                  # seq_name='Input', seq_label='N', 
                  # anz_name='Encoder', anz_label='X'):
    sequence_encoder(latent_wires+trash_wires, data)
    qml.Barrier(latent_wires+trash_wires)
    if inverse:
        qml.adjoint(ansatz_y)(latent_wires+trash_wires, weights)
    else:
        ansatz_y(latent_wires+trash_wires, weights)
    qml.Barrier(latent_wires+trash_wires)
    swap_test(trash_wires, comp_wires, test_wire)

### Estimates the shape of the "half_qae_enc_with_swap"
#   n_latent_wires: number of latent qubits
#   n_trash_wires: number of trash qubits
#   n_layers: number of layers (repeats)
#   returns n_weights: number of required weight parameters
#   returns n_inputs: number of input qubits
#   returns n_wires: total number of qubits used by this model
def half_qae_enc_with_swap_shape(n_latent_wires, n_trash_wires, n_layers=2):
    n_wires = n_latent_wires + 2*n_trash_wires + 1
    n_inputs = n_latent_wires + n_trash_wires
    n_weights = ansatz_y_shape(n_inputs, n_layers)
    return n_weights, n_inputs, n_wires


### Estimates the shape of the "half_qae_enc_with_swap"
#   n_latent: number of latent qubits
#   n_trash: number of trash qubits
#   n_layers: number of layers (repeats)
#   returns n_enc_weights: number of required encoder weight parameters
#   returns n_dec_weights: number of required decoder weight parameters
#   returns n_inputs: number of input qubits
#   returns n_wires: total number of qubits used by this model
def full_qae_shape(n_latent, n_trash, n_extra, n_layers=1, rot='Ry'):
    n_inputs = n_latent + n_trash
    n_wires = n_latent + n_trash + n_extra
    if rot == 'Ry':
        n_weights = ansatz_y_shape(n_wires, n_layers)
    elif rot == 'Rxyz':
        n_weights = ansatz_xyz_shape(n_wires, n_layers)
    else:
        n_weights = 0
    return (2*np.prod(n_weights)), n_inputs, n_wires


### Full-Circuit: Input + Encoder + Decoder + Output + No Swap Test
#   wires: list/array of wires to create a full QAE
#   n_latent: number of latent qubits
#   n_trash: number of trash qubits
#   n_extra: number of additional qubits to increase circuit breadth
#   n_layers: number of layers (repeats)
#   rot: rotation type, 'Ry' or 'Rxyz'
#   data: list/array of input values to be angle encoded
#   weights: list/array of weights to be used in weight blocks
#   add_outseq: if True, the inverse of the input sequence will be added on output
#     If so, data needs to be split into input and output sequence
#   invert_dec: If True the decoder ansatz will be inverted
def full_qae(wires, n_latent, n_trash, n_extra, data, weights, n_layers=1, rot='Ry', add_outseq=True, invert_dec=True):
    latent_wires = wires[0:n_latent]
    trash_wires = wires[n_latent:n_latent+n_trash]
    extra_wires = wires[n_latent+n_trash:]
    data_wires = latent_wires + trash_wires
    anz_wires = latent_wires + trash_wires + extra_wires
    
    n_anz_wires = n_latent + n_trash + n_extra 
    n_data = n_latent + n_trash

    # Calculate shape of enc and dec ansatze, both are the same
    if rot=='Ry':
        ansatz = ansatz_y
        enc_weights_shape = ansatz_y_shape(n_anz_wires, n_layers)
    else:
        ansatz = ansatz_xyz
        enc_weights_shape = ansatz_xyz_shape(n_anz_wires, n_layers)
    dec_weights_shape = enc_weights_shape

    # Splitand shape weights for enc and dec ansatze, incoming weights are flat
    enc_weights_len = np.prod(enc_weights_shape)
    enc_weights = weights[:enc_weights_len].reshape(enc_weights_shape)
    dec_weights = weights[enc_weights_len:].reshape(dec_weights_shape)

    # Add noisy input encoder
    sequence_encoder(data_wires, data)
    qml.Barrier(wires)

    # Add weight encoder
    ansatz(anz_wires, enc_weights)
    qml.Barrier(wires)

    # Add initialisation of trash and extra space
    for w in trash_wires: qml.measure(w, reset=True)
    for w in extra_wires: qml.measure(w, reset=True)
    qml.Barrier(wires)

    # Add weight decoder
    if invert_dec:
        qml.adjoint(ansatz)(anz_wires, dec_weights)
    else:
        ansatz(anz_wires, dec_weights)
    qml.Barrier(wires)

    # Add output sequence if needed
    if add_outseq:
        qml.adjoint(sequence_encoder)(data_wires, data)

### This version takes data as a tuple - needs to be changed
def full_qae_improved(wires, n_latent, n_trash, n_extra, data, weights, n_layers=1, rot='Ry', add_outseq=True, invert_dec=True):
    latent_wires = wires[0:n_latent]
    trash_wires = wires[n_latent:n_latent+n_trash]
    extra_wires = wires[n_latent+n_trash:]
    data_wires = latent_wires + trash_wires
    anz_wires = latent_wires + trash_wires + extra_wires
    
    n_anz_wires = n_latent + n_trash + n_extra 
    n_data = n_latent + n_trash

    # Prepare data for input and output (if needed)
    if add_outseq:
        data_in, data_out = data.split(n_data)
    else:
        data_in = data

    # Calculate shape of enc and dec ansatze, both are the same
    if rot=='Ry':
        ansatz = ansatz_y
        enc_weights_shape = ansatz_y_shape(n_anz_wires, n_layers)
    else:
        ansatz = ansatz_xyz
        enc_weights_shape = ansatz_xyz_shape(n_anz_wires, n_layers)
    dec_weights_shape = enc_weights_shape

    # Splitand shape weights for enc and dec ansatze, incoming weights are flat
    enc_weights_len = np.prod(enc_weights_shape)
    enc_weights = weights[:enc_weights_len].reshape(enc_weights_shape)
    dec_weights = weights[enc_weights_len:].reshape(dec_weights_shape)

    # Add noisy input encoder
    sequence_encoder(data_wires, data_in)
    qml.Barrier(wires)

    # Add weight encoder
    ansatz(anz_wires, enc_weights)
    qml.Barrier(wires)

    # Add initialisation of trash and extra space
    for w in trash_wires: qml.measure(w, reset=True)
    for w in extra_wires: qml.measure(w, reset=True)
    qml.Barrier(wires)

    # Add weight decoder
    if invert_dec:
        qml.adjoint(ansatz)(anz_wires, dec_weights)
    else:
        ansatz(anz_wires, dec_weights)
    qml.Barrier(wires)

    # Add output sequence if needed
    if add_outseq:
        qml.adjoint(sequence_encoder)(data_wires, data_out)


############## Qiskit functions to be translated

### Full-Circuit: Input + Encoder + Decoder + Output + No Swap Test
#   Note: Position of the output block may vary
#   Note: Decoder could be an inverse of encoder or become an independent block
# def train_qae_xyz_xdf(num_latent, num_trash, fm_reps=1, reps=2, ent='sca', added_width=1,
#                       in_seq_name='Input', in_seq_label='I', 
#                       out_seq_name='Output', out_seq_label='O', 
#                       enc_name='Encoder', enc_label='X',
#                       dec_name='Decoder', dec_label='Y', invert_dec=True,
#                       rotation_blocks=['rx', 'ry'], add_statevector=False):

#     ### Create a circuit and its components
#     lt_qubits = num_latent+num_trash
#     qr = QuantumRegister(lt_qubits+added_width, "q")
#     cr = ClassicalRegister(lt_qubits, "c")
#     in_qc, _ = sequence_encoder(lt_qubits, label=in_seq_label)
#     in_qc.name = in_seq_name
#     out_qc, _ = sequence_encoder(lt_qubits, label=out_seq_label)
#     out_qc.name = out_seq_name
#     enc_qc = ansatz_xy(lt_qubits+added_width, reps=reps, ent=ent, label=enc_label, rotation_blocks=rotation_blocks).decompose()
#     enc_qc.name = enc_name
#     dec_qc = ansatz_xy(lt_qubits+added_width, reps=reps, ent=ent, label=dec_label, rotation_blocks=rotation_blocks).decompose()
#     dec_qc.name = dec_name

#     ### Input
#     qc = QuantumCircuit(qr, cr)
#     qc.append(in_qc, qargs=range(lt_qubits))

#     ### Encoder
#     qc.barrier()
#     qc.append(enc_qc, qargs=range(lt_qubits+added_width))

#     ### Latent / Trash
#     qc.barrier()
#     for i in range(num_trash):
#         qc.reset(num_latent + i)
#     for i in range(added_width):
#         qc.reset(lt_qubits+i)

#     ### Decoder
#     qc.barrier()
#     if invert_dec:
#         dec_inv_qc = dec_qc.inverse()
#     else:
#         dec_inv_qc = dec_qc
#     qc.append(dec_inv_qc, qargs=range(lt_qubits+added_width))

#     ### Inverted output (trans input)
#     qc.barrier()
#     out_inv_qc = out_qc.inverse()
#     qc.append(out_inv_qc, qargs=range(lt_qubits))

#     ### Measurements
#     qc.barrier()
#     if add_statevector:
#         qc.save_statevector()
#     for i in range(len(qc.qubits)-added_width):
#         qc.measure(qr[i], cr[i])
    
#     ### Collect weight parameters
#     train_weight_params = []
#     for enc_p in enc_qc.parameters:
#         train_weight_params.append(enc_p)    
#     for dec_p in dec_qc.parameters:
#         train_weight_params.append(dec_p)

#     ### Collect in/out parameters
#     in_out_params = []
#     for in_p in in_qc.parameters:
#         in_out_params.append(in_p)    
#     for out_p in out_qc.parameters:
#         in_out_params.append(out_p)        
    
#     return qc, in_out_params, enc_qc.parameters, dec_qc.parameters, train_weight_params


### Full-Circuit: Input + Encoder + Decoder
#   Note: Position of the output block may vary
#   Note: Decoder could be an inverse of encoder or become an independent block
# def qae_xyz_xdf(num_latent, num_trash, fm_reps=1, reps=2, ent='sca', added_width=1,
#               in_seq_name='Input', in_seq_label='I', 
#               # out_seq_name='Output', out_seq_label='O', 
#               enc_name='Encoder', enc_label='X',
#               dec_name='Decoder', dec_label='Y', invert_dec=True,
#               rotation_blocks=['rx', 'ry'],
#               add_statevector=False):

#     ### Create a circuit and its components
#     lt_qubits = num_latent+num_trash
#     qr = QuantumRegister(lt_qubits+added_width, "q")
#     cr = ClassicalRegister(lt_qubits, "c")
#     in_qc, _ = sequence_encoder(lt_qubits, label=in_seq_label)
#     in_qc.name = in_seq_name
#     enc_qc = ansatz_xy(lt_qubits+added_width, reps=reps, ent=ent, label=enc_label, rotation_blocks=rotation_blocks).decompose()
#     enc_qc.name = enc_name
#     dec_qc = ansatz_xy(lt_qubits+added_width, reps=reps, ent=ent, label=dec_label, rotation_blocks=rotation_blocks).decompose()
#     dec_qc.name = dec_name

#     ### Input
#     qc = QuantumCircuit(qr, cr)
#     qc.append(in_qc, qargs=range(lt_qubits))

#     ### Encoder
#     qc.barrier()
#     qc.append(enc_qc, qargs=range(lt_qubits+added_width))

#     ### Latent / Trash
#     qc.barrier()
#     for i in range(num_trash):
#         qc.reset(num_latent + i)
#     for i in range(added_width): 
#         qc.reset(lt_qubits+i)

#     ### Decoder
#     qc.barrier()
#     if invert_dec:
#         dec_inv_qc = dec_qc.inverse()
#     else:
#         dec_inv_qc = dec_qc
#     qc.append(dec_inv_qc, qargs=range(lt_qubits+added_width))

#     ### Measurements
#     qc.barrier()
#     if add_statevector:
#         qc.save_statevector()
#     for i in range(len(qc.qubits)-added_width):
#         qc.measure(qr[i], cr[i])
    
#     ### Collect weight parameters
#     train_weight_params = []
#     for enc_p in enc_qc.parameters:
#         train_weight_params.append(enc_p)    
#     for dec_p in dec_qc.parameters:
#         train_weight_params.append(dec_p)

#     return qc, in_qc.parameters, enc_qc.parameters, dec_qc.parameters, train_weight_params






######### Junk

### Half-QAE Encoder with Sidekick Decoder$^\dagger$: Noisy Input + Encoder + Swap Test + Sidekick (Decoder$^\dagger$ + Pure Input)
#   Creates an encoder using the previously trained sidekick decoder$^\dagger$ by converging their common latent space.
#   The encoder is to be trained with noisy input and ensuring its latent space converges with that produced by the decoder$^\dagger$ from pure data.
# def half_qae_encoder_with_sidekick(num_latent, num_trash, reps=4, ent='circular',
#                   pure_seq_name='Pure Input', pure_seq_label='I', 
#                   noisy_seq_name='Noisy Input', noisy_seq_label='N', 
#                   enc_name='Encoder', enc_label='X',
#                   dec_name='Decoder', dec_label='Y'):
#     num_qubits = num_latent + num_trash
#     qr = QuantumRegister(2 * num_qubits + 1, "q")
#     cr = ClassicalRegister(1, "c")
#     pure_in_qc, _ = sequence_encoder(num_qubits, label=pure_seq_label)
#     # pure_in_qc = pure_in_qc.inverse()
#     pure_in_qc.name = pure_seq_name
#     encoder_qc = ansatz(num_qubits, reps=reps, ent=ent, label=enc_label).decompose()
#     encoder_qc.name = enc_name
    
#     noisy_in_qc, _ = sequence_encoder(num_qubits, label=noisy_seq_label)
#     noisy_in_qc.name = noisy_seq_name
#     decoder_qc = ansatz(num_qubits, reps=reps, ent=ent, label=dec_label).decompose()
#     decoder_qc = decoder_qc.inverse()
#     decoder_qc.name = dec_name
#     swap_qc = swap_test(num_latent)
#     swap_qc.name = 'Swap'
#     swap_qlist = list(range(num_latent))+list(range(num_qubits, num_qubits+num_latent))+[2*num_qubits]
    
#     qc = QuantumCircuit(qr, cr)
#     qc.append(pure_in_qc, qargs=range(num_qubits))
#     qc.append(noisy_in_qc, qargs=range(num_qubits, 2*num_qubits))
#     qc.barrier()
#     qc.append(decoder_qc, qargs=range(num_qubits))
#     qc.append(encoder_qc, qargs=range(num_qubits, 2*num_qubits))
#     qc.barrier()
#     qc.append(swap_qc, qargs=swap_qlist, cargs=[0])

#     return qc, pure_in_qc.parameters, decoder_qc.parameters, noisy_in_qc.parameters, encoder_qc.parameters


### Half-QAE Encoder: Input + Encoder without a Swap Test
#   Assesses the state of all trash qubits to be $\lvert 0 \rangle$ by their direct measurement and 
#   estimating the probability $P(\lvert 0 \rangle)^n$ (where $n$ is the number of qubits in the trash space. 
#   This approach to measuring qubit similarity is not as nuanced as what's is provided by swap test, 
#   as we miss on the state proximity determined by their inner product. 
#   However, the measurement is fast(er) and does not need additional qubits. 
#   The circuit training needs the cost function $cost = P(1-\lvert 0 \rangle)^n$, which needs to be minimised.
# def half_qae_encoder(num_latent, num_trash, reps=4, ent='circular',
#                   seq_name='Input', seq_label='N', 
#                   anz_name='Encoder', anz_label='X'):
#     qr = QuantumRegister(num_latent + num_trash, "q")
#     cr = ClassicalRegister(num_trash, "c")
#     fm_qc, _ = sequence_encoder(num_latent+num_trash, label=seq_label)
#     fm_qc.name = seq_name
#     anz_qc = ansatz(num_latent+num_trash, reps=reps, ent=ent, label=anz_label).decompose()
#     anz_qc.name = anz_name
    
#     circuit = QuantumCircuit(qr, cr)

#     ### Sequence
#     circuit.append(fm_qc, qargs=range(num_latent+num_trash))

#     ### Encoder
#     circuit.barrier()
#     circuit.append(anz_qc, qargs=range(num_latent + num_trash))
    
#     ### Measurements
#     circuit.barrier()
#     for i in range(num_trash):
#         circuit.measure(qr[num_latent+i], cr[i])
    
#     return circuit, fm_qc, anz_qc


### Full-QAE Stacked Encoder and previously trained Decoder$^\dagger$: Noisy Input + Encoder + Latent Space + Decoder + Pure Input$^\dagger$
#   Creates a full QAE to train its encoder using the previously trained decoder$^\dagger$ by converging the output to $\vert 0 \rangle^n$.
#   The encoder is to be trained with noisy input trough the latent space, then the decoder (no $^\dagger$), pure data$^\dagger$, to result in $\vert 0 \rangle^n$.
#   For training, all qubits of this Full-QAE will be measured.
# def full_qae_stacked(num_latent, num_trash, reps=4, ent='circular',
#                   pure_seq_name='Pure Input', pure_seq_label='I', 
#                   noisy_seq_name='Noisy Input', noisy_seq_label='N', 
#                   enc_name='Encoder', enc_label='X',
#                   dec_name='Decoder', dec_label='Y'):
#     num_qubits = num_latent + num_trash
#     qr = QuantumRegister(num_qubits, "q")
#     cr = ClassicalRegister(num_qubits, "c")
#     pure_in_qc, _ = sequence_encoder(num_qubits, label=pure_seq_label)
#     pure_in_qc.name = pure_seq_name
#     pure_in_qc = pure_in_qc.inverse()
#     encoder_qc = ansatz(num_qubits, reps=reps, ent=ent, label=enc_label).decompose()
#     encoder_qc.name = enc_name
    
#     noisy_in_qc, _ = sequence_encoder(num_qubits, label=noisy_seq_label)
#     noisy_in_qc.name = noisy_seq_name
#     decoder_qc = ansatz(num_qubits, reps=reps, ent=ent, label=dec_label).decompose()
#     decoder_qc = decoder_qc.inverse()
#     decoder_qc.name = dec_name
    
#     qc = QuantumCircuit(qr, cr)
#     qc.append(noisy_in_qc, qargs=range(num_qubits))
#     qc.barrier()
#     qc.append(encoder_qc, qargs=range(num_qubits))
#     qc.barrier()
#     for i in range(num_trash):
#         qc.reset(num_latent + i)
#     qc.barrier()
#     qc.append(decoder_qc, qargs=range(num_qubits))
#     qc.barrier()
#     qc.append(pure_in_qc, qargs=range(num_qubits))
#     qc.barrier()
#     for i in range(num_qubits):
#         qc.measure(qr[i], cr[i])

#     return qc, pure_in_qc.parameters, decoder_qc.parameters, noisy_in_qc.parameters, encoder_qc.parameters


### Full-QAE Encoder with Sidekick Decoder$^\dagger$: Noisy Input + Encoder + Swap Test + Sidekick (Decoder$^\dagger$ + Pure Input)
#   The encoder is to be trained with noisy input. Both the encoder and decoder need to be trained together.
#   They converge onto the pure data encoded in separate qubits via a swap test.
# def full_qae_encoder_with_swap(num_latent, num_trash, reps=4, ent='circular',
#                   pure_seq_name='Pure Input', pure_seq_label='I', 
#                   noisy_seq_name='Noisy Input', noisy_seq_label='N', 
#                   enc_name='Encoder', enc_label='X',
#                   dec_name='Decoder', dec_label='Y'):

#     num_qubits = num_latent + num_trash
#     qr = QuantumRegister(2 * num_qubits + 1, "q")
#     cr = ClassicalRegister(1, "c")

#     pure_in_qc, _ = sequence_encoder(num_qubits, label=pure_seq_label)
#     pure_in_qc.name = pure_seq_name

#     encoder_qc = ansatz(num_qubits, reps=reps, ent=ent, label=enc_label).decompose()
#     encoder_qc.name = enc_name
    
#     noisy_in_qc, _ = sequence_encoder(num_qubits, label=noisy_seq_label)
#     noisy_in_qc.name = noisy_seq_name
    
#     decoder_qc = ansatz(num_qubits, reps=reps, ent=ent, label=dec_label).decompose()
#     decoder_qc = decoder_qc.inverse()
#     decoder_qc.name = dec_name

#     swap_qc = swap_test(num_qubits)
#     swap_qc.name = 'Swap'
    
#     qc = QuantumCircuit(qr, cr)
#     qc.append(pure_in_qc, qargs=range(num_qubits, 2*num_qubits))
#     qc.append(noisy_in_qc, qargs=range(num_qubits))
#     qc.barrier()
#     qc.append(encoder_qc, qargs=range(num_qubits))
#     qc.barrier()
#     for i in range(num_trash):
#         qc.reset(num_latent + i)
#     qc.barrier()
#     qc.append(decoder_qc, qargs=range(num_qubits)) #, 2*num_qubits))
#     qc.barrier()
#     qc.append(swap_qc, qargs=qc.qubits, cargs=[0])

#     return qc, pure_in_qc.parameters, decoder_qc.parameters, noisy_in_qc.parameters, encoder_qc.parameters


### Full-Circuit: Input + Encoder + Decoder + Output + No Swap Test
#   Note: Position of the output block may vary
#   Note: Decoder could be an inverse of encoder or become an independent block
# def train_qae(num_latent, num_trash, reps=4, ent='sca',
#               in_seq_name='Input', in_seq_label='I', 
#               out_seq_name='Output', out_seq_label='O', 
#               enc_name='Encoder', enc_label='X',
#               dec_name='Decoder', dec_label='Y',
#               keep_encoder=False):

#     ### Create a circuit and its components
#     lt_qubits = num_latent+num_trash
#     qr = QuantumRegister(lt_qubits, "q")
#     cr = ClassicalRegister(lt_qubits, "c")
#     in_qc, _ = sequence_encoder(lt_qubits, label=in_seq_label)
#     in_qc.name = in_seq_name
#     out_qc, _ = sequence_encoder(lt_qubits, label=out_seq_label)
#     out_qc.name = out_seq_name
#     enc_qc = ansatz(lt_qubits, reps=reps, ent=ent, label=enc_label).decompose()
#     enc_qc.name = enc_name
#     dec_qc = ansatz(lt_qubits, reps=reps, ent=ent, label=dec_label).decompose()
#     dec_qc.name = dec_name

#     ### Input
#     qc = QuantumCircuit(qr, cr)
#     qc.append(in_qc, qargs=range(lt_qubits))

#     ### Encoder
#     qc.barrier()
#     qc.append(enc_qc, qargs=range(lt_qubits))

#     ### Latent / Trash
#     qc.barrier()
#     for i in range(num_trash):
#         qc.reset(num_latent + i)

#     ### Decoder
#     qc.barrier()
#     if keep_encoder:
#         dec_inv_qc = enc_qc.inverse()
#     else:
#         dec_inv_qc = dec_qc.inverse()
#     qc.append(dec_inv_qc, qargs=range(lt_qubits))

#     ### Inverted output (trans input)
#     qc.barrier()
#     out_inv_qc = out_qc.inverse()
#     qc.append(out_inv_qc, qargs=range(lt_qubits))

#     ### Measurements
#     qc.barrier()
#     for i in range(len(qc.qubits)):
#         qc.measure(qr[i], cr[i])
    
#     ### Collect weight parameters
#     train_weight_params = []
#     for enc_p in enc_qc.parameters:
#         train_weight_params.append(enc_p)    
#     if not keep_encoder:
#         for dec_p in dec_qc.parameters:
#             train_weight_params.append(dec_p)

#     ### Collect in/out parameters
#     in_out_params = []
#     for in_p in in_qc.parameters:
#         in_out_params.append(in_p)    
#     for out_p in out_qc.parameters:
#         in_out_params.append(out_p)        
    
#     if keep_encoder:
#         return qc, in_out_params, enc_qc.parameters, enc_qc.parameters, train_weight_params
#     else:
#         return qc, in_out_params, enc_qc.parameters, dec_qc.parameters, train_weight_params

### Testing Circuit with Two - The entire QAE
#   The full Autoencoder consists of both the Encoder and Decoder, which is simply an inverted Encoder. 
#   Both the Encoder and Decoder can be initialised using the same parameters obtained from the Encoder (plus swap test) training.
#   By applying the full QAE circuit to a test dataset, we can then determine the model accuracy.
# def qae_xyz(lat_no, trash_no, reps=2, ent='sca', add_statevector=False, 
#         classreg=False, meas_q=None, invert_dec=True,
#         in_seq_name='Noisy In', in_seq_label='N', 
#         enc_name='Encoder', enc_label='X',
#         dec_name='Decoder', dec_label='Y',
#         rotation_blocks=['rx', 'ry']
#         ):

#     # Prepare a circuit
#     lt_qubits = lat_no+trash_no
#     qr = QuantumRegister(lat_no + trash_no, 'q')
#     cr = ClassicalRegister(1, 'meas')
#     # qae_qc = QuantumCircuit(lat_no + trash_no, 1)
#     if classreg:
#         qae_qc = QuantumCircuit(qr, cr, name='qae')
#     else:
#         qae_qc = QuantumCircuit(qr, name='qae')

#     # Create all QAE components
#     in_qc, _ = sequence_encoder(lat_no + trash_no, label=in_seq_label)
#     in_qc.name = in_seq_name
#     enc_qc = ansatz_xy(lt_qubits, reps=reps, ent=ent, label=enc_label, rotation_blocks=rotation_blocks).decompose()
#     enc_qc.name = enc_name
#     dec_qc = ansatz_xy(lt_qubits, reps=reps, ent=ent, label=dec_label, rotation_blocks=rotation_blocks).decompose()
#     dec_qc.name = dec_name

#     # Create a circuit
#     qae_qc.append(in_qc, qargs=range(lat_no + trash_no))
#     qae_qc.barrier()
#     qae_qc.append(enc_qc, qargs=range(lat_no + trash_no))
#     qae_qc.barrier()
    
#     for i in range(trash_no):
#         qae_qc.reset(lat_no + i)
    
#     qae_qc.barrier()
#     if invert_dec:
#         dec_inv_qc = dec_qc.inverse()
#     else:
#         dec_inv_qc = dec_qc
#     qae_qc.append(dec_inv_qc, qargs=range(lat_no + trash_no))
#     qae_qc.barrier()

#     # Add optional measurement
#     if add_statevector:
#         qae_qc.save_statevector()
#     if classreg:
#         if meas_q != None:
#             qae_qc.measure(meas_q, 0)
#         else:
#             qae_qc.measure_all()

#     return qae_qc, in_qc, enc_qc, dec_qc

### Testing Circuit - The entire QAE
#   The full Autoencoder consists of both the Encoder and Decoder, which is simply an inverted Encoder. 
#   Both the Encoder and Decoder can be initialised using the same parameters obtained from the Encoder (plus swap test) training.
#   By applying the full QAE circuit to a test dataset, we can then determine the model accuracy.
# def qae(lat_no, trash_no, reps=2, ent='sca', add_statevector=False, 
#         classreg=False, meas_q=None, keep_encoder=False, invert_dec=True,
#         in_seq_name='Noisy In', in_seq_label='N', 
#         enc_name='Encoder', enc_label='X',
#         dec_name='Decoder', dec_label='Y'):

#     # Prepare a circuit
#     qr = QuantumRegister(lat_no + trash_no, 'q')
#     cr = ClassicalRegister(1, 'meas')
#     # qae_qc = QuantumCircuit(lat_no + trash_no, 1)
#     if classreg:
#         qae_qc = QuantumCircuit(qr, cr, name='qae')
#     else:
#         qae_qc = QuantumCircuit(qr, name='qae')

#     # Create all QAE components
#     in_qc, _ = sequence_encoder(lat_no + trash_no, label=in_seq_label)
#     in_qc.name = in_seq_name
#     enc_qc = ansatz(lat_no+trash_no, reps=reps, ent=ent, label=enc_label)
#     enc_qc.name = enc_name
#     dec_qc = ansatz(lat_no+trash_no, reps=reps, ent=ent, label=dec_label)
#     dec_qc.name = dec_name

#     # Create a circuit
#     qae_qc.append(in_qc, qargs=range(lat_no + trash_no))
#     qae_qc.barrier()
#     qae_qc.append(enc_qc, qargs=range(lat_no + trash_no))
#     qae_qc.barrier()
    
#     for i in range(trash_no):
#         qae_qc.reset(lat_no + i)
    
#     qae_qc.barrier()
#     if keep_encoder:
#         dec_inv_qc = enc_qc.inverse()
#     elif invert_dec:
#         dec_inv_qc = dec_qc.inverse()
#     else:
#         dec_inv_qc = dec_qc
#     qae_qc.append(dec_inv_qc, qargs=range(lat_no + trash_no))
#     qae_qc.barrier()

#     # Add optional measurement
#     if add_statevector:
#         qae_qc.save_statevector()
#     if classreg:
#         if meas_q != None:
#             qae_qc.measure(meas_q, 0)
#         else:
#             qae_qc.measure_all()

#     return qae_qc, in_qc, enc_qc, dec_qc
