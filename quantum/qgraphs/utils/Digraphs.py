# Support functions for QGraph project
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for quantum time series analysis
# Date: 2024

##### Initial settings

import sys
sys.path.append('.')
sys.path.append('..')
sys.path

import os
import numpy as np
import pylab
import math
import json

import networkx as nx
from networkx.readwrite import json_graph

import matplotlib.pyplot as plt
from matplotlib import set_loglevel
set_loglevel("error")

from IPython.display import clear_output


### Draw a digraph
def draw_digraph(G, ax=None, weight_prec=3, font_size=12):
    pos = nx.shell_layout(G)
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.planar_layout(G)
    # pos = nx.spring_layout(G)
    # cstyle = "arc3,rad=0.3"
    cstyle = "arc3"
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=font_size, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="grey", ax=ax, connectionstyle=cstyle)


### Draw a weighted digraph
def draw_weighted_digraph(G, attr_name, ax=None, weight_prec=3, font_size=12):
    pos = nx.shell_layout(G)
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.planar_layout(G)
    # pos = nx.spring_layout(G)
    # cstyle = "arc3,rad=0.3"
    cstyle = "arc3"
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=font_size, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="grey", ax=ax, connectionstyle=cstyle)

    labels = {tuple(edge) : f"{np.round(attrs[attr_name], weight_prec)}" for *edge, attrs in G.edges(data=True)}
    
    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        label_pos=0.35,
        font_color="blue",
        bbox={"alpha": 0},
        verticalalignment='baseline', # top, bottom, center, center_baseline
        ax=ax,
    )


### Saving of a graph to a file
def save_digraph(G, fpath, vers=0):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    g_node_links = nx.node_link_data(G)
    with open(f'{fpath}', 'w') as f:
        json.dump(g_node_links, f)
        f.close()


### Loading a graph from a file
def load_digraph(fpath):
    if not os.path.exists(fpath):
        print(f'*** ERROR: The digraph file does not exist or is corrupted: {fpath}')
        return nx.null_graph()
    else:
        try:
            f = open(fpath, 'r')
        except OSError:
            print(f'*** ERROR: Could not open/read the digraph file: {fpath}')
            return nx.null_graph()
        with f:
            with open(f'{fpath}', 'r') as f:
                G_node_links = json.load(f)
                f.close()
            G = nx.node_link_graph(G_node_links)
            return G

### Convert a digraph to its adjacency matrix
def digraph_to_adjmat(G):
    adj_comp = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    return adj_comp.todense()

### Return digraph details
def digraph_details(G):
    return nx.node_link_data(G)