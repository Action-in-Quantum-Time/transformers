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


##### Graph drawing


### Draw a digraph
def draw_digraph(G, ax=None, weight_prec=3, font_size=12, 
                 rcParams=(8, 6), save_plot=None):
    
    # Set graph plotting parameter
    pos = nx.shell_layout(G)
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.planar_layout(G)
    # pos = nx.spring_layout(G)
    # cstyle = "arc3,rad=0.3"
    cstyle = "arc3"
    
    # Plot graph
    if rcParams is not None:
        plt.rcParams["figure.figsize"] = rcParams

    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=font_size, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="grey", ax=ax, connectionstyle=cstyle)

    # Final charting
    if save_plot is not None:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, format='eps')
    plt.draw()


### Draw a weighted digraph
def draw_weighted_digraph(G, attr_name, ax=None, weight_prec=3, font_size=12, 
                          rcParams=(8, 6), save_plot=None):
    
    # Set graph plotting parameter
    pos = nx.shell_layout(G)
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.planar_layout(G)
    # pos = nx.spring_layout(G)
    # cstyle = "arc3,rad=0.3"
    cstyle = "arc3"

    # Plot graph
    if rcParams is not None:
        plt.rcParams["figure.figsize"] = rcParams

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

    # Final charting
    if save_plot is not None:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, format='eps')
    plt.draw()


### Draw two weighted digraphs
def draw_weighted_digraphs(GS, titles=['Original digraph', 'Predicted digraph'],
                           attr_name='weight', ax=None, weight_prec=3, font_size=12, 
                           rcParams=(8, 6), draw_plot=True, save_plot=None):
    # Get both graphs
    G1 = GS[0]
    G2 = GS[1]

    # cstyle = "arc3,rad=0.3"
    cstyle = "arc3"

    # Plot graph
    if rcParams is not None:
        plt.rcParams["figure.figsize"] = rcParams

    ### G1
    # Set graph plotting parameter
    pos = nx.shell_layout(G1)
    # pos = nx.kamada_kawai_layout(G1)
    # pos = nx.planar_layout(G1)
    # pos = nx.spring_layout(G1)
    plt.subplot(1, 2, 1)  # row 1, column 2, count 1
    plt.title(titles[0])
    nx.draw_networkx_nodes(G1, pos, ax=ax)
    nx.draw_networkx_labels(G1, pos, font_size=font_size, ax=ax)
    nx.draw_networkx_edges(G1, pos, edge_color="grey", ax=ax, connectionstyle=cstyle)

    labels = {tuple(edge) : f"{np.round(attrs[attr_name], weight_prec)}" for *edge, attrs in G1.edges(data=True)}
    
    nx.draw_networkx_edge_labels(
        G1,
        pos,
        labels,
        label_pos=0.35,
        font_color="blue",
        bbox={"alpha": 0},
        verticalalignment='baseline', # top, bottom, center, center_baseline
        ax=ax,
    )

    ### G2
    # Set graph plotting parameter
    pos = nx.shell_layout(G1)
    # pos = nx.kamada_kawai_layout(G1)
    # pos = nx.planar_layout(G1)
    # pos = nx.spring_layout(G1)
    plt.subplot(1, 2, 2) # row 1, column 2, count 2
    plt.title(titles[1])
    nx.draw_networkx_nodes(G2, pos, ax=ax)
    nx.draw_networkx_labels(G2, pos, font_size=font_size, ax=ax)
    nx.draw_networkx_edges(G2, pos, edge_color="grey", ax=ax, connectionstyle=cstyle)

    labels = {tuple(edge) : f"{np.round(attrs[attr_name], weight_prec)}" for *edge, attrs in G2.edges(data=True)}
    
    nx.draw_networkx_edge_labels(
        G2,
        pos,
        labels,
        label_pos=0.35,
        font_color="blue",
        bbox={"alpha": 0},
        verticalalignment='baseline', # top, bottom, center, center_baseline
        ax=ax,
    )

    plt.tight_layout(w_pad=3)
    
    # Final charting
    if save_plot is not None:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, format='eps')
    if draw_plot:
        plt.draw()
    else:
        plt.close()


##### Graph saving and loading


### Saving of a graph to a file
def save_digraph(G, fpath, vers=0):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    g_node_links = nx.node_link_data(G, edges="links")
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
            G = nx.node_link_graph(G_node_links, edges='links')
            return G

### Convert a digraph to its adjacency matrix
def digraph_to_adjmat(G):
    adj_comp = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    return adj_comp.todense()

### Return digraph details
def digraph_details(G):
    return nx.node_link_data(G, edges="links")


##### Graph generation


### Convert an adjacency of a unweighted graph to adjacency of a weighted graph
#   Two methods of generating weights:
#   - rand: random weights are generated
#   - scale: existing weights are scaled
def digraph_adj_weigh(unw_adj, method='rand'):
    w_adj = unw_adj.copy().astype(float)
    for r in range(unw_adj.shape[0]):
        r_sum = sum(unw_adj[r])
        r_nz = np.count_nonzero(unw_adj[r])
        if r_sum != 0.0:
            # Edges available - generate weights
            if method == 'rand':
                nz_weights = np.random.random(r_nz)
            else:
                nz_weights = np.array([num*1.0 for num in unw_adj[r] if num])
            nz_weights /= nz_weights.sum()
            w_no = 0
            for c in range(unw_adj.shape[1]):
                if unw_adj[r, c] > 0:
                    w_adj[r, c] = nz_weights[w_no]
                    w_no += 1
    return w_adj

### Expand a weighted digraph to eliminate vertices with out-dgree=0
def digraph_adj_expand(w_adj):
    exp_adj = w_adj.copy() #.toarray()
    for r in range(w_adj.shape[0]):
        r_sum = np.count_nonzero(w_adj[r])
        if r_sum == 0:
            # No outgoing links - create a loop
            exp_adj[r, r] = 1.0
    return exp_adj

### Prepare a quantum digraph for quantum modeling
#   Convert an undirected graph into QGraph
def digraph_expanded_and_weighed(g, method='rand'):
    g_adj = nx.adjacency_matrix(g).toarray() # todense()
    g_adj_expanded = digraph_adj_expand(g_adj)
    g_adj_weighed = digraph_adj_weigh(g_adj_expanded, method=method)
    g_new = nx.DiGraph(g_adj_weighed)
    return g_new