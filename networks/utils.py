import pyarrow.parquet as pq
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx
from sklearn.manifold import TSNE
import igraph as ig

def _read_parquet_edges(path: str) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()

def _to_netx(df: pd.DataFrame, source='in',target='out'):
    return nx.from_pandas_edgelist(df,source=source,target=target, create_using=nx.Graph)

def to_torch_data(edges_path: str):
    G = _to_netx(_read_parquet_edges(edges_path))
    return from_networkx(G)


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1],s=70, c=color, cmap="Set2")
    plt.show()

def community_search(n_nodes: int, edges_list: list, epochs: int = 1000):
    g = ig.Graph()
    g.add_vertices(n_nodes)
    g.add_edges(edges_list)
    max_mod = 0
    best_comm = []
    for _ in range(epochs):
        g_comm = g.community_leiden(objective_function="modularity", n_iterations=-1)
        if g_comm.modularity > max_mod:
            max_mod = g_comm.modularity
            sbest_comm = g_comm.membership
    return max_mod, best_comm

def save_csv(file_name: str):
    import csv
    with open(file_name, 'w') as f:
        write = csv.writer(f)
        write.writerow(best_comm)







for _ in range(1000):
    g_comm = g.community_leiden(objective_function="modularity", n_iterations=-1)
    if g_comm.modularity > max_mod:
        max_mod = g_comm.modularity
        best_comm = g_comm.membership

import csv
with open('karate_community', 'w') as f:
    write = csv.writer(f)
    write.writerow(best_comm)
