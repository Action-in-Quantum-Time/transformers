import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


G_nx = nx.karate_club_graph()
print(G_nx.number_of_edges(), G_nx.number_of_nodes())

from torch_geometric.datasets import KarateClub

dataset = KarateClub()
print(dataset)
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')