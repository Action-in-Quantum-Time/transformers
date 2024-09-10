## Załadowanie danych facebook 
from torch_geometric.datasets import FacebookPagePage

dataset = FacebookPagePage(root="./data/Facebook/")
data = dataset[0]

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
print(f'Contains self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


data.train_mask = range(18000)
data.val_mask = range(18001, 20000)
data.test_mask = range(20001, 22470)


from MyNN import MLP

# ==================
# TESTY 
# dla roznych wymiarow warstwy ukrytej 
# nie ma roznic w wynikach zawsze jest ok 75%


mlp = MLP(dataset.num_features, 16, dataset.num_classes)
print(mlp)
mlp.fit(data, epochs=100)
acc = mlp.test(data)
print(f'MLP test accuracy: {acc*100:.2f}%')




# X_train = data.x[data.train_mask]
# y_train = data.y[data.train_mask]
# X_test = data.x[data.test_mask]
# y_test = data.y[data.test_mask]

#print(y_test)
# from sklearn.linear_model import LogisticRegression

# model = LogisticRegression(**parameters)
# model.fit(X_train, y_train.values.ravel())

# from sklearn.metrics import roc_auc_score, average_precision_score




