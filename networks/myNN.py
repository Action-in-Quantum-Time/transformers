import torch 
from torch.nn import Linear
from torch_geometric.nn.conv import GCNConv, GATConv

import torch.nn.functional as F

# import torchmetrics

# accuracy_score = torchmetrics.classification.Accuracy(task="binary")
#accuracy_score = torchmetrics.classification.AveragePrecision(task="binary")

# accuracy_score = torchmetrics.classification.AUROC(task='binary')
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#accuracy_score.to(device)


# from sklearn.metrics import accuracy_score

def accuracy_score(y_pred, y_true):
    ## need for pytorch implementation with cuda and mps device
    return torch.sum(y_pred == y_true)/len(y_true)


     

TORCH_SEED = 123


class MLP(torch.nn.Module):

    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        torch.manual_seed(TORCH_SEED)
        self.linear1 = Linear(dim_in, dim_h)
        self.linear2 = Linear(dim_h, dim_out)

    def forward(self, X):
        X = self.linear1(X)
        X = torch.relu(X)
        X = self.linear2(X)
        return F.log_softmax(X, dim=1)

    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out = self(data.x)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            # funkcja do optymalizacji
            acc = accuracy_score(data.y[data.train_mask], out[data.train_mask].argmax(dim=1))
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy_score(data.y[data.val_mask], out[data.val_mask].argmax(dim=1))
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Val Loss {val_loss:.2f} | Val Acc: {val_acc*100:.2f}% ')

    def test(self, data):
        self.eval()
        out = self(data.x)
        acc = accuracy_score(data.y[data.test_mask], out[data.test_mask].argmax(dim=1))
        return acc




class GCN(torch.nn.Module):

    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        torch.manual_seed(TORCH_SEED)
        self.conv1 = GCNConv(dim_in, dim_h)
        self.conv2 = GCNConv(dim_h,dim_out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            # funkcja do optymalizacji
            acc = accuracy_score(data.y[data.train_mask], out[data.train_mask].argmax(dim=1))
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy_score(data.y[data.val_mask], out[data.val_mask].argmax(dim=1))
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Val Loss {val_loss:.2f} | Val Acc: {val_acc*100:.2f}% ')

    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        acc = accuracy_score(data.y[data.test_mask], out[data.test_mask].argmax(dim=1))
        return acc


class GAT(torch.nn.Module):

    def __init__(self, dim_in, dim_h, dim_out, heads):
        super().__init__()
        torch.manual_seed(TORCH_SEED)
        self.conv1 = GATConv(dim_in, dim_h, heads)
        self.conv2 = GATConv(heads*dim_h, dim_out, heads)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            # funkcja do optymalizacji
            acc = accuracy_score(data.y[data.train_mask], out[data.train_mask].argmax(dim=1))
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy_score(data.y[data.val_mask], out[data.val_mask].argmax(dim=1))
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Val Loss {val_loss:.2f} | Val Acc: {val_acc*100:.2f}% ')

    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        acc = accuracy_score(data.y[data.test_mask], out[data.test_mask].argmax(dim=1))
        return acc