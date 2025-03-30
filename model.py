import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid, Reddit
from sklearn.metrics import f1_score
import torch.nn.functional as F
import numpy as np
from attack import StructureAttack, AdversarialAttack

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GCN model definition
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1) 

# GAT model definition
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout_rate=0.2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        self.convs.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads))
        self.convs.append(GATConv(hidden_channels*heads, out_channels, heads=1))
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = F.relu(self.convs[0](x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.convs[1](x, edge_index)) 
        x = self.dropout(x)
        x = self.convs[2](x, edge_index)
        return F.log_softmax(x, dim=1) 

# Load Cora or Citeseer dataset
def load_data():
    dataset = Planetoid(root='./data/Cora', name='Cora')
    # dataset = Reddit(root='./data/Reddit')
    data = dataset[0].to(device)

    original_train_labels = data.y[data.train_mask].cpu().numpy()
    original_unique, original_counts = np.unique(original_train_labels, return_counts=True)
    original_dist = np.zeros(dataset.num_classes)
    original_dist[original_unique] = original_counts / original_counts.sum()

    return data, dataset.num_features, dataset.num_classes, original_dist

# Reset model, optimizer, and learning rate scheduler
def reset_model(model_name, num_features, num_classes):
    if model_name == 'GCN':
        model = GCN(num_features, 16, num_classes)
    elif model_name == 'GAT':
        model = GAT(num_features, 16, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    return model, optimizer, scheduler

# Train model
def train_model(model, data, optimizer, scheduler, epochs=300):
    for epoch in range(epochs):
        model.train()
        pred = model(data.x, data.edge_index)
        loss = F.nll_loss(pred[data.train_mask], data.y[data.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

# Test model accuracy
def test_model_accuracy(model, data):
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index)
        _, pred_classes = pred.max(dim=1)
        correct = (pred_classes[data.test_mask] == data.y[data.test_mask]).sum()
        accuracy = int(correct) / int(data.test_mask.sum())
    return accuracy


# Test model F1 score
def test_model_f1_score(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

    true_labels = data.y[data.test_mask].cpu().numpy()
    predicted_labels = pred[data.test_mask].cpu().numpy()
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    return f1

def adversarial_train(model, data, optimizer, scheduler, epochs=300):
    model.train()

    for epoch in range(epochs):
        output_clean = model(data.x, data.edge_index)
        loss_clean = F.nll_loss(output_clean[data.train_mask], data.y[data.train_mask])

        perturbed_data = AdversarialAttack(model, data).FGSMattack(0.1).to(device)
        output_adv = model(perturbed_data.x, perturbed_data.edge_index).to(device)
        loss_adv = F.nll_loss(output_adv[data.train_mask], data.y[data.train_mask])
        
        total_loss = loss_clean + 0.5 * loss_adv
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()