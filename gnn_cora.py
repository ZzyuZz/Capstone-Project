import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F

from attack import StructureAttack, Nettack

# uesd Cora dataset
dataset = Planetoid(root='Cora', name='Cora')

# GCN model setting
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))    

        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs[:-1]:  
            x = conv(x, edge_index).relu()
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index)
        return x

# create gnn model and get data
model = GCN(dataset.num_features, 32, dataset.num_classes)
data = dataset[0]

# set L2 regularization and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) 

# reset model, optimizer and learning rate scheduler function
def reset():
    data = dataset[0]
    model = GCN(dataset.num_features, 32, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# train model function
def train_model(model, data, optimizer, scheduler, epochs=200):
    for epoch in range(epochs):
        model.train()
        pred = model(data.x, data.edge_index)
        loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() 
        # if epoch % 20 == 0:
        #     print(f'Epoch {epoch}, Loss: {loss.item()}')

# test model function
def test_model(model, data):
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index)
        _, pred_classes = pred.max(dim=1) 
        correct = (pred_classes[data.test_mask] == data.y[data.test_mask]).sum()
        accuracy = int(correct) / int(data.test_mask.sum())  
    return accuracy

# train the model
train_model(model, data, optimizer, scheduler)
# test the model
accuracy = test_model(model, data)
print(f'original Accuracy: {accuracy:.4f}')

# # label attack test
# attck_data = StructureAttack.label_attack(data)
# reset()
# train_model(model, attck_data, optimizer, scheduler)
# accuracy = test_model(model, data)
# print(f'After label attack Accuracy: {accuracy:.4f}')

# # edge attack test
# attck_data = StructureAttack.edge_attack(data)
# reset()
# train_model(model, attck_data, optimizer, scheduler)
# accuracy = test_model(model, data)
# print(f'After edge attack Accuracy: {accuracy:.4f}')

# edge attack test
reset()
net_attack = Nettack(data, modification_budget=3)
attck_data  = net_attack.adversarial_attack()
train_model(model, attck_data, optimizer, scheduler)
accuracy = test_model(model, data)
print(f'After adversarial attack Accuracy: {accuracy:.4f}')