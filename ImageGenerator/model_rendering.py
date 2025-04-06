import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse

# Command 
parser = argparse.ArgumentParser(description='GNN Model Selection')
parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT'],
                    help="Select model GCN or GAT (default: GCN)")
args = parser.parse_args()

# Used Cora dataset
dataset = Planetoid(root='../data/Cora', name='Cora')

# GCN model setting 
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

# GAT model setting 
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

# Create GNN model
if args.model == 'GCN':
    model = GCN(dataset.num_features, 16, dataset.num_classes, dropout_rate=0.2)
else:
    model = GAT(dataset.num_features, 16, dataset.num_classes, heads=8, dropout_rate=0.2)
data = dataset[0]

# Set L2 regularization and Learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Train model function
def train_model(model, data, optimizer, scheduler, epochs=300, patience=300):
    best_val_loss = float('inf')  
    patience_counter = 0         
    train_losses = []             
    val_accuracies = []          
    test_accuracies = []       
    
    for epoch in range(epochs):
        model.train()
        pred = model(data.x, data.edge_index)
        loss = F.nll_loss(pred[data.train_mask], data.y[data.train_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())  
        
        model.eval()
        with torch.no_grad():
            val_pred = model(data.x, data.edge_index)
            val_loss = F.nll_loss(val_pred[data.val_mask], data.y[data.val_mask])
            
            _, val_classes = val_pred.max(dim=1)
            val_correct = (val_classes[data.val_mask] == data.y[data.val_mask]).sum()
            val_acc = val_correct.item() / data.val_mask.sum().item()
            val_accuracies.append(val_acc)
        
        # Stop program by patience count
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 
        else:
            patience_counter += 1
            # print(patience_counter) 
            
        # Check accuracy every 20 epochs
        if epoch % 20 == 0:
            test_acc = test_model(model, data)
            test_accuracies.append(test_acc)
            print(f'Epoch {epoch:03d} | '
                  f'Train Loss: {loss:.4f} | '
                  f'Val Acc: {val_acc:.4f} | '
                  f'Test Acc: {test_acc:.4f}')
        
        # check patience
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch}')
            break
    
    # Fill the remaining accuracy values
    while len(test_accuracies) < len(train_losses):
        test_accuracies.append(test_accuracies[-1] if test_accuracies else 0)
    
    return train_losses, test_accuracies

# Test model function
def test_model(model, data):
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index)
        _, pred_classes = pred.max(dim=1) 
        correct = (pred_classes[data.test_mask] == data.y[data.test_mask]).sum()
        accuracy = int(correct) / int(data.test_mask.sum())  
    return accuracy

# Train the model and get the result
losses, accuracies = train_model(model, data, optimizer, scheduler, epochs=300, patience=300)

# Plotting Loss and Accuracy
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(losses, label='Training Loss', color='blue')
plt.title('Training Loss Over Epochs'+" ("+args.model+")")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Test Accuracy', color='orange')
plt.title('Test Accuracy Over Epochs'+" ("+args.model+")")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
