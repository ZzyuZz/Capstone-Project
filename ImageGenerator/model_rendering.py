import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Used Cora dataset
dataset = Planetoid(root='Cora', name='Cora')

# GCN model setting
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)  # Apply dropout after first layer
        x = self.conv2(x, edge_index)
        return x

# Create GNN model
model = GCN(dataset.num_features, 16, dataset.num_classes, dropout_rate=0.2)

# Get data
data = dataset[0]

# Set L2 regularization 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Learning rate scheduler

# Train model function
def train_model(model, data, optimizer, scheduler, epochs=300):
    losses = []
    accuracies = []
    for epoch in range(epochs):
        model.train()
        pred = model(data.x, data.edge_index)
        loss =  F.nll_loss(pred[data.train_mask], data.y[data.train_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() 
        
        # Collect loss
        losses.append(loss.item())
        
        # Check accuracy every 20 epochs
        if epoch % 20 == 0:
            accuracy = test_model(model, data)
            accuracies.append(accuracy)
            print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy:.4f}')
    
    # Fill the remaining accuracy values
    while len(accuracies) < len(losses):
        accuracies.append(accuracies[-1])
    
    return losses, accuracies

# Test model function
def test_model(model, data):
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index)
        _, pred_classes = pred.max(dim=1) 
        correct = (pred_classes[data.test_mask] == data.y[data.test_mask]).sum()
        accuracy = int(correct) / int(data.test_mask.sum())  
    return accuracy

# Train the model
losses, accuracies = train_model(model, data, optimizer, scheduler)

# Plotting Loss and Accuracy
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss', color='blue')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Accuracy', color='orange')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()