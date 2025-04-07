import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch import Tensor
from attack import AdversarialAttack

def adversarial_train(model, data, epochs=300):
    model.train()

    for epoch in range(epochs):
        output_clean = model(data.x, data.edge_index)
        loss_clean = F.nll_loss(output_clean[data.train_mask], data.y[data.train_mask])

        perturbed_data = AdversarialAttack(model, data).FGSMattack()
        output_adv = model(perturbed_data.x, perturbed_data.edge_index)
        loss_adv = F.nll_loss(output_adv[data.train_mask], data.y[data.train_mask])
        
        total_loss = loss_clean + 0.5 * loss_adv
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

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

gnn_model = GCN(
    in_channels=1433, 
    hidden_channels=2,    
    out_channels=7,      
    dropout_rate=0.2
)

optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()

def train():
    gnn_model.train()
    optimizer.zero_grad()
    out = gnn_model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

class SingleNodeAdversarialAttack:
    def __init__(self, model, data, epsilon=0.1, target_node=None):
        self.model = model
        self.data = data
        self.epsilon = epsilon
        self.target_node = target_node if target_node else data.test_mask.nonzero()[0][0].item()

    def attack(self):
        self.model.eval()
        
        perturbed_data = Data(
            x=self.data.x.clone(),
            edge_index=self.data.edge_index.clone(),
            y=self.data.y.clone()
        )
        
        perturbed_data.x.requires_grad_(True)
        perturbed_data.x.retain_grad() 
        
        output = self.model(perturbed_data.x, perturbed_data.edge_index)
        loss = F.nll_loss(output[self.target_node].unsqueeze(0), 
                         self.data.y[self.target_node].unsqueeze(0))
        
        self.model.zero_grad()
        loss.backward()
        
        # Just attack target node
        with torch.no_grad():
            perturbation = self.epsilon * perturbed_data.x.grad.sign()
            perturbed_data.x[self.target_node] += perturbation[self.target_node]
            perturbed_data.x = torch.clamp(perturbed_data.x, 0, 1)
        
        return perturbed_data

    def evaluate_attack(self, perturbed_data):
        self.model.eval()
        output = self.model(perturbed_data.x, perturbed_data.edge_index)
        perturbed_pred = output.argmax(dim=1)
        
        print(f"Sign Node Attack Evaluation ")
        print(f"Target Node {self.target_node}:")
        print(f"Original prediction: {self.data.y[self.target_node].item()} "
              f"(Accuracy: {output[self.target_node].max().exp().item():.4f})")
        print(f"Attack prediction: {perturbed_pred[self.target_node].item()} "
              f"(Accuracy: {output[self.target_node][perturbed_pred[self.target_node]].exp().item():.4f})")
        
dataset = Planetoid(root='./data/Cora', name='Cora')
data = dataset[0] 
attack = SingleNodeAdversarialAttack(model=gnn_model, 
                              data=data,
                              epsilon=0.1,
                              target_node=121)
# Train model
for epoch in range(1, 301):
    loss = train()

# Clear optimizer and start adversarial training
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01, weight_decay=5e-4)
adversarial_train(gnn_model, data)

perturbed_graph = attack.attack()
attack.evaluate_attack(perturbed_graph)