import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops, to_undirected, remove_self_loops
from torch.nn.functional import cosine_similarity

# Set random seed for reproducibility
# torch.manual_seed(30)
# np.random.seed(30)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Structure Attack ##
class StructureAttack:
    # label attack in Core dataset. Randomly convert 30% of labels
    def label_attack(data, attack_rate=0.3):
        labels = data.y.cpu().numpy()  
        train_mask = data.train_mask.cpu().numpy()
        num_train_nodes = train_mask.sum()
        num_attack = int(attack_rate * num_train_nodes)
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)

        attack_nodes = np.random.choice(np.where(train_mask)[0], num_attack, replace=False)

        for node in attack_nodes:
            current_label = labels[node]
            new_label = np.random.choice([label for label in range(num_classes) if label != current_label])
            labels[node] = new_label

        data.y = torch.tensor(labels).to(device)  
        return data

    # Edge attack in Core dataset. Randomly convert 30% of edge-- delete 30% edge and add 30% fake edge
    def edge_attack(data, attack_rate=0.3):
        edge_index = data.edge_index
        total_edges = edge_index.size(1)
        num_to_delete = int(attack_rate * total_edges)
        num_to_add = int(attack_rate * total_edges)

        # Delete edges
        attack_indices = torch.randperm(total_edges)[:num_to_delete]
        mask = torch.ones(total_edges, dtype=torch.bool, device=device)
        mask[attack_indices] = False
        data.edge_index = edge_index[:, mask]

        # Add edges
        num_nodes = data.num_nodes
        existing_edges = set()

        edges = data.edge_index.t().cpu().numpy()
        for src, dst in edges:
            existing_edges.add((src, dst))
            existing_edges.add((dst, src))

        new_edges = []
        attempts = 0
        max_attempts = 2 * num_to_add  

        # Create new edge and check it a duplicate or not 
        while len(new_edges) < num_to_add and attempts < max_attempts:
            batch_size = min(1000, num_to_add - len(new_edges))
            src = torch.randint(0, num_nodes, (batch_size,), device=device)
            dst = torch.randint(0, num_nodes, (batch_size,), device=device)
            
            mask = (src != dst)
            for i in range(batch_size):
                if mask[i]:
                    a, b = src[i].item(), dst[i].item()
                    if (a, b) not in existing_edges and (b, a) not in existing_edges:
                        new_edges.append((a, b))
                        existing_edges.add((a, b))
                        existing_edges.add((b, a)) 
            attempts += batch_size


        if new_edges:
            new_edges_tensor = torch.tensor(new_edges, dtype=torch.long, device=device).t()
            data.edge_index = torch.cat([data.edge_index, new_edges_tensor], dim=1)

        return data

## FGSM Adversarial Attack ##
class AdversarialAttack:
    def __init__(self, model, data, epsilon=0.1):
        self.model = model
        self.data = data
        self.epsilon = epsilon

    def FGSMattack(self):
        self.model.eval()
        data = self.data

        perturbed_data = data.clone()
        perturbed_data.x.requires_grad = True

        output = self.model(perturbed_data.x, perturbed_data.edge_index)
        loss = F.nll_loss(output[data.test_mask], data.y[data.test_mask])

        self.model.zero_grad()
        loss.backward()

        # FGSM attack
        perturbation = self.epsilon * perturbed_data.x.grad.sign()
        perturbed_data.x = perturbed_data.x.detach() + perturbation
        perturbed_data.x = perturbed_data.x.clamp(0, 1)

        return perturbed_data

class FeatureAdversarialAttack:
    def __init__(self, model, data, topk_features=200, n_steps=5, step_size=0.1, eps=0.5):
        self.model = model.eval()
        self.data = data.clone()
        self.topk_features = topk_features      
        self.n_steps = n_steps                 
        self.step_size = step_size              
        self.eps = eps                         
        self.original_x = data.x.detach().clone()  

    # find 30% important node and reture 
    def _identify_critical_nodes(self):
        with torch.enable_grad():
            x = self.original_x.requires_grad_(True)
            loss = F.nll_loss(self.model(x, self.data.edge_index), self.data.y)
            node_grad = torch.autograd.grad(loss, x)[0].abs().sum(1) 
            
        num_target = int(self.data.num_nodes * 0.3)  
        return node_grad.topk(num_target).indices   

    def attack(self):
        target_nodes = self._identify_critical_nodes()
        perturbed_x = self.original_x.clone()
        
        for _ in range(self.n_steps):
            perturbed_x = perturbed_x.detach().requires_grad_(True)
            loss = F.nll_loss(
                self.model(perturbed_x, self.data.edge_index)[target_nodes],
                self.data.y[target_nodes]
            )
            grad = torch.autograd.grad(loss, perturbed_x, retain_graph=True)[0]

            node_grad = grad[target_nodes].abs()
            _, topk_indices = node_grad.topk(self.topk_features, dim=1)
            
            delta = torch.zeros_like(perturbed_x)
            delta[target_nodes] = self.step_size * grad[target_nodes].sign()
            delta[target_nodes] *= torch.zeros_like(delta[target_nodes]).scatter(1, topk_indices, 1)

            perturbed_x = (perturbed_x + delta).clamp(
                min=self.original_x - self.eps,
                max=self.original_x + self.eps
            ).clamp(0, 1).detach()

        attacked_data = self.data.clone()
        attacked_data.x = perturbed_x
        return attacked_data

