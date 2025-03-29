import torch
import numpy as np
import torch.nn.functional as F

# set random seed for reproducibility
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

        attack_nodes = np.random.choice(np.where(train_mask)[0], num_attack, replace=False)

        for node in attack_nodes:
            current_label = labels[node]
            new_label = np.random.choice([label for label in range(7) if label != current_label])
            labels[node] = new_label

        data.y = torch.tensor(labels).to(device)  
        return data

    # edge attack in Core dataset. Randomly convert 30% of edge-- delete 30% edge and add 30% fake edge
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
        new_edges = []

        for _ in range(num_to_add):
            src, dst = torch.randint(0, num_nodes, (2,), device=device)
            while (src, dst) in data.edge_index.t().tolist() or src == dst:
                src, dst = torch.randint(0, num_nodes, (2,), device=device)
            new_edges.append((src, dst))

        if new_edges:
            new_edges_tensor = torch.tensor(new_edges, device=device).t()
            data.edge_index = torch.cat((data.edge_index, new_edges_tensor), dim=1)

        return data

## Adversarial Attack ##
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
