import torch
import numpy as np
from torch_geometric.utils import from_networkx, to_networkx
import networkx as nx

# set random seed for reproducibility
# torch.manual_seed(30)
# np.random.seed(30)

## Structure Attack ##
# label attack in Core dataset. Randomly convert 30% of labels
class StructureAttack:
    def label_attack(data, attack_rate=0.3):
        G = to_networkx(data, to_undirected=True)  
        labels = data.y.numpy() 
        num_nodes = len(labels)  
        num_attack = int(attack_rate * num_nodes)  
        
        attack_nodes = np.random.choice(num_nodes, num_attack, replace=False)
        
        for node in attack_nodes:
            neighbors = list(G.neighbors(node))  
            if neighbors: 
                current_label = labels[node]
                possible_labels = [label for label in labels[neighbors] if label != current_label]
                
                if possible_labels:  
                    new_label = np.random.choice(possible_labels) 
                    # print(f'Node {node} label changed from {current_label} to {new_label}')  
                    labels[node] = new_label  
        
        data.y = torch.tensor(labels)  
        return data  

    # edge attack in Core dataset. Randomly convert 30% of edge-- delete 30% edge and add 30% fake edge
    def edge_attack(data, attack_rate=0.3):
        edge_index = data.edge_index 
        total_edges = edge_index.size(1)  
        num_to_delete = int(attack_rate * total_edges)  
        num_to_add = int(attack_rate * total_edges)

        # delete egde
        attack_indices = torch.randperm(total_edges)[:num_to_delete]

        mask = torch.ones(total_edges, dtype=torch.bool)
        mask[attack_indices] = False

        data.edge_index = edge_index[:, mask]

        # add edge
        num_nodes = data.num_nodes
        new_edges = []

        for _ in range(num_to_add):
            src, dst = torch.randint(0, num_nodes, (2,))
            while (src, dst) in data.edge_index.t().tolist() or src == dst:
                src, dst = torch.randint(0, num_nodes, (2,))
            new_edges.append((src, dst))

        if new_edges:
            new_edges_tensor = torch.tensor(new_edges).t()
            data.edge_index = torch.cat((data.edge_index, new_edges_tensor), dim=1)

        return data

## Model Attack ##
# Adversarial Attack 
class Nettack:
    def __init__(self, data, modification_budget=3):
        self.data = data
        self.graph = to_networkx(self.data, to_undirected=True)
        self.modification_budget = modification_budget

    def adversarial_attack(self):
        central_nodes = self.select_critical_nodes()
        return self.apply_perturbations(central_nodes)

    # return taget nodes
    def select_critical_nodes(self):
        centrality = nx.degree_centrality(self.graph)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    
        return [node for node, _ in sorted_nodes[:self.modification_budget]]

    # apply perturbations and return changed data
    def apply_perturbations(self, nodes):
        # for node in self.graph.nodes:
        #     if node >= self.data.x.size(0):
        #         print(f"Node {node} is missing features!")
        #         continue

        for node in nodes:
            # 随机生成新的特征
            new_feature = np.random.rand(self.data.x.size(1))
            self.data.x[node] = torch.tensor(new_feature, dtype=torch.float)

            # 随机移除一个邻居的边
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                edge_to_remove = (node, np.random.choice(neighbors))
                self.graph.remove_edge(*edge_to_remove)

        modified_data = from_networkx(self.graph)

        # 更新特征
        modified_data.x = self.data.x
