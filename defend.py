import numpy as np
import torch.nn.functional as F

## Detector ##
# Detect label attack
def detect_label_attack(attacked_data, expected_dist, tolerance=0.01):
    attacked_train_labels = attacked_data.y[attacked_data.train_mask].cpu().numpy()
    unique, counts = np.unique(attacked_train_labels, return_counts=True)

    attacked_dist = np.zeros_like(expected_dist)
    attacked_dist[unique] = counts / counts.sum()
    
    deviation = np.abs(attacked_dist - expected_dist)
    max_deviation = deviation.max()
    
    if max_deviation > tolerance:
        # print(f"Label attack detected! Maximum deviation: {max_deviation:.2f}")
        return True
    return False

# Detect edge attack
def detect_edge_attack(original_data, attacked_data, tolerance=0.01):
    original_set = set(map(tuple, original_data.edge_index.t().tolist()))
    attacked_set = set(map(tuple, attacked_data.edge_index.t().tolist()))

    intersection = original_set.intersection(attacked_set)
    union = original_set.union(attacked_set)
    if 1.0 - len(intersection) / len(union)  > tolerance:
        # print(f"Edge attack detected! Change ratio: {1 - len(intersection)/len(union):.2f}")
        return True
    return False