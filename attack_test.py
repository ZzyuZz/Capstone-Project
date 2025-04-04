import argparse
import torch
from model import load_data, reset_model, train_model, test_model_accuracy, test_model_f1_score
from attack import StructureAttack, AdversarialAttack, FeatureAdversarialAttack

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_test(model_type, dataset='Cora', attack_type=None):
    data, num_features, num_classes, expected_dist = load_data(dataset)
    original_data = data
    model, optimizer, scheduler = reset_model(model_type, num_features, num_classes)
    
    # Attack type select
    if attack_type:
        if attack_type == 'label':
            data = StructureAttack.label_attack(data).to(device)
        elif attack_type == 'edge':
            data = StructureAttack.edge_attack(data).to(device)
        elif attack_type == 'fgsm':
            attacker = AdversarialAttack(model, data)
            data = attacker.FGSMattack().to(device)
        elif attack_type == 'keyNodeAttack':
            attacker = FeatureAdversarialAttack(model, data)
            data = attacker.attack().to(device)
    
    # train and test data
    train_model(model, data, optimizer, scheduler)
    accuracy = test_model_accuracy(model, original_data)
    f1score = test_model_f1_score(model, original_data)
    
    # get result 
    attack_desc = f"After {attack_type.replace('_', ' ').title()} Attack" if attack_type else "Original"
    return f"{model_type} {attack_desc} - Accuracy: {accuracy:.4f}, F1: {f1score:.4f}"

def main():
    # command setup
    parser = argparse.ArgumentParser(description='Model Robustness Test')
    parser.add_argument('--model', choices=['GCN', 'GAT', 'all'], default='all',
                       help='Model type to test (GCN, GAT, or all)')
    parser.add_argument('--attack', choices=['label', 'edge', 'fgsm', 'keyNodeAttack', 'all', 'none'], default='all',
                       help='Attack type to test (label, edge, fgsm, keyNodeAttack, all, or none)')
    parser.add_argument('--data', choices=['Cora', 'CiteSeer', 'PubMed'], default='Cora',
                       help='Dataset selected to test (Cora, CiteSeer, PubMed)')
    args = parser.parse_args()

    models = ['GCN', 'GAT'] if args.model == 'all' else [args.model]
    attacks = ['label', 'edge', 'fgsm', 'keyNodeAttack'] if args.attack == 'all' else (
        [args.attack] if args.attack != 'none' else []
    )

    print("Test dataset: "+args.data)

    for model_type in models:
        print("Test Model: "+model_type)
        print(run_test(model_type, args.data))
        
        for attack_type in attacks:
            print(run_test(model_type, args.data, attack_type))

if __name__ == "__main__":
    main()
