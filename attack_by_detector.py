import argparse
from model import load_data, reset_model, train_model, test_model_accuracy, test_model_f1_score, adversarial_train
from attack import StructureAttack, AdversarialAttack
from defend import detect_label_attack, detect_edge_attack


def run_test(model_type, attack_type=None):
    data, num_features, num_classes, expected_dist = load_data()
    original_data = data.clone()
    model, optimizer, scheduler = reset_model(model_type, num_features, num_classes)
    
    # Attack type select
    if attack_type:
        if attack_type == 'label':
            data = StructureAttack.label_attack(data)
        elif attack_type == 'edge':
            data = StructureAttack.edge_attack(data)
        elif attack_type == 'fgsm':
            attacker = AdversarialAttack(model, data)
            data = attacker.FGSMattack()

    if detect_label_attack(data, expected_dist):
        return "Label attack detected !"
    elif detect_edge_attack(original_data, data):
        return "Edge attack detected !"
    
    # train by dector and adversarial train, 
    adversarial_train(model, original_data, optimizer, scheduler)
    model1, optimizer, scheduler = reset_model(model_type, num_features, num_classes)
    train_model(model, data, optimizer, scheduler)
    # test
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
    parser.add_argument('--attack', choices=['label', 'edge', 'fgsm', 'all', 'none'], default='all',
                       help='Attack type to test (label, edge, fgsm, all, or none)')
    args = parser.parse_args()

    models = ['GCN', 'GAT'] if args.model == 'all' else [args.model]
    attacks = ['label', 'edge', 'fgsm'] if args.attack == 'all' else (
        [args.attack] if args.attack != 'none' else []
    )

    for model_type in models:
        print(run_test(model_type))
        
        for attack_type in attacks:
            print(run_test(model_type, attack_type))

if __name__ == "__main__":
    main()

