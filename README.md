# Capstone-Project
Capstone Project Assignment( ZK3)- An Empirical Study of Attacks Against Graph Learning Models<br>
Author： Hu Jun, Louis
## Environment establishment(Optional)
```
python -m venv .venv
.venv\Scripts\activate
```

## Create an project environment
```
pip install -r requirements.txt
```

## Run
<hr>
### Attak Test
```
python attack_test.py 
```
You can add the following instructions to command:<br>
--model : to select model GCN or GAT (default: GCN)<br>
--attack : to select the attack type to test (label, edge, fgsm, keyNodeAttack, all, or none)<br>
--data : to select the dataset (Cora, CiteSeer, PubMed)<br>
<br>
### Attak Test by Detector
```
python attack_by_detector.py 
```
You can add the following instructions to command:<br>
--model : to select model GCN or GAT (default: GCN)<br>
--attack : to select the attack type to test (label, edge, fgsm, keyNodeAttack, all, or none)<br>
--data : to select the dataset (Cora, CiteSeer, PubMed)<br>
<br>
### Model Rendered Image
```
python model_rendering.py 
```
You can add the following instructions to command:<br>
--model : to select model GCN or GAT (default: GCN)<br>
