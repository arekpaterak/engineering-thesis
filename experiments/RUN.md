# Saved commands

## Training

### TSP
Train with REINFORCE on TSP:

- with the MultiBinary action (but sampling k nodes to destroy):
```
python -m experiments.tsp.train.reinforce_multibinary --n-epochs 10 --max-t 50 --track --n-instances 1 --learning-rate 1e-5 --entropy-coefficient 0.5 --proportion 0.2
```

- with the Discrete action space:
```
python -m experiments.tsp.train.reinforce_discrete --seed 1 --n-epochs 5 --max-t 50 --track --n-instances 5 --learning-rate 1e-3
```

### CVRP
TODO

## Evaluation

### TSP
Evaluate a trained model on TSP:
```
python -m experiments.tsp.eval.trained_model --max-t 10 --proportion 0.2 --instance-name 20_1000_0 --instances-dir-name train
```

Evaluate the random baseline on TSP:
```
python -m experiments.tsp.eval.random_baseline --max-t 10 --proportion 0.5 --instance-name 20_1000_0 --instances-dir-name train
```

Evaluate the adaptive baseline on TSP:
```
python -m experiments.tsp.eval.adaptive_baseline --max-t 300 --initial-proportion 0.1 --adaptation-rate 0.05 --adaptation-timelimit-in-s 10 --instance-name 20_1000_0 --instances-dir-name train
```

### CVRP
TODO