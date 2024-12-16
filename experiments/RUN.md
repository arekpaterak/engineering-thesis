# Saved commands

## Training

### TSP
Train with REINFORCE on TSP:

- with the MultiBinary action (but :
```
python -m experiments.tsp.train.reinforce_multibinary --n-epochs 50 --max-t 50 --track --n-instances 1 --learning-rate 1e-5 --entropy-coefficient 0.5
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
TODO
```

Evaluate the random baseline on TSP:
```
python -m experiments.tsp.eval.random_baseline --max-t 300 --proportion 0.5 --instance-name 20_1000_0 --instances-dir-name train
```

Evaluate the adaptive baseline on TSP:
```
TODO
```

### CVRP
TODO