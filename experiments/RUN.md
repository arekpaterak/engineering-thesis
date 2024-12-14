# Saved commands

## Training

### TSP
Train with REINFORCE on TSP:

- with the MultiBinary action space:
```
python -m experiments.tsp.train.reinforce_multibinary --n-epochs 5 --max-t 50 --track --n-instances 1 --learning-rate 1e-5
```

- with the Discrete action space:
```
python -m experiments.tsp.train.reinforce_discrete --seed 1 --n-epochs 5 --max-t 50 --track --n-instances 5 --learning-rate 1e-3
```

### CVRP
TODO

## Evaluation

### TSP
Evaluate a trained agent on TSP:
```
TODO
```

Evaluate the random baseline on TSP:
```
python -m experiments.tsp.eval.random_baseline --max-t 50 --track --proportion 0.5
```

Evaluate the adaptive baseline on TSP:
```
TODO
```
### CVRP
TODO