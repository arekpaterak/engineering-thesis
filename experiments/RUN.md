# Saved commands

## Training

### TSP
Train with REINFORCE on TSP:

- with the MultiBinary action space (but sampling k nodes to destroy):
```
python -m experiments.tsp.train.reinforce_multibinary --n-epochs 100 --max-t 100 --track --instances 0 --learning-rate 1e-5 --entropy-coefficient 0.1 --proportion 0.1 --max-grad-norm 2
```

### CVRP
TODO

## Evaluation

### TSP
Evaluate a trained model on TSP:
```
python -m experiments.tsp.eval.trained_model --max-t 50 --proportion 0.2 --instance-name 20_1000_10 --instances-dir-name train --model-tag latest --seed 13
```

Evaluate the random baseline on TSP:
```
python -m experiments.tsp.eval.random_baseline --max-t 50 --proportion 0.2 --instance-name 20_1000_0 --instances-dir-name train --seed 1 --processes 1 --solver cp-sat
```

Evaluate the adaptive baseline on TSP:
```
python -m experiments.tsp.eval.adaptive_baseline --max-t 300 --initial-proportion 0.1 --adaptation-rate 0.05 --adaptation-timelimit-in-s 10 --instance-name 20_1000_0 --instances-dir-name train
```

### CVRP
TODO