# Saved commands

## Training

Train with REINFORCE on TSP:

```
python -m experiments.tsp.train.reinforce --n_epochs 50 --max-t 50 --track --n_envs 5 --learning-rate 2.5e-4
```

## Evaluation

Evaluate a trained agent on TSP:
TODO

Evaluate the random baseline on TSP:
```
python -m experiments.tsp.eval.random_baseline --max-t 50 --track --proportion 0.5
```

Evaluate the adaptive baseline on TSP:
TODO
