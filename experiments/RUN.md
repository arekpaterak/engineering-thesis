# Saved commands

## Training

### TSP
Train with REINFORCE on the TSP:

- with the MultiBinary action space (but sampling k nodes to destroy):

n=20
```
python -m experiments.tsp.train.reinforce_multibinary --n-epochs 100 --max-t 10 --track --instances 0 9 --learning-rate 1e-3 --entropy-coefficient 0.0 --k 5 --max-grad-norm 2 --num-layers 3 --gat-v2 --num-heads 8 --no-fully-connected --no-normalize-returns --dropout 0.1
```

n=50
```
python -m experiments.tsp.train.reinforce_multibinary --n-epochs 10 --max-t 50 --track --instances 0 --problem-sizes 50 --learning-rate 1e-5 --entropy-coefficient 5.0 --proportion 0.1 --max-grad-norm 2
```

n=100
```
python -m experiments.tsp.train.reinforce_multibinary --n-epochs 10 --max-t 50 --track --instances 0 --problem-sizes 100 --learning-rate 1e-5 --entropy-coefficient 1.0 --proportion 0.2 --max-grad-norm 2
```

### CVRP

Train with REINFORCE on the CVRP:

n=20
```
python -m experiments.cvrp.train.reinforce --n-epochs 100 --max-t 10 --track --instances 0 9 --learning-rate 1e-3 --entropy-coefficient 0.0 --k 5 --max-grad-norm 2 --num-layers 3 --gat-v2 --num-heads 8 --no-normalize-returns --problem-sizes 20 --gamma 0.9 --dropout 0.1
```

n=100
```
python -m experiments.cvrp.train.reinforce --n-epochs 100 --max-t 10 --track --instances 0 9 --learning-rate 1e-3 --entropy-coefficient 0.0 --k 5 --max-grad-norm 2 --num-layers 3 --gat-v2 --num-heads 8 --no-normalize-returns --problem-sizes 100 --gamma 0.9 --dropout 0.1
```

---

## Evaluation

### TSP
Evaluate a trained model on the TSP:
```
python -m experiments.tsp.eval.trained_model --max-t 50 --k 5 --instance-name 20_1000_0 --instances-dir-name generated/train --model-tag v9 --seed 0
```

```
python -m experiments.tsp.eval.trained_model --max-t 50 --k 5 --instance-name eil51 --instances-dir-name tsplib --model-tag v9 --seed 0
```

Evaluate the random baseline on the TSP:
```
python -m experiments.tsp.eval.random_baseline --max-t 50 --k 2 --instance-name 20_1000_0 --instances-dir-name generated/train --seed 0
```

Evaluate the adaptive baseline on the TSP:
```
python -m experiments.tsp.eval.adaptive_baseline --max-t 300 --initial-proportion 0.1 --adaptation-rate 0.05 --adaptation-timelimit-in-s 10 --instance-name 20_1000_0 --instances-dir-name train
```

### CVRP

Evaluate a trained model on the CVRP:
```
python -m experiments.cvrp.eval.trained_model --max-t 50 --k 5 --instances XML50_1123_109 --instances-dir-name generated/test --model-tag v58 --seed 0
```

Evaluate the random baseline on the CVRP:
```
python -m experiments.cvrp.eval.random_baseline --max-t 50 --k 6 --instance-name XML20_1123_00 --instances-dir-name generated/train --seed 0
```
