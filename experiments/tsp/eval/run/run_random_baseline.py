import subprocess
from itertools import product

problem_sizes = [50]
instances_idx = range(0, 1)
proportions = [0.1, 0.2, 0.3]
seeds = range(10)

if __name__ == '__main__':
    for problem_size, proportion, instance_idx, seed in product(problem_sizes, proportions, instances_idx, seeds):
        print(f"{problem_size}, {instance_idx}, {proportion}, {seed}")
        subprocess.run(f"python -m experiments.tsp.eval.random_baseline --max-t 50 --proportion {proportion} --instance-name {problem_size}_1000_{instance_idx} --instances-dir-name generated/train --seed {seed} --solver gecode --no-debug")
