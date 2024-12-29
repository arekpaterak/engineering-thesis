import subprocess
from itertools import product

problem_sizes = [50, 100]
dir = "test"
instances_idx = range(100, 110) if dir == "test" else range(0, 10)
ks = [5]
seeds = range(10)

if __name__ == '__main__':
    for problem_size, k, instance_idx, seed in product(problem_sizes, ks, instances_idx, seeds):
        print(f"{problem_size}, {instance_idx}, {k}, {seed}")

        subprocess.run(f"python -m experiments.tsp.eval.random_baseline --max-t 50 --k {k} --instance-name {problem_size}_1000_{instance_idx} --instances-dir-name generated/{dir} --seed {seed} --solver gecode --no-debug")
