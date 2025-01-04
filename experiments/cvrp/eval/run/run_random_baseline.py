import subprocess
from itertools import product

problem_sizes = [50, 100]
dir = "test"
instances = []
destruction_degrees = [5]
seeds = range(10)

if __name__ == '__main__':
    for problem_size, k, instance, seed in product(
        problem_sizes, destruction_degrees, instances, seeds
    ):
        print(f"{problem_size}, {instance}, {k}, {seed}")

        subprocess.run(f"python -m experiments.cvrp.eval.random_baseline --max-t 50 --k {k} --instance-name {instance} --instances-dir-name generated/{dir}/XMLike{problem_size} --seed {seed} --solver gecode --no-debug")
