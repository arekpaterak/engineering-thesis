import subprocess
from itertools import product

problem_sizes = [50, 100]
dir = "train"
index_range = range(100, 110) if dir == "test" else range(10)
instance_parameters = "1123"
instances = [f"XML{problem_size}_{instance_parameters}_{idx:02d}" for problem_size, idx in product(problem_sizes, index_range)]
destruction_degrees = [5]
seeds = range(10)

if __name__ == '__main__':
    print(instances)

    for k, instance, seed in product(
        destruction_degrees, instances, seeds
    ):
        print(f"{instance=}, {k=}, {seed=}")

        subprocess.run(f"python -m experiments.cvrp.eval.random_baseline --max-t 50 --k {k} --instance-name {instance} --instances-dir-name generated/{dir} --seed {seed} --solver gecode --no-debug")
