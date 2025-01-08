import subprocess
from itertools import product

problem_sizes = [100]
dir = "test"
index_range = range(100, 110) if dir == "test" else range(10)
instance_parameters = "1123"
instances = [f"XML{problem_size}_{instance_parameters}_{idx:02d}" for problem_size, idx in product(problem_sizes, index_range)]
destruction_degrees = [5]
seeds = range(10)

greedy = False

model_name = "cvrp_model__k_5"
model_tag = "v58"


if __name__ == '__main__':
    for k, seed in product(
        destruction_degrees, seeds
    ):
        print(f"{k=}, {seed=}")

        subprocess.run(f"python -m experiments.cvrp.eval.trained_model --max-t 50 --k {k} --instances {' '.join(instances)} --instances-dir-name generated/{dir} --seed {seed} --no-debug --model-name {model_name} --model-tag {model_tag} {'--greedy' if greedy else ''}")
