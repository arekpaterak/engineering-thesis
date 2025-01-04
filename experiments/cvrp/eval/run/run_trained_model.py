import subprocess
from itertools import product

problem_sizes = [20]
instances_idx = range(109, 110)
ks = [5]
seeds = range(10)

model_name = "model__k_4"
model_tag = "v51"

if __name__ == '__main__':
    for problem_size, k, instance_idx, seed in product(problem_sizes, ks, instances_idx, seeds):
        print(f"{problem_size}, {instance_idx}, {k}, {seed}")

        subprocess.run(f"python -m experiments.tsp.eval.trained_model --max-t 50 --k {k} --instance-name {problem_size}_1000_{instance_idx} --instances-dir-name generated/test --seed {seed} --solver gecode --no-debug --model-name {model_name} --model-tag {model_tag}")
