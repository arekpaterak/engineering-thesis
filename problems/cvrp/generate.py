import os
import subprocess
from dataclasses import dataclass

import tyro

from problems.cvrp.cvrp import CVRP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Args:
    n: int
    num_nodes: int
    depot_positioning: int = 1
    customer_positioning: int = 1
    demand_distribution: int = 2
    average_route_size: int = 3
    dir: str = "generated"
    starting_n: int = 0


if __name__ == '__main__':
    args = tyro.cli(Args)

    dir = os.path.join(BASE_DIR, "data", args.dir)

    print(dir)

    for instance_n in range(args.n):
        # instance_n is used as a seed
        instance_n = instance_n + args.starting_n
        subprocess.run(f"python {os.path.join(BASE_DIR, 'data', 'XML100/generator.py')} {args.num_nodes} {args.depot_positioning} {args.customer_positioning} {args.demand_distribution} {args.average_route_size} {instance_n} {instance_n} {dir}")

        vrp_filepath = os.path.join(dir, f"XML{args.num_nodes}_{args.depot_positioning}{args.customer_positioning}{args.demand_distribution}{args.average_route_size}_{instance_n:02d}.vrp")

        generated_instance = CVRP.parse_from_vrp_file(vrp_filepath)

        json_filepath = vrp_filepath.replace(".vrp", ".json")
        generated_instance.save(json_filepath, use_ints=True)
