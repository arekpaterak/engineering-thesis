import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro

from problems.tsp.tsp import TravelingSalesmanProblem


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Args:
    n: int
    num_nodes: int
    max_coordinate: int = 1000
    dir: str = os.path.join(BASE_DIR, "data", "generated")
    starting_n: int = 0
    use_ints: bool = True
    seed: int = 13


if __name__ == '__main__':
    args = tyro.cli(Args)

    np.random.seed(args.seed)

    for instance_n in range(args.n):
        filename = f"{args.num_nodes}_{args.max_coordinate}_{args.starting_n + instance_n}.json"

        generated_instance = TravelingSalesmanProblem.generate(
            num_nodes=args.num_nodes,
            max_coordinate=args.max_coordinate,
        )
        generated_instance.save(os.path.join(args.dir, filename), use_ints=args.use_ints)
