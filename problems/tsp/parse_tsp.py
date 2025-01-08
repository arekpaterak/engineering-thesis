from dataclasses import dataclass

import tyro

from config import *
from problems.tsp.tsp import TravelingSalesmanProblem


@dataclass
class Args:
    instance_name: str = "eil51"
    dir: str = "tsplib"


if __name__ == '__main__':
    args = tyro.cli(Args)

    dir = os.path.join(TSP_DATA_DIR, args.dir)

    filename = f"{args.instance_name}.json"

    tsp_filepath = os.path.join(
        dir,
        f"{args.instance_name}.tsp"
    )

    generated_instance = TravelingSalesmanProblem.parse_from_tsp_file(tsp_filepath)

    generated_instance.save(os.path.join(dir, filename), use_ints=True)
