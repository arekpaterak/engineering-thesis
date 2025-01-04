import os


BASE_PATH = "D:\\Coding\\University\\S7\\engineering-thesis"

DB_PATH = os.path.join(BASE_PATH, "experiments", "results.db")

# ==== TSP ====
TSP_DATA_DIR = os.path.join(BASE_PATH, "problems", "tsp", "data")

TSP_SOLVERS_DIR = os.path.join(BASE_PATH, "problems", "tsp", "minizinc")
TSP_INIT_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_init_circuit.mzn")
TSP_REPAIR_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_repair_circuit.mzn")

# ==== CVRP ====
CVRP_DATA_DIR = os.path.join(BASE_PATH, "problems", "cvrp", "data")

CVRP_SOLVERS_DIR = os.path.join(BASE_PATH, "problems", "cvrp", "minizinc")
CVRP_INIT_SOLVER_PATH = os.path.join(CVRP_SOLVERS_DIR, "cvrp_init.mzn")
CVRP_REPAIR_SOLVER_PATH = os.path.join(CVRP_SOLVERS_DIR, "cvrp_repair.mzn")
