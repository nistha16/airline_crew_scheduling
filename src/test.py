# import csv
import statistics
import sys
import time
from pathlib import Path

from utils import load_spp_instance
from simulated_annealing import run_sa
from ga_standard import run_standard_bga
from improved_ga import run_improved_bga

OPTIMAL = {
    "sppnw41": 10972.50,
    "sppnw42": 7485.00,
    "sppnw43": 8897.00,
}

NUM_RUNS = 30


# def save_csv(filename, seeds, costs):
#     with open(filename, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["seed", "best_feasible_cost"])
#         for seed, cost in zip(seeds, costs):
#             writer.writerow([seed, cost if cost is not None else "INFEASIBLE"])


def summarize(name, algo, costs):
    opt = OPTIMAL.get(name)
    feasible = [c for c in costs if c is not None]

    print(f"\n--- {algo} on {name} ---")
    print(f"Feasible: {len(feasible)}/{len(costs)}")

    if feasible:
        mean_val = statistics.mean(feasible)
        std_val = statistics.stdev(feasible) if len(feasible) > 1 else 0.0
        print(f"Mean: {mean_val:.2f}")
        print(f"Std:  {std_val:.2f}")
        print(f"Best: {min(feasible)}")
        print(f"Worst: {max(feasible)}")
        if opt:
            print(f"Optimal: {opt}")
            print(f"Best gap:  {100*(min(feasible) - opt)/opt:.2f}%")
            print(f"Mean gap:  {100*(mean_val - opt)/opt:.2f}%")
    else:
        print("No feasible solutions found.")


def run_experiment(name, algo_label, run_fn, inst):
    costs = []
    t0 = time.time()
    opt = OPTIMAL.get(name)

    for seed in range(NUM_RUNS):
        cost = run_fn(inst, seed)
        costs.append(cost)
        gap = f" gap={100*(cost - opt)/opt:.2f}%" if cost is not None and opt else ""
        print(f"  {algo_label} seed {seed:02d}: {cost}{gap}")

    # save_csv(f"{algo_label.lower().replace(' ', '_')}_results_{name}.csv",
            #  list(range(NUM_RUNS)), costs)
    summarize(name, algo_label, costs)
    print(f"  Time: {time.time() - t0:.1f}s")


USAGE = """Usage: python3 src/test.py <data_file> <algorithm>

  algorithm: sa, bga, ibga, or all

Examples:
  python3 src/test.py data/sppnw41.txt sa
  python3 src/test.py data/sppnw42.txt bga
  python3 src/test.py data/sppnw43.txt all
"""

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(USAGE)
        sys.exit(1)

    path = Path(sys.argv[1])
    algo = sys.argv[2]

    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    inst = load_spp_instance(path, strict=True)
    name = inst.name
    print(f"\nFile: {name} (m={inst.m}, n={inst.n})")
    print(f"{'='*50}")

    if algo in ("all", "sa"):
        run_experiment(name, "SA", run_sa, inst)
    if algo in ("all", "bga"):
        run_experiment(name, "BGA", run_standard_bga, inst)
    if algo in ("all", "ibga"):
        run_experiment(name, "IBGA", run_improved_bga, inst)
