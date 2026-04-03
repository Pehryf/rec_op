import csv
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from bench_single import load_csv, bench_one, WORKER_COUNTS

csv.field_size_limit(sys.maxsize)

CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dataset_raw", "chunks")
TSP_DIR = os.path.join(CHUNKS_DIR, "tsp_dataset")
WORLD_DIR = os.path.join(CHUNKS_DIR, "world")


def find_tsp_by_size(target_size):
    best_file, best_diff = None, float("inf")
    for f in sorted(os.listdir(TSP_DIR)):
        if not f.endswith(".csv"):
            continue
        with open(os.path.join(TSP_DIR, f)) as fh:
            row = next(csv.DictReader(fh))
            n = int(row["num_cities"])
            if abs(n - target_size) < best_diff:
                best_diff = abs(n - target_size)
                best_file = f
                if best_diff == 0:
                    break
    return os.path.join(TSP_DIR, best_file)


def main():
    # -- build dataset list --
    datasets = []

    for target in [20, 50, 100]:
        path = find_tsp_by_size(target)
        cities, best_known = load_csv(path)
        datasets.append((f"tsp_{len(cities)}", path, cities, best_known))

    world_path = os.path.join(WORLD_DIR, "world.csv")
    cities, best_known = load_csv(world_path)
    datasets.append((f"world_{len(cities)}", world_path, cities, best_known))

    # -- run benchmarks --
    all_results = {}
    for name, path, cities, best_known in datasets:
        print(f"\n--- {name} ({len(cities)} cities) ---")
        _, _, results = bench_one(path)
        all_results[name] = {"cities": cities, "best_known": best_known, "runs": results}

    # -- plot --
    n_datasets = len(datasets)
    n_agents = len(WORKER_COUNTS)
    fig, axes = plt.subplots(2 + n_agents, n_datasets, figsize=(5 * n_datasets, 5 * (2 + n_agents)))

    # row 1: convergence curves per dataset
    for col, (name, *_) in enumerate(datasets):
        ax = axes[0][col]
        data = all_results[name]
        for nb_agents in WORKER_COUNTS:
            run = data["runs"][nb_agents]
            ax.plot(run["history"], label=f"{nb_agents} agents")
        if data["best_known"]:
            ax.axhline(y=data["best_known"], color="green", linestyle="--", label="best known")
        ax.set_title(f"{name} ({len(data['cities'])} cities)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        ax.legend(fontsize=8)

    # row 2: bar charts - best cost + time
    for col, (name, *_) in enumerate(datasets):
        ax = axes[1][col]
        data = all_results[name]
        x = np.arange(n_agents)
        costs = [data["runs"][n]["best_cost"] for n in WORKER_COUNTS]
        times = [data["runs"][n]["time"] for n in WORKER_COUNTS]

        ax_time = ax.twinx()
        ax.bar(x - 0.15, costs, 0.3, color="steelblue")
        ax_time.bar(x + 0.15, times, 0.3, color="coral")

        if data["best_known"]:
            ax.axhline(y=data["best_known"], color="green", linestyle="--", linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in WORKER_COUNTS])
        ax.set_xlabel("Agents")
        ax.set_ylabel("Best cost", color="steelblue")
        ax_time.set_ylabel("Time (s)", color="coral")
        ax.set_title(f"{name} - cost vs time")

    # rows 3+: tour plots per (agent count, dataset)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for row_idx, nb_agents in enumerate(WORKER_COUNTS):
        for col, (name, *_) in enumerate(datasets):
            ax = axes[2 + row_idx][col]
            data = all_results[name]
            cities = data["cities"]
            run = data["runs"][nb_agents]
            turn = run["best_turn"]

            cx = [c[0] for c in cities]
            cy = [c[1] for c in cities]
            ax.scatter(cx, cy, c="black", s=10, zorder=3)

            route_x = [cities[i][0] for i in turn] + [cities[turn[0]][0]]
            route_y = [cities[i][1] for i in turn] + [cities[turn[0]][1]]
            ax.plot(route_x, route_y, c=colors[row_idx % len(colors)], linewidth=0.8, zorder=2)

            ax.set_title(f"{name} | {nb_agents} agents | cost: {run['best_cost']:.1f}")
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=150)
    print("\nSaved benchmark_results.png")
    plt.show()


if __name__ == "__main__":
    main()
