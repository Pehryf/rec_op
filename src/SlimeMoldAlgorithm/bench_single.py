import csv
import ast
import sys
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from graph import Graph
from population import Population

csv.field_size_limit(sys.maxsize)

WORKER_COUNTS = [10, 20, 40, 80]
T = 200
PHASE_LENGTH = 15
OPT_RATIO = 0.25
MAX_OPT_PASSES = 5


def load_csv(filepath):
    """Load cities from a CSV file. Supports two formats:
    - tsp_dataset: columns instance_id, num_cities, city_coordinates, ..., total_distance
    - world / solomon: columns with x,y or XCOORD.,YCOORD.
    Returns (cities, best_known) where best_known can be None.
    """
    with open(filepath) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        if "city_coordinates" in headers:
            row = next(reader)
            coords = ast.literal_eval(row["city_coordinates"])
            cities = [(c[0], c[1]) for c in coords]
            best_known = float(row["total_distance"]) if "total_distance" in headers else None
        else:
            x_col = next(h for h in headers if h.strip().lower() in ("x", "xcoord."))
            y_col = next(h for h in headers if h.strip().lower() in ("y", "ycoord."))
            cities = []
            for row in reader:
                cities.append((float(row[x_col]), float(row[y_col])))
            best_known = None

    return cities, best_known


def load_csvs(filepaths):
    """Load and merge cities from multiple CSV files."""
    all_cities = []
    for fp in filepaths:
        cities, _ = load_csv(fp)
        all_cities.extend(cities)
    return all_cities, None


def run_sma(graph, nb_agents, phase_length=PHASE_LENGTH, opt_ratio=OPT_RATIO, max_opt_passes=MAX_OPT_PASSES, live_ax=None, cities=None, color="red"):
    population = Population(graph, nb_agents)

    def on_step(t, pop, history, phase, stopped):
        best_cost = pop.best.cost
        if stopped:
            print(f"    iter {t:>3} [{phase:<11}] | cost: {best_cost:>10.1f} | STOP: stale", flush=True)
            return
        print(f"    iter {t:>3} [{phase:<11}] | cost: {best_cost:>10.1f}", flush=True)

        if live_ax and cities:
            ax_tour, ax_conv = live_ax
            ax_tour.cla()
            cx = [c[0] for c in cities]
            cy = [c[1] for c in cities]
            ax_tour.scatter(cx, cy, c="black", s=10, zorder=3)
            turn = pop.best.turn
            route_x = [cities[i][0] for i in turn] + [cities[turn[0]][0]]
            route_y = [cities[i][1] for i in turn] + [cities[turn[0]][1]]
            ax_tour.plot(route_x, route_y, c=color, linewidth=0.8, zorder=2)
            ax_tour.set_title(f"{nb_agents} agents | iter {t} | cost: {best_cost:.1f}")
            ax_tour.set_aspect("equal")

            ax_conv.cla()
            ax_conv.plot(history, c=color)
            ax_conv.set_xlabel("Iteration")
            ax_conv.set_ylabel("Cost")
            ax_conv.set_title("Convergence")
            plt.pause(0.01)

    return population.run(T, phase_length=phase_length, opt_ratio=opt_ratio, max_opt_passes=max_opt_passes, on_step=on_step)


def bench_one(filepath, worker_counts=None, live=False, phase_length=PHASE_LENGTH, opt_ratio=OPT_RATIO, max_opt_passes=MAX_OPT_PASSES):
    """Run benchmark on one or more CSVs. filepath can be a string or a list of strings.
    Returns (cities, best_known, results_dict)."""
    if worker_counts is None:
        worker_counts = WORKER_COUNTS

    if isinstance(filepath, list):
        cities, best_known = load_csvs(filepath)
    else:
        cities, best_known = load_csv(filepath)
    graph = Graph(cities)
    results = {}

    live_fig = None
    live_ax = None
    if live:
        plt.ion()
        live_fig, (ax_tour, ax_conv) = plt.subplots(1, 2, figsize=(12, 5))
        live_ax = (ax_tour, ax_conv)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for idx, nb_agents in enumerate(worker_counts):
        print(f"  Running {nb_agents} agents on {len(cities)} cities...", flush=True)
        color = colors[idx % len(colors)]
        start = time.time()
        best_cost, iters, history, best_turn = run_sma(
            graph, nb_agents,
            phase_length=phase_length, opt_ratio=opt_ratio, max_opt_passes=max_opt_passes,
            live_ax=live_ax if live else None,
            cities=cities if live else None,
            color=color,
        )
        elapsed = time.time() - start
        results[nb_agents] = {
            "best_cost": best_cost,
            "iters": iters,
            "time": elapsed,
            "history": history,
            "best_turn": best_turn,
        }
        print(f"  {nb_agents:>3} agents | cost: {best_cost:>10.2f} | iters: {iters:>4} | {elapsed:.2f}s")

    if live:
        plt.ioff()
        plt.close(live_fig)

    return cities, best_known, results


def plot_single(cities, best_known, results, title="benchmark", worker_counts=None):
    """Plot convergence, cost vs time bars, and tour for each agent count."""
    if worker_counts is None:
        worker_counts = sorted(results.keys())

    n_agents = len(worker_counts)
    fig, axes = plt.subplots(2, n_agents, figsize=(5 * n_agents, 10))
    if n_agents == 1:
        axes = [[axes[0]], [axes[1]]]

    # row 1: tour per agent count
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for col, nb_agents in enumerate(worker_counts):
        ax = axes[0][col]
        run = results[nb_agents]
        turn = run["best_turn"]

        cx = [c[0] for c in cities]
        cy = [c[1] for c in cities]
        ax.scatter(cx, cy, c="black", s=10, zorder=3)

        route_x = [cities[i][0] for i in turn] + [cities[turn[0]][0]]
        route_y = [cities[i][1] for i in turn] + [cities[turn[0]][1]]
        ax.plot(route_x, route_y, c=colors[col % len(colors)], linewidth=0.8, zorder=2)

        ax.set_title(f"{nb_agents} agents | cost: {run['best_cost']:.1f}")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)

    # row 2: convergence curves all on one plot + bar chart
    mid = n_agents // 2
    ax_conv = axes[1][mid]
    for col, nb_agents in enumerate(worker_counts):
        run = results[nb_agents]
        ax_conv.plot(run["history"], label=f"{nb_agents} agents", color=colors[col % len(colors)])
    if best_known:
        ax_conv.axhline(y=best_known, color="green", linestyle="--", label="best known")
    ax_conv.set_title(f"{title} - convergence")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Cost")
    ax_conv.legend(fontsize=8)

    # bar chart on another subplot
    bar_idx = 0 if mid != 0 else min(1, n_agents - 1)
    ax_bar = axes[1][bar_idx]
    x = np.arange(n_agents)
    costs = [results[n]["best_cost"] for n in worker_counts]
    times = [results[n]["time"] for n in worker_counts]
    ax_time = ax_bar.twinx()
    ax_bar.bar(x - 0.15, costs, 0.3, color="steelblue")
    ax_time.bar(x + 0.15, times, 0.3, color="coral")
    if best_known:
        ax_bar.axhline(y=best_known, color="green", linestyle="--", linewidth=1)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([str(n) for n in worker_counts])
    ax_bar.set_xlabel("Agents")
    ax_bar.set_ylabel("Best cost", color="steelblue")
    ax_time.set_ylabel("Time (s)", color="coral")
    ax_bar.set_title(f"{title} - cost vs time")

    # hide unused subplots
    used = {mid, bar_idx}
    for col in range(n_agents):
        if col not in used:
            axes[1][col].axis("off")

    plt.suptitle(f"{title} ({len(cities)} cities)", fontsize=14)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SMA on a single CSV")
    parser.add_argument("csv_file", nargs="+", help="Path(s) to CSV file(s) — multiple files are merged")
    parser.add_argument("--agents", nargs="+", type=int, default=WORKER_COUNTS, help="Agent counts to test")
    parser.add_argument("-o", "--output", default=None, help="Save plot to file")
    parser.add_argument("--live", action="store_true", help="Show live tour and convergence during run")
    parser.add_argument("--phase-length", type=int, default=PHASE_LENGTH, help=f"Phase length (default: {PHASE_LENGTH})")
    parser.add_argument("--opt-ratio", type=float, default=OPT_RATIO, help=f"2-opt ratio within phase (default: {OPT_RATIO})")
    parser.add_argument("--max-opt-passes", type=int, default=MAX_OPT_PASSES, help=f"Max 2-opt passes per iteration (default: {MAX_OPT_PASSES})")
    args = parser.parse_args()

    files = args.csv_file
    if len(files) == 1:
        filepath = files[0]
        title = filepath.split("/")[-1].replace(".csv", "")
    else:
        filepath = files
        title = f"merged_{len(files)}files"

    print(f"Benchmarking {len(files)} file(s) (phase_length={args.phase_length}, opt_ratio={args.opt_ratio}, max_opt_passes={args.max_opt_passes})")
    cities, best_known, results = bench_one(filepath, args.agents, live=args.live, phase_length=args.phase_length, opt_ratio=args.opt_ratio, max_opt_passes=args.max_opt_passes)
    fig = plot_single(cities, best_known, results, title=title, worker_counts=args.agents)

    out = args.output or f"bench_{title}.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    plt.show()
