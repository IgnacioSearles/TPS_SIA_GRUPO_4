"""Runs the full algorithm x heuristic x level grid with per-run timeouts.

Each configuration is executed in its own subprocess (multiprocessing, spawn
start method) so that a run which explodes in time or memory can be killed
without taking down the rest of the benchmark or leaking memory into the
next run. Results are written incrementally to a CSV so partial progress
survives a crash/interrupt.
"""
import csv
import argparse
import os
import sys
import time
import multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_level
from algorithms import ALGORITHMS, HEURISTICS
from engine import run_search

RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'benchmark_results.csv')

FIELDS = [
    "level", "n_boxes", "n_goals", "algorithm", "heuristic", "pruning_mode",
    "success", "timed_out", "cost", "expanded_nodes", "frontier_nodes",
    "time_sec", "wall_time_sec", "timeout_budget_sec",
]


def _build_grid():
    levels = [f"level-{i}" for i in range(1, 5)] #levels 1-4
    configs = []

    # Main grid: dead_squares pruning throughout.
    for level in levels:
        configs.append(dict(level=level, algorithm="bfs", heuristic=None,
                             pruning_mode="dead_squares", timeout=300))
        configs.append(dict(level=level, algorithm="dfs", heuristic=None,
                             pruning_mode="dead_squares", timeout=300))
        configs.append(dict(level=level, algorithm="iddfs", heuristic=None,
                             pruning_mode="dead_squares", timeout=300))
        for heuristic in ("manhattan", "hungarian", "push_distance_nearest", "push_distance", "manhattan_player",
                          "hungarian_player", "push_distance_player", "push_distance_nearest_player",
                          "manhattan_player_all", "hungarian_player_all", "push_distance_player_all"):
            configs.append(dict(level=level, algorithm="astar", heuristic=heuristic,
                                 pruning_mode="dead_squares", timeout=450))
            configs.append(dict(level=level, algorithm="greedy", heuristic=heuristic,
                                 pruning_mode="dead_squares", timeout=450))

    # Pruning-mode comparison: same configs but with "local" pruning.
    for level in levels:
        configs.append(dict(level=level, algorithm="bfs", heuristic=None,
                             pruning_mode="local", timeout=300))
        configs.append(dict(level=level, algorithm="astar", heuristic="hungarian",
                             pruning_mode="local", timeout=450))
    return configs


def _worker(config, queue):
    try:
        level_path = os.path.join(os.path.dirname(__file__), '..', 'levels', f"{config['level']}.txt")
        initial_state, level = load_level(level_path)

        algo_class = ALGORITHMS[config["algorithm"]]
        if config["algorithm"] in {"astar", "greedy"}:
            algorithm = algo_class(heuristic=HEURISTICS[config["heuristic"]], pruning_mode=config["pruning_mode"])
        else:
            algorithm = algo_class(pruning_mode=config["pruning_mode"])

        result = run_search(algorithm, initial_state, level)
        queue.put({
            "n_boxes": len(initial_state.boxes),
            "n_goals": len(level.goals),
            "success": result.success,
            "cost": result.cost,
            "expanded_nodes": result.expanded_nodes,
            "frontier_nodes": result.frontier_nodes,
            "time_sec": result.processing_time_sec,
        })
    except Exception as exc:  # surface the error instead of a silent hang
        queue.put({"error": repr(exc)})


def run_with_timeout(config):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_worker, args=(config, queue))

    start = time.perf_counter()
    process.start()
    process.join(config["timeout"])
    wall_time = time.perf_counter() - start

    row = {
        "level": config["level"],
        "algorithm": config["algorithm"],
        "heuristic": config["heuristic"] or "",
        "pruning_mode": config["pruning_mode"],
        "timeout_budget_sec": config["timeout"],
        "wall_time_sec": round(wall_time, 4),
    }

    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        row.update({"n_boxes": "", "n_goals": "", "success": False, "timed_out": True,
                    "cost": "", "expanded_nodes": "", "frontier_nodes": "", "time_sec": ""})
        return row

    try:
        payload = queue.get_nowait()
    except Exception:
        payload = {"error": f"exit_code={process.exitcode}"}

    if "error" in payload:
        row.update({"n_boxes": "", "n_goals": "", "success": False, "timed_out": False,
                     "cost": "", "expanded_nodes": "", "frontier_nodes": "", "time_sec": "",
                     "error": payload["error"]})
        return row

    payload["timed_out"] = False
    row.update(payload)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Run the Sokoban algorithm/heuristic benchmark."
    )
    parser.add_argument(
        "--results-path",
        default=RESULTS_PATH,
        help=(
            "CSV output path. Defaults to the repository's "
            "results/benchmark_results.csv."
        ),
    )
    args = parser.parse_args()

    results_path = os.path.abspath(args.results_path)
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    configs = _build_grid()

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        f.flush()

        completed = 0
        total = len(configs)
        for cfg in configs:
            # Run configurations sequentially: the next one starts only after
            # the previous subprocess has finished and its result was written.
            row = run_with_timeout(cfg)
            writer.writerow(row)
            f.flush()
            completed += 1
            label = f"{cfg['algorithm']}" + (f"+{cfg['heuristic']}" if cfg['heuristic'] else "")
            status = "TIMEOUT" if row.get("timed_out") else ("OK" if row.get("success") else "FAIL")
            print(f"[{completed}/{total}] {cfg['level']} {label} ({cfg['pruning_mode']}) -> {status} "
                  f"wall={row['wall_time_sec']}s", flush=True)

    print(f"\nDone. Results written to {results_path}")


if __name__ == "__main__":
    main()
