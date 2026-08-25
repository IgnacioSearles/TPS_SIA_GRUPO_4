import argparse
import sys
from utils import load_level
from algorithms import ALGORITHMS, HEURISTICS
from engine import run_search


def main():
    parser = argparse.ArgumentParser(description="Sokoban Solver")
    parser.add_argument("--level", type=str, default="levels/level-1.txt", help="Path to the level file")
    parser.add_argument("--algorithm", type=str, default="bfs", choices=list(ALGORITHMS.keys()), help="Search algorithm to use")
    parser.add_argument("--heuristic", type=str, default="hungarian", choices=list(HEURISTICS.keys()), help="Heuristic function (only for A* and Greedy)")
    parser.add_argument("--pruning", type=str, default="dead_squares", choices=["dead_squares", "local"], help="Pruning mode")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    level_path = args.level
    algorithm_name = args.algorithm.lower()
    heuristic_name = args.heuristic.lower()
    pruning_mode = args.pruning.lower()

    if any(arg.startswith("--heuristic") for arg in sys.argv) and algorithm_name not in {"astar", "greedy"}:
        parser.error("A heuristic can only be selected for 'astar' or 'greedy'")

    print(f"Loading level: {level_path}")
    initial_state, level = load_level(level_path)
    print(f"  Player: {initial_state.player_pos}")
    print(f"  Boxes:  {set(initial_state.boxes)}")
    print(f"  Goals:  {set(level.goals)}")

    print(f"\nRunning {algorithm_name.upper()}...")
    algorithm_class = ALGORITHMS[algorithm_name]
    if algorithm_name in {"astar", "greedy"}:
        algorithm = algorithm_class(
            heuristic=HEURISTICS[heuristic_name],
            pruning_mode=pruning_mode,
        )
        print(f"  Heuristic: {heuristic_name}")
    else:
        algorithm = algorithm_class(pruning_mode=pruning_mode)

    print(f"  Pruning:   {pruning_mode}")

    result = run_search(algorithm, initial_state, level)

    if not result.success:
        print("No solution found.")
        return

    print(f"  Solved in {result.processing_time_sec:.3f}s")
    print(f"  Cost (moves):     {result.cost}")
    print(f"  Expanded nodes:   {result.expanded_nodes}")
    print(f"  Frontier nodes:   {result.frontier_nodes}")
    print(f"  Path: {' '.join(a.char for a in result.path)}")

if __name__ == "__main__":
    main()
