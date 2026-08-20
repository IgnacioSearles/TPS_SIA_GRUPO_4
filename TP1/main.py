import sys
from utils import load_level
from algorithms import ALGORITHMS, HEURISTICS
from engine import run_search


def main():
    level_path = sys.argv[1] if len(sys.argv) > 1 else "levels/level-1.txt"
    algorithm_name = sys.argv[2].lower() if len(sys.argv) > 2 else "bfs"
    heuristic_name = sys.argv[3].lower() if len(sys.argv) > 3 else "hungarian"

    if algorithm_name not in ALGORITHMS:
        valid_algorithms = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unknown algorithm '{algorithm_name}'. Valid options: {valid_algorithms}")

    if len(sys.argv) > 3 and algorithm_name not in {"astar", "greedy"}:
        raise ValueError("A heuristic can only be selected for 'astar' or 'greedy'")

    if algorithm_name in {"astar", "greedy"} and heuristic_name not in HEURISTICS:
        valid_heuristics = ", ".join(sorted(HEURISTICS))
        raise ValueError(
            f"Unknown heuristic '{heuristic_name}'. Valid options: {valid_heuristics}"
        )

    print(f"Loading level: {level_path}")
    initial_state, level = load_level(level_path)
    print(f"  Player: {initial_state.player_pos}")
    print(f"  Boxes:  {set(initial_state.boxes)}")
    print(f"  Goals:  {set(level.goals)}")

    print(f"\nRunning {algorithm_name.upper()}...")
    algorithm_class = ALGORITHMS[algorithm_name]
    if algorithm_name in {"astar", "greedy"}:
        algorithm = algorithm_class(heuristic=HEURISTICS[heuristic_name])
        print(f"  Heuristic: {heuristic_name}")
    else:
        algorithm = algorithm_class()

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
