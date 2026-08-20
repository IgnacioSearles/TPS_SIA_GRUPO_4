import sys
from utils import load_level
from algorithms import ALGORITHMS
from engine import run_search


def main():
    level_path = sys.argv[1] if len(sys.argv) > 1 else "levels/level-1.txt"
    algorithm_name = sys.argv[2].lower() if len(sys.argv) > 2 else "bfs"

    if algorithm_name not in ALGORITHMS:
        valid_algorithms = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unknown algorithm '{algorithm_name}'. Valid options: {valid_algorithms}")

    print(f"Loading level: {level_path}")
    initial_state, level = load_level(level_path)
    print(f"  Player: {initial_state.player_pos}")
    print(f"  Boxes:  {set(initial_state.boxes)}")
    print(f"  Goals:  {set(level.goals)}")

    print(f"\nRunning {algorithm_name.upper()}...")
    result = run_search(ALGORITHMS[algorithm_name](), initial_state, level)

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
