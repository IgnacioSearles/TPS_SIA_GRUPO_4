import sys
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_level
from algorithms import ALGORITHMS, HEURISTICS
from engine import run_search

def get_path_states(initial_state, path, level, pruning_mode):
    states = [initial_state]
    current = initial_state
    for action in path:
        for successor_state, successor_action, _ in current.get_successors(level, pruning_mode):
            if successor_action == action:
                states.append(successor_state)
                current = successor_state
                break
    return states

def draw_state(ax, state, level, cost=None):
    ax.clear()
    
    # Calculate bounds
    all_points = level.walls | level.goals | state.boxes | {state.player_pos}
    if not all_points:
        return
        
    min_x = min(p[0] for p in all_points)
    max_x = max(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_y = max(p[1] for p in all_points)
    
    # Setup grid
    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(max_y + 1, min_y - 1)  # Inverted Y for proper grid layout
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw walls (grey squares)
    for wx, wy in level.walls:
        rect = plt.Rectangle((wx - 0.5, wy - 0.5), 1, 1, facecolor='gray', edgecolor='black')
        ax.add_patch(rect)
        
    # Draw goals (red dots)
    for gx, gy in level.goals:
        ax.plot(gx, gy, marker='o', markersize=10, color='red', alpha=0.5)
        
    # Draw boxes (brown squares)
    for bx, by in state.boxes:
        color = 'green' if (bx, by) in level.goals else 'saddlebrown'
        rect = plt.Rectangle((bx - 0.4, by - 0.4), 0.8, 0.8, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        
    # Draw player (blue circle)
    px, py = state.player_pos
    ax.plot(px, py, marker='o', markersize=15, color='blue')
    
    if cost is not None:
        ax.set_title(f"Costo (Movimientos): {cost}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Sokoban Levels and Solutions")
    parser.add_argument("level", help="Path to the level file")
    parser.add_argument("algorithm", nargs="?", default="bfs", help="Algorithm to use for solving")
    parser.add_argument("heuristic", nargs="?", default=None, help="Heuristic for A* or greedy")
    parser.add_argument(
        "pruning_mode",
        nargs="?",
        default="dead_squares",
        choices=("dead_squares", "local"),
        help="Deadlock pruning mode",
    )
    parser.add_argument("--static", action="store_true", help="Generate a static SVG of the initial state")
    parser.add_argument("--animate", action="store_true", help="Generate a GIF animation of the solution")
    parser.add_argument("--outdir", default="figures", help="Output directory")
    args = parser.parse_args()
    
    if not args.static and not args.animate:
        print("Please specify --static and/or --animate")
        sys.exit(1)
        
    os.makedirs(args.outdir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.level))[0]
    
    initial_state, level = load_level(args.level)
    algorithm_name = args.algorithm.lower()
    heuristic_name = args.heuristic.lower() if args.heuristic else "hungarian"

    if algorithm_name not in ALGORITHMS:
        valid_algorithms = ", ".join(sorted(ALGORITHMS))
        parser.error(
            f"unknown algorithm '{algorithm_name}'. Valid options: {valid_algorithms}"
        )

    if args.heuristic and algorithm_name not in {"astar", "greedy"}:
        parser.error("a heuristic can only be selected for 'astar' or 'greedy'")

    if algorithm_name in {"astar", "greedy"} and heuristic_name not in HEURISTICS:
        valid_heuristics = ", ".join(sorted(HEURISTICS))
        parser.error(
            f"unknown heuristic '{heuristic_name}'. Valid options: {valid_heuristics}"
        )
    
    if args.static:
        fig, ax = plt.subplots(figsize=(6, 6))
        draw_state(ax, initial_state, level)
        ax.set_title("Estado Inicial")
        out_path = os.path.join(args.outdir, f"{base_name}_static.svg")
        plt.savefig(out_path, format='svg', bbox_inches='tight')
        print(f"Saved static visualization to {out_path}")
        plt.close(fig)
        
    if args.animate:
        algo_class = ALGORITHMS[algorithm_name]
        if algorithm_name in {"astar", "greedy"}:
            algorithm = algo_class(
                heuristic=HEURISTICS[heuristic_name],
                pruning_mode=args.pruning_mode,
            )
        else:
            algorithm = algo_class(pruning_mode=args.pruning_mode)

        print(f"Solving with {algorithm_name.upper()}...")
        print(f"  Heuristic: {heuristic_name if algorithm_name in {'astar', 'greedy'} else 'n/a'}")
        print(f"  Pruning:   {args.pruning_mode}")
        result = run_search(algorithm, initial_state, level)
        if not result.success:
            print("No solution found to animate.")
            sys.exit(1)
            
        print(f"Solution found in {result.cost} moves. Generating animation...")
        states = get_path_states(
            initial_state,
            result.path,
            level,
            args.pruning_mode,
        )
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        def update(frame):
            draw_state(ax, states[frame], level, cost=frame)
            return []
            
        anim = FuncAnimation(fig, update, frames=len(states), interval=200, blit=True)
        
        if algorithm_name in {"astar", "greedy"}:
            solution_name = f"{algorithm_name}_{heuristic_name}_{args.pruning_mode}"
        else:
            solution_name = f"{algorithm_name}_{args.pruning_mode}"
        out_path = os.path.join(args.outdir, f"{base_name}_{solution_name}_solution.gif")
        try:
            anim.save(out_path, writer='pillow')
            print(f"Saved animation to {out_path}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("You might need to install pillow or imagemagick.")
            
        plt.close(fig)

if __name__ == "__main__":
    main()
