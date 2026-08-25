"""
Interactive entry point for the Sokoban solver.

Lets you pick, via a simple menu, the level, search algorithm, heuristic
(when applicable) and deadlock-pruning mode, then runs the search and
optionally saves a static SVG and/or an animated GIF of the solution.

Usage:
    python interactive_main.py [--levels-dir levels] [--outdir figures]
"""

import os
import sys
import glob
import argparse

from utils import load_level
from algorithms import ALGORITHMS, HEURISTICS
from engine import run_search

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ── Menu helpers ──────────────────────────────────────────────────────────

def prompt_choice(label: str, options, default=None):
    options = list(options)
    print(f"\n{label}:")
    for i, opt in enumerate(options, 1):
        marker = "  (default)" if opt == default else ""
        print(f"  {i}. {opt}{marker}")

    suffix = f" (default {default})" if default is not None else ""
    while True:
        raw = input(f"Select [1-{len(options)}]{suffix}: ").strip()
        if not raw and default is not None:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("Invalid choice, try again.")


def select_level(levels_dir: str) -> str:
    levels = sorted(glob.glob(os.path.join(levels_dir, "*.txt"))) if os.path.isdir(levels_dir) else []
    if levels:
        return prompt_choice("Select a level", levels, default=levels[0])
    return input("Enter path to level file: ").strip()


def select_algorithm() -> str:
    return prompt_choice("Select algorithm", sorted(ALGORITHMS), default="bfs")


def select_heuristic() -> str:
    return prompt_choice("Select heuristic", sorted(HEURISTICS), default="hungarian")


def select_pruning_mode() -> str:
    return prompt_choice("Select deadlock pruning mode", ["dead_squares", "local"], default="dead_squares")


def select_visualization() -> str:
    return prompt_choice("Visualization", ["none", "static", "animate", "both"], default="both")


# ── Visualization (adapted from 02_visualize.py) ─────────────────────────

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

    all_points = level.walls | level.goals | state.boxes | {state.player_pos}
    if not all_points:
        return

    min_x = min(p[0] for p in all_points)
    max_x = max(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_y = max(p[1] for p in all_points)

    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(max_y + 1, min_y - 1)  # inverted Y for a natural grid layout
    ax.set_aspect('equal')
    ax.axis('off')

    for wx, wy in level.walls:
        rect = plt.Rectangle((wx - 0.5, wy - 0.5), 1, 1, facecolor='gray', edgecolor='black')
        ax.add_patch(rect)

    for gx, gy in level.goals:
        ax.plot(gx, gy, marker='o', markersize=10, color='red', alpha=0.5)

    for bx, by in state.boxes:
        color = 'green' if (bx, by) in level.goals else 'saddlebrown'
        rect = plt.Rectangle((bx - 0.4, by - 0.4), 0.8, 0.8, facecolor=color, edgecolor='black')
        ax.add_patch(rect)

    px, py = state.player_pos
    ax.plot(px, py, marker='o', markersize=15, color='blue')

    if cost is not None:
        ax.set_title(f"Costo (Movimientos): {cost}")


def save_static(initial_state, level, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_state(ax, initial_state, level)
    ax.set_title("Estado Inicial")
    plt.savefig(out_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved static visualization to {out_path}")


def save_animation(initial_state, level, path, pruning_mode, out_path):
    states = get_path_states(initial_state, path, level, pruning_mode)
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        draw_state(ax, states[frame], level, cost=frame)
        return []

    anim = FuncAnimation(fig, update, frames=len(states), interval=200, blit=True)
    try:
        anim.save(out_path, writer='pillow')
        print(f"Saved animation to {out_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("You might need to install pillow or imagemagick.")
    plt.close(fig)


# ── Main flow ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactively solve and visualize a Sokoban level")
    parser.add_argument("--levels-dir", default="levels", help="Directory to look for level files")
    parser.add_argument("--outdir", default="figures", help="Output directory for visualizations")
    args = parser.parse_args()

    level_path = select_level(args.levels_dir)
    algorithm_name = select_algorithm()

    heuristic_name = None
    if algorithm_name in {"astar", "greedy"}:
        heuristic_name = select_heuristic()

    pruning_mode = select_pruning_mode()
    viz_choice = select_visualization()

    print(f"\nLoading level: {level_path}")
    initial_state, level = load_level(level_path)
    print(f"  Player: {initial_state.player_pos}")
    print(f"  Boxes:  {set(initial_state.boxes)}")
    print(f"  Goals:  {set(level.goals)}")

    algorithm_class = ALGORITHMS[algorithm_name]
    if algorithm_name in {"astar", "greedy"}:
        algorithm = algorithm_class(heuristic=HEURISTICS[heuristic_name], pruning_mode=pruning_mode)
    else:
        algorithm = algorithm_class(pruning_mode=pruning_mode)

    print(f"\nRunning {algorithm_name.upper()}...")
    print(f"  Heuristic: {heuristic_name or 'n/a'}")
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

    if viz_choice == "none":
        return

    os.makedirs(args.outdir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(level_path))[0]
    if algorithm_name in {"astar", "greedy"}:
        solution_name = f"{algorithm_name}_{heuristic_name}_{pruning_mode}"
    else:
        solution_name = f"{algorithm_name}_{pruning_mode}"

    if viz_choice in {"static", "both"}:
        save_static(initial_state, level, os.path.join(args.outdir, f"{base_name}_static.svg"))

    if viz_choice in {"animate", "both"}:
        save_animation(
            initial_state, level, result.path, pruning_mode,
            os.path.join(args.outdir, f"{base_name}_{solution_name}_solution.gif"),
        )


if __name__ == "__main__":
    main()