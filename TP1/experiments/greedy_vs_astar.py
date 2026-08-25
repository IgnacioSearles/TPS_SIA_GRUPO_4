import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_level
from algorithms import ALGORITHMS, HEURISTICS
from engine import run_search

def run_experiment(levels, algorithms, heuristics, runs=10):
    # results[level][algo][heuristic] = [times]
    results = {lvl: {algo: {h: [] for h in heuristics} for algo in algorithms} for lvl in levels}
    
    for lvl in levels:
        level_path = os.path.join(os.path.dirname(__file__), '..', 'levels', f"{lvl}.txt")
        print(f"Testing {lvl}...")
        for algo_name in algorithms:
            for h_name in heuristics:
                print(f"  {algo_name} + {h_name}: ", end="", flush=True)
                for i in range(runs):
                    initial_state, level = load_level(level_path)
                    
                    algo_class = ALGORITHMS[algo_name]
                    algorithm = algo_class(heuristic=HEURISTICS[h_name], pruning_mode="dead_squares")
                    
                    res = run_search(algorithm, initial_state, level)
                    
                    if not res.success:
                        print("F", end="", flush=True)
                        continue
                    
                    results[lvl][algo_name][h_name].append(res.processing_time_sec)
                    print(".", end="", flush=True)
                print()
    return results

def plot_results(results, levels, algorithms, heuristics):
    x = np.arange(len(levels))
    
    combos = [(a, h) for a in algorithms for h in heuristics]
    width = 0.8 / len(combos)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    greedy_colors = ['#9ecae1', '#4292c6', '#084594']
    astar_colors = ['#fcbba1', '#ef3b2c', '#99000d']
    
    color_map = {}
    for i, h in enumerate(heuristics):
        if "greedy" in algorithms:
            color_map[("greedy", h)] = greedy_colors[i % len(greedy_colors)]
        if "astar" in algorithms:
            color_map[("astar", h)] = astar_colors[i % len(astar_colors)]
            
    for i, (algo, h) in enumerate(combos):
        means = []
        stds = []
        for lvl in levels:
            times = results[lvl][algo][h]
            if times:
                means.append(np.mean(times))
                stds.append(np.std(times))
            else:
                means.append(0)
                stds.append(0)
                
        offset = (i - len(combos) / 2 + 0.5) * width
        
        color = color_map.get((algo, h), None)
        
        # Use capsize for error bars so they look good
        ax.bar(x + offset, means, width, yerr=stds, label=f"{algo.upper()} - {h.capitalize()}", color=color, edgecolor='black', capsize=5)
        
    ax.set_ylabel('Tiempo de Procesamiento (segundos) - Escala Logarítmica')
    ax.set_yscale('log')
    ax.set_title('Comparación de Tiempos de Ejecución: Greedy vs A* (10 ejecuciones)')
    ax.set_xticks(x)
    
    # Format labels from 'level-1' to 'Nivel 1'
    spanish_levels = [lvl.replace('level-', 'Nivel ') for lvl in levels]
    ax.set_xticklabels(spanish_levels)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'greedy_vs_astar_times.png')
    plt.savefig(out_path)
    print(f"\nPlot saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare Greedy vs A* execution times.")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per time sample")
    args = parser.parse_args()

    levels = ["level-1", "level-2", "level-3", "level-4"]
    algorithms = ["greedy", "astar"]
    
    # Selecting representative heuristics
    heuristics = ["manhattan", "hungarian", "push_distance"]
    
    print(f"Starting experiment with {args.runs} runs per configuration...")
    results = run_experiment(levels, algorithms, heuristics, runs=args.runs)
    plot_results(results, levels, algorithms, heuristics)

if __name__ == "__main__":
    main()
