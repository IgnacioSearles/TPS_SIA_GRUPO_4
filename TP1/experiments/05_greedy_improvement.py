import sys
import os
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_level
from algorithms import ALGORITHMS, HEURISTICS
from engine import run_search

def run_no_timeout(algo_name, heuristic_name, level_path):
    initial_state, level = load_level(level_path)
    if algo_name in ("astar", "greedy"):
        algorithm = ALGORITHMS[algo_name](heuristic=HEURISTICS[heuristic_name])
    else:
        algorithm = ALGORITHMS[algo_name]()
    return run_search(algorithm, initial_state, level)

def heuristic_label(value: str) -> str:
    labels = {
        "manhattan": "Manhattan",
        "hungarian": "Hungarian",
        "push_distance_nearest": "Push distance nearest",
        "push_distance": "Push distance",
    }
    if value.endswith("_player_all"):
        base = value.removesuffix("_player_all")
        return f"{labels.get(base, base)} + jugador (todas)"
    if value.endswith("_player"):
        base = value.removesuffix("_player")
        return f"{labels.get(base, base)} + jugador (no resueltas)"
    return labels.get(value, value)

def main():
    levels = [1, 2, 3, 4]
    heuristics = [h for h in HEURISTICS.keys() if not h.endswith("_unsolved")]
    runs = 10
    
    # Storage for times
    bfs_times = {l: [] for l in levels}
    greedy_times = {h: {l: [] for l in levels} for h in heuristics}
    
    print("--- Recolectando tiempos para BFS ---")
    for l in levels:
        level_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'levels', f'level-{l}.txt'))
        print(f"Level {l}: ", end='', flush=True)
        for r in range(runs):
            res = run_no_timeout("bfs", None, level_path)
            if res:
                bfs_times[l].append(res.processing_time_sec)
                print(f"{res.processing_time_sec:.3f}s ", end='', flush=True)
        print()
        
    print("\n--- Recolectando tiempos para Greedy ---")
    for h in heuristics:
        for l in levels:
            level_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'levels', f'level-{l}.txt'))
            print(f"Greedy {h} | Level {l}: ", end='', flush=True)
            for r in range(runs):
                res = run_no_timeout("greedy", h, level_path)
                if res:
                    greedy_times[h][l].append(res.processing_time_sec)
                    print(f"{res.processing_time_sec:.3f}s ", end='', flush=True)
            print()

    # Calculate improvement matrix and error
    I_matrix = np.full((len(heuristics), len(levels)), np.nan)
    err_matrix = np.full((len(heuristics), len(levels)), np.nan)
    
    for c, l in enumerate(levels):
        if not bfs_times[l]:
            continue
        mu_bfs = np.mean(bfs_times[l])
        std_bfs = np.std(bfs_times[l])
        
        for r, h in enumerate(heuristics):
            if not greedy_times[h][l]:
                continue
            mu_greedy = np.mean(greedy_times[h][l])
            std_greedy = np.std(greedy_times[h][l])
            
            # Improvement I = 1 - (mu_greedy / mu_bfs)
            I = 1.0 - (mu_greedy / mu_bfs)
            
            # Error propagation
            if mu_greedy > 0 and mu_bfs > 0:
                err = (mu_greedy / mu_bfs) * np.sqrt((std_greedy / mu_greedy)**2 + (std_bfs / mu_bfs)**2)
            else:
                err = 0.0
                
            I_matrix[r, c] = I * 100
            err_matrix[r, c] = err * 100

    # Generate Heatmap
    fig, ax = plt.subplots(figsize=(14, max(6, 0.5 * len(heuristics) + 2)))
    
    # Mask NaN values
    masked_I = np.ma.masked_invalid(I_matrix)
    
    # Calculate limits for colormap
    finite = I_matrix[np.isfinite(I_matrix)]
    limit = max(abs(float(finite.min())), abs(float(finite.max())), 1) if finite.size else 1
    
    image = ax.imshow(masked_I, aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
    
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"Nivel {l}" for l in levels])
    ax.set_yticks(range(len(heuristics)))
    
    # Some heuristic names have 'unsolved' suffix which Canonical heuristic removes
    ax.set_yticklabels([heuristic_label(h.replace('_unsolved', '')) for h in heuristics])
    
    ax.set_title("Mejora porcentual de tiempo de Greedy respecto de BFS")
    
    for row in range(I_matrix.shape[0]):
        for col in range(I_matrix.shape[1]):
            val = I_matrix[row, col]
            err = err_matrix[row, col]
            if np.isfinite(val):
                ax.text(col, row, f"{val:.1f}%\n±{err:.1f}%", ha="center", va="center", color="black", fontsize=9)
                
    fig.colorbar(image, ax=ax, label="Mejora respecto de BFS (%)")
    
    plt.tight_layout()
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures', 'benchmark_1_4'))
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, 'greedy_improvement_time.png')
    plt.savefig(out_path, format='png', dpi=300)
    print(f"\nGráfico guardado en {out_path}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
