import sys
import os
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress specific matplotlib warnings that can occur with log scale and negative error bars
warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory to path so we can import from main project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_level
from algorithms import ALGORITHMS
from engine import run_search

def worker(algo_name, level_path, result_queue):
    initial_state, level = load_level(level_path)
    algorithm = ALGORITHMS[algo_name]()
    res = run_search(algorithm, initial_state, level)
    result_queue.put(res)

def run_with_timeout(algo_name, level_path, timeout=45):
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(target=worker, args=(algo_name, level_path, queue))
    
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return None
        
    try:
        res = queue.get_nowait()
        return res
    except Exception:
        return None

def main():
    level = 1
    level_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'levels', f'level-{level}.txt'))
    algorithms = ["bfs", "dfs", "iddfs"]
    runs = 10
    timeout = 45
    
    times = {a: [] for a in algorithms}
    timeouts = {a: False for a in algorithms}
    
    # Store metrics from the first successful run
    metrics = {
        'expanded_nodes': {a: 0 for a in algorithms},
        'frontier_nodes': {a: 0 for a in algorithms},
        'cost': {a: 0 for a in algorithms}
    }
    
    print(f"--- Nivel {level} ---")
    for a in algorithms:
        print(f"Algoritmo {a.upper()}:", end='', flush=True)
        first_success = False
        for r in range(runs):
            res = run_with_timeout(a, level_path, timeout)
            if res is None:
                timeouts[a] = True
                print(f" TIMEOUT ({timeout}s)", end='', flush=True)
                break
            else:
                if not first_success:
                    metrics['expanded_nodes'][a] = res.expanded_nodes if res.success else 0
                    metrics['frontier_nodes'][a] = res.frontier_nodes if res.success else 0
                    metrics['cost'][a] = res.cost if res.success else 0
                    first_success = True
                times[a].append(res.processing_time_sec * 1000) # Convert to ms
                print(f" {res.processing_time_sec * 1000:.1f}ms", end='', flush=True)
        print()
        
    # Generate 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Comparación de Algoritmos No Informados (Nivel {level})")
    
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c']
    algo_labels = [a.upper() for a in algorithms]
    
    # Plot Processing Time (Log Scale with Error Bars)
    means = []
    stds = []
    for a in algorithms:
        if timeouts[a]:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(np.mean(times[a]))
            stds.append(np.std(times[a]))
            
    axs[0, 0].bar(algo_labels, means, yerr=stds, capsize=5, color=palette)
    axs[0, 0].set_title('Tiempo de Procesamiento (ms)')
    axs[0, 0].set_ylabel('Tiempo (milisegundos) [Escala Logarítmica]')
    axs[0, 0].set_yscale('log')
    
    # Plot Expanded Nodes
    expanded = [metrics['expanded_nodes'][a] for a in algorithms]
    axs[0, 1].bar(algo_labels, expanded, color=palette)
    axs[0, 1].set_title('Nodos Expandidos')
    axs[0, 1].set_ylabel('Cantidad [Escala Logarítmica]')
    axs[0, 1].set_yscale('log')
    
    # Plot Frontier Nodes
    frontier = [metrics['frontier_nodes'][a] for a in algorithms]
    axs[1, 0].bar(algo_labels, frontier, color=palette)
    axs[1, 0].set_title('Nodos Frontera')
    axs[1, 0].set_ylabel('Cantidad')
    
    # Plot Cost
    cost = [metrics['cost'][a] for a in algorithms]
    axs[1, 1].bar(algo_labels, cost, color=palette)
    axs[1, 1].set_title('Costo de la Solución (movimientos)')
    axs[1, 1].set_ylabel('Costo')
    
    plt.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)
    
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, f'uninformed_comparison_level{level}.png')
    plt.savefig(out_path, format='png')
    print(f"\nGráfico guardado en {out_path}")

    print("\n--- Costo de las Soluciones ---")
    for a in algorithms:
        if timeouts[a]:
            print(f"Algoritmo {a.upper()}: TIMEOUT")
        else:
            print(f"Algoritmo {a.upper()}: {metrics['cost'][a]} movimientos")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
