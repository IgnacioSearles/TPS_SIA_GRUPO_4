import sys
import os
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import from the main project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_level
from algorithms import ALGORITHMS, HEURISTICS
from engine import run_search

def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python 01_compare_algorithms.py "
            "<level_path> [heuristic] [pruning_mode]"
        )
        sys.exit(1)

    level_path = sys.argv[1]
    heuristic_name = sys.argv[2].lower() if len(sys.argv) > 2 else "hungarian"
    pruning_mode = sys.argv[3].lower() if len(sys.argv) > 3 else "dead_squares"

    if heuristic_name not in HEURISTICS:
        valid_heuristics = ", ".join(sorted(HEURISTICS))
        raise ValueError(
            f"Unknown heuristic '{heuristic_name}'. Valid options: {valid_heuristics}"
        )
    if pruning_mode not in {"dead_squares", "local"}:
        raise ValueError(
            f"Unknown pruning mode '{pruning_mode}'. Valid options: dead_squares, local"
        )
    
    metrics = {
        'algorithm': [],
        'processing_time_ms': [],
        'expanded_nodes': [],
        'frontier_nodes': [],
        'cost': []
    }
    
    print(f"Running experiment on level: {level_path}")
    
    for algo_name, AlgoClass in ALGORITHMS.items():
        print(f"Running {algo_name.upper()}...")
        initial_state, level = load_level(level_path)
        
        # Instantiate the algorithm
        if algo_name in {"astar", "greedy"}:
            algorithm = AlgoClass(
                heuristic=HEURISTICS[heuristic_name],
                pruning_mode=pruning_mode,
            )
        else:
            algorithm = AlgoClass(pruning_mode=pruning_mode)
        result = run_search(algorithm, initial_state, level)
        
        metrics['algorithm'].append(algo_name.upper())
        if result.success:
            metrics['processing_time_ms'].append(result.processing_time_sec * 1000)
            metrics['expanded_nodes'].append(result.expanded_nodes)
            metrics['frontier_nodes'].append(result.frontier_nodes)
            metrics['cost'].append(result.cost)
        else:
            # If no solution found, just put 0 to indicate failure
            metrics['processing_time_ms'].append(result.processing_time_sec * 1000)
            metrics['expanded_nodes'].append(result.expanded_nodes)
            metrics['frontier_nodes'].append(result.frontier_nodes)
            metrics['cost'].append(0)

    # Ensure figures directory exists
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))
    os.makedirs(figures_dir, exist_ok=True)
    
    # Generate bar charts
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Comparación de Algoritmos en {os.path.basename(level_path)}")
    
    # Generate distinct colors for each algorithm
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = [palette[i % len(palette)] for i in range(len(metrics['algorithm']))]
    
    # Plot Processing Time
    axs[0, 0].bar(metrics['algorithm'], metrics['processing_time_ms'], color=colors)
    axs[0, 0].set_title('Tiempo de Procesamiento (ms)')
    axs[0, 0].set_ylabel('Tiempo (milisegundos)')
    
    # Plot Expanded Nodes
    axs[0, 1].bar(metrics['algorithm'], metrics['expanded_nodes'], color=colors)
    axs[0, 1].set_title('Nodos Expandidos')
    axs[0, 1].set_ylabel('Cantidad')
    
    # Plot Frontier Nodes
    axs[1, 0].bar(metrics['algorithm'], metrics['frontier_nodes'], color=colors)
    axs[1, 0].set_title('Nodos Frontera')
    axs[1, 0].set_ylabel('Cantidad')
    
    # Plot Cost
    axs[1, 1].bar(metrics['algorithm'], metrics['cost'], color=colors)
    axs[1, 1].set_title('Costo de la Solución (movimientos)')
    axs[1, 1].set_ylabel('Costo')
    
    plt.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)
    
    # Save the figure
    base_name = os.path.splitext(os.path.basename(level_path))[0]
    figure_path = os.path.join(figures_dir, f"comparison_{base_name}.svg")
    plt.savefig(figure_path, format='svg')
    print(f"Figure saved to {figure_path}")

if __name__ == "__main__":
    main()
