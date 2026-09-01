# Sokoban Solver

This project implements an automated solver for the classic puzzle game **Sokoban**. It allows users to load custom levels and solve them using various search algorithms, including BFS, DFS, IDDFS, Greedy, and A*. The solver incorporates advanced optimization techniques such as dead-square pruning and several heuristic functions (like Manhattan distance and Hungarian algorithm-based metrics) to efficiently find solutions.

**Team Members (Grupo 4):**
- Ignacio Searles
- Ivo Vilamowski
- Agustin Galan
- Nicolas Koron
- Toribio Viton Sconza

## Requirements
- Python 3.9+ installed and on your PATH.

## Running the Solver

**Windows:** double-click `run.bat` (or run it from a terminal).

**macOS / Linux:**
```bash
chmod +x run.sh   # only needed once
./run.sh
```

These scripts simply launch `main.py`. By default, you'll see the help text explaining the available parameters.

## Optional Flags
You can pass the script's own flags through the launcher, e.g.:
```bash
./run.sh --level levels/level-1.txt --algorithm bfs --animate
```
```bash
run.bat --level levels/level-1.txt --algorithm bfs --animate
```

## Troubleshooting
- **"python is not recognized" (Windows):** Python isn't on PATH — reinstall and check the "Add to PATH" box, or use the Microsoft Store Python.
- **Permission denied running `run.sh`:** run `chmod +x run.sh` once.