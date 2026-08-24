# Sokoban Solver — Setup

## Requirements
- Python 3.9+ installed and on your PATH.
  - Windows: download from https://www.python.org/downloads/ and check
    **"Add python.exe to PATH"** during install.
  - macOS: `brew install python3` (or use the python.org installer).
  - Linux: use your distro's package manager, e.g. `sudo apt install python3 python3-venv`.

## Running it

**Windows:** double-click `run.bat` (or run it from a terminal).

**macOS / Linux:**
```bash
chmod +x run.sh   # only needed once
./run.sh
```

Either script will, the first time you run it:
1. Create a local virtual environment in `.venv/` (isolated from any other
   Python packages on your system — no conflicts, no admin rights needed).
2. Install the pinned dependencies from `requirements.txt`.
3. Launch `interactive_main.py`.

Every run after that reuses the same `.venv/`, so it starts in a couple of
seconds and always installs the same dependency versions on every machine.

## Optional flags
You can still pass the script's own flags through the launcher, e.g.:
```bash
./run.sh --levels-dir levels --outdir figures
```
```
run.bat --levels-dir levels --outdir figures
```

## If something goes wrong
- **"python is not recognized" (Windows):** Python isn't on PATH — reinstall
  and check the "Add to PATH" box, or use the Microsoft Store Python.
- **Permission denied running `run.sh`:** run `chmod +x run.sh` once.
- Delete the `.venv/` folder any time to force a clean reinstall.