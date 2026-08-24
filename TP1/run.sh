#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR" || {
        echo ""
        echo "Could not create a virtual environment. Make sure Python 3 is installed."
        exit 1
    }
fi

source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r "$DIR/requirements-notebook.txt"

echo ""
python "$DIR/interactive_main.py" "$@"