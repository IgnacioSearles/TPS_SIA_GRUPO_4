"""Checkpoint por época y exportación de mediciones parciales."""
from dataclasses import asdict

import pandas as pd

from src.tp4.checkpoint import load_checkpoint, save_checkpoint
from src.tp4.plotting import plot_learning_curves
from src.tp4.storage import atomic_file


def epoch_callback(directory, control):
    def completed(checkpoint):
        save_checkpoint(directory / 'checkpoint.npz', checkpoint)
        if control is not None:
            control.check()
    return completed


def export_partial_history(directory):
    checkpoint_path = directory / 'checkpoint.npz'
    if not checkpoint_path.exists():
        return
    checkpoint = load_checkpoint(checkpoint_path)
    with atomic_file(directory / 'history.csv') as stream:
        pd.DataFrame([asdict(epoch) for epoch in checkpoint.history]).to_csv(stream, index=False)
    plot_learning_curves(directory / 'learning_curves.png', checkpoint.history)
