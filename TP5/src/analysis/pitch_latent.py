"""Comparación de 2, 4 y 16 dimensiones latentes.

python -m src.analysis.pitch_latent --prepare-final
python -m src.analysis.pitch_latent --final
"""

import argparse
from pathlib import Path

from src.analysis.pitch_comparison import (
    LATENT, VALIDATION_TARGETS, FINAL_TARGETS, SEEDS, VARIANTS,
    compare_study, prepare_final_study,
)
from src.options import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def compare_latents(root, phase="validation"):
    return compare_study(root, LATENT, phase)


def prepare_final(root, report):
    return prepare_final_study(root, report, LATENT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-final", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.prepare_final and args.final:
        parser.error("--prepare-final y --final son excluyentes")
    try:
        report = compare_latents(PROJECT_ROOT, "final" if args.final else "validation")
        if args.prepare_final:
            prepare_final(PROJECT_ROOT, report)
    except (ConfigurationError, KeyError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
