"""Comparación de la pérdida auxiliar de interpolación: lambda 0, 0.1 y 1.

python -m src.analysis.pitch_interploss --prepare-final
python -m src.analysis.pitch_interploss --final
"""

import argparse
from pathlib import Path

from src.analysis.pitch_comparison import (
    INTERPOLATION, VALIDATION_TARGETS, FINAL_TARGETS, SEEDS, VARIANTS,
    compare_study, prepare_final_study,
)
from src.options import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def compare_interpolation_weights(root, phase="validation"):
    return compare_study(root, INTERPOLATION, phase)


def prepare_final(root, report):
    return prepare_final_study(root, report, INTERPOLATION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-final", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.prepare_final and args.final:
        parser.error("--prepare-final y --final son excluyentes")
    try:
        report = compare_interpolation_weights(PROJECT_ROOT, "final" if args.final else "validation")
        if args.prepare_final:
            prepare_final(PROJECT_ROOT, report)
    except (ConfigurationError, KeyError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
