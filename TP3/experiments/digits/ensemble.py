"""Ensembles of already-trained exercise 3 networks, evaluated on validation only.

Every candidate network is a 784-128-128-10 trained on the same merged data and the
same stratified validation split. An ensemble averages the networks' class
probabilities: softmax outputs as they are, sigmoid outputs normalized to sum 1 per
image. Nothing is retrained. The ensembles are declared in the config before
looking at their results, so picking the best of them on validation stays honest.

Safety checks, failing fast: all networks must share the validation split, and each
one must reproduce the validation accuracy recorded when it was trained. Networks
with identical weights (exact reproductions) are counted once.

The winner is the best ensemble on validation (fewer networks on a tie). Only with
`--evaluate-test` is the test file opened, and only for that winner.

Run from TP3:
    python -m experiments.digits.ensemble [config.json] [--evaluate-test]
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

import nn  # noqa: F401 - register model components
from datasets.digit_dataset_loader import load_digit_arrays
from experiments.config import Config, deep_merge, read_json
from experiments.digits.data import merge_digit_files, split_train_validation
from nn.network import Sequential, build_model, load_weights

DEFAULTS: Config = {
    "ensembles": {
        "3 sigmoides (modelo final)": ["reports/digits_e3_final/entrenamiento/architecture_00/seed_*"],
        "5 softmax": ["reports/digits_e3_softmax_tuned/runs/row_01/seed_*"],
        "todas las sigmoides sin L2": ["reports/digits_e3_final/entrenamiento/architecture_00/seed_*",
                                       "reports/digits_e3_softmax_tuned/runs/row_00/seed_*"],
        "sigmoides + softmax": ["reports/digits_e3_final/entrenamiento/architecture_00/seed_*",
                                "reports/digits_e3_softmax_tuned/runs/row_00/seed_*",
                                "reports/digits_e3_softmax_tuned/runs/row_01/seed_*"],
        "todo, con L2": ["reports/digits_e3_final/entrenamiento/architecture_00/seed_*",
                         "reports/digits_e3_softmax_tuned/runs/row_00/seed_*",
                         "reports/digits_e3_softmax_tuned/runs/row_01/seed_*",
                         "reports/digits_e3_l2/wd_*/architecture_00/seed_*"],
        "5 softmax, η coseno": ["reports/digits_e3_softmax_cosine/runs/row_01/seed_*"],
        "10 softmax, η constante + coseno": ["reports/digits_e3_softmax_tuned/runs/row_01/seed_*",
                                             "reports/digits_e3_softmax_cosine/runs/row_01/seed_*"],
    },
    "test_file": "datasets/digits_test.csv",
    # Recorded and recomputed validation accuracy must agree to this tolerance.
    "reproduction_tolerance": 1e-9,
    "output_directory": "reports/digits_e3_ensemble",
}


@dataclass(frozen=True)
class Member:
    run_directory: Path
    model_spec: dict
    split: tuple
    recorded_accuracy: float
    weights_digest: str


def read_member(run_directory: Path) -> Member:
    """Model spec, data split and recorded accuracy of one saved run, whichever study wrote it."""
    protocol = json.loads((run_directory / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    if "training_config" in protocol:
        data_config, model_spec = protocol["training_config"], protocol["training_config"]["model"]
    elif "architecture" in protocol:
        data_config = protocol["config"]
        model_spec = {**data_config["model"], "layers": protocol["architecture"]}
    else:
        raise ValueError(f"{run_directory}: unknown protocol format")
    split = (tuple(data_config["files"]), data_config["validation_ratio"], data_config["split_seed"])
    weights = (run_directory / "best_weights.npz").read_bytes()
    return Member(run_directory, model_spec, split, summary["best"]["accuracy"], hashlib.sha256(weights).hexdigest())


def resolve_members(patterns: list[str]) -> list[Member]:
    """All runs matching the patterns, without duplicates of the same weights."""
    directories = sorted({Path(path) for pattern in patterns for path in glob(pattern)})
    if not directories:
        raise FileNotFoundError(f"No runs match {patterns}")
    members: dict[str, Member] = {}
    for directory in directories:
        member = read_member(directory)
        members.setdefault(member.weights_digest, member)
    return list(members.values())


def class_probabilities(net: Sequential, X: np.ndarray) -> np.ndarray:
    """Network outputs as a distribution over digits (sigmoid outputs are renormalized)."""
    outputs = net.forward(X)
    return outputs / outputs.sum(axis=1, keepdims=True)


def ensemble_accuracy(probabilities: list[np.ndarray], y: np.ndarray) -> float:
    return float((np.mean(probabilities, axis=0).argmax(axis=1) == y).mean())


def load_member(member: Member) -> Sequential:
    net = build_model(member.model_spec, np.random.default_rng(0))
    return load_weights(net, member.run_directory / "best_weights.npz")


def select_winner(ensembles: pd.DataFrame) -> str:
    """Best validation accuracy; on a tie, the ensemble with fewer networks."""
    ranked = ensembles.sort_values(["validation_accuracy", "networks"], ascending=[False, True])
    return str(ranked.iloc[0]["ensemble"])


def evaluate_test(members: list[Member], test_file: str) -> dict:
    """The only use of the test file: the winning ensemble and, for reference, its networks alone."""
    X, y = load_digit_arrays(test_file)
    probabilities = [class_probabilities(load_member(member), X) for member in members]
    single = [float((p.argmax(axis=1) == y).mean()) for p in probabilities]
    accuracy = ensemble_accuracy(probabilities, y)
    return {"test_samples": int(len(y)), "test_accuracy": accuracy,
            "test_errors": int(round((1 - accuracy) * len(y))),
            "single_network_test_mean": float(np.mean(single)), "single_network_test_std": float(np.std(single)),
            "networks": [member.run_directory.as_posix() for member in members]}


def evaluate(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[Member]]]:
    members_by_ensemble = {name: resolve_members(patterns) for name, patterns in config["ensembles"].items()}
    all_members = {member.weights_digest: member for members in members_by_ensemble.values() for member in members}
    splits = {member.split for member in all_members.values()}
    if len(splits) != 1:
        raise ValueError(f"Networks use different validation splits: {splits}")
    files, validation_ratio, split_seed = splits.pop()
    data = merge_digit_files(list(files))
    _, validation_rows = split_train_validation(data.y, validation_ratio, split_seed)
    X, y = data.X[validation_rows], data.y[validation_rows]

    probabilities, model_rows = {}, []
    for digest, member in all_members.items():
        probabilities[digest] = class_probabilities(load_member(member), X)
        accuracy = float((probabilities[digest].argmax(axis=1) == y).mean())
        if abs(accuracy - member.recorded_accuracy) > config["reproduction_tolerance"]:
            raise ValueError(f"{member.run_directory}: validation accuracy {accuracy} does not reproduce "
                             f"the recorded {member.recorded_accuracy}")
        model_rows.append({"run_directory": member.run_directory.as_posix(),
                           "output_activation": member.model_spec["output_activation"],
                           "validation_accuracy": accuracy})

    ensemble_rows = []
    for name, members in members_by_ensemble.items():
        single = [float((probabilities[m.weights_digest].argmax(axis=1) == y).mean()) for m in members]
        accuracy = ensemble_accuracy([probabilities[m.weights_digest] for m in members], y)
        ensemble_rows.append({"ensemble": name, "networks": len(members), "validation_accuracy": accuracy,
                              "validation_errors": int(round((1 - accuracy) * len(y))),
                              "single_network_mean": float(np.mean(single)),
                              "single_network_best": float(np.max(single))})
    return pd.DataFrame(model_rows), pd.DataFrame(ensemble_rows), members_by_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation accuracy of ensembles of trained E3 networks")
    parser.add_argument("config", nargs="?", help="optional JSON config; omitted keys use defaults")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="score only the ensemble chosen on validation on the test file, once")
    args = parser.parse_args()
    # Ensemble names contain "η"; the Windows console's default code page cannot print it.
    sys.stdout.reconfigure(encoding="utf-8")
    config = deep_merge(DEFAULTS, read_json(args.config)) if args.config else DEFAULTS
    models, ensembles, members_by_ensemble = evaluate(config)
    winner = select_winner(ensembles)
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    models.to_csv(output / "models.csv", index=False)
    ensembles.to_csv(output / "ensembles.csv", index=False)
    summary = {"winner_on_validation": winner}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.evaluate_test:
        # Its own file, so a later validation-only run never erases the test result.
        test = {"ensemble": winner, **evaluate_test(members_by_ensemble[winner], config["test_file"])}
        (output / "test.json").write_text(json.dumps(test, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        print(models.to_string(index=False))
        print()
        print(ensembles.to_string(index=False))
    print(f"\nElegido en validación: {winner}")
    if args.evaluate_test:
        print(json.dumps(test, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
