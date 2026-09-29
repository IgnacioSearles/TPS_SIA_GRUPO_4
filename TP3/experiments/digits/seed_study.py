"""Paired five-seed comparison of finalist optimizers, keeping the data split fixed."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import config_from_cli, deep_merge
from experiments.digits.baseline import DEFAULTS, run

STUDY_DEFAULTS = deep_merge(DEFAULTS, {
    "model": {"layers": [784, 64, 32, 10]},
    "training": {"epochs": 500, "shuffle_seed": 42},
    "study": {"seeds": [42, 7, 21, 84, 123], "reuse_directory": "results/digits_optimizers",
              "optimizers": [{"name": "sgd", "lr": 0.1},
                             {"name": "momentum", "lr": 0.1, "momentum": 0.5},
                             {"name": "momentum", "lr": 0.1, "momentum": 0.9}]},
    "output": {"directory": "results/digits_seeds"},
})


def configurations(config):
    seeds = config["study"]["seeds"]
    optimizers = config["study"]["optimizers"]
    if len(seeds) < 2 or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Need at least two distinct non-negative integer seeds")
    if not optimizers or len({json.dumps(spec, sort_keys=True) for spec in optimizers}) != len(optimizers):
        raise ValueError("Need distinct optimizer configurations")
    runs = []
    for seed in seeds:
        for index, optimizer in enumerate(optimizers):
            current = copy.deepcopy(config)
            current.pop("study")
            current["seed"] = seed
            current["training"]["shuffle_seed"] = seed
            current["optimizer"] = copy.deepcopy(optimizer)
            current["output"]["directory"] = str(Path(config["output"]["directory"]) / f"seed_{seed}_optimizer_{index}")
            runs.append(current)
    return runs


def comparable(config):
    result = copy.deepcopy(config)
    result.pop("output", None)
    return result


def aggregate(rows):
    frame = pd.DataFrame(rows)
    metrics = ["validation_accuracy", "validation_loss", "validation_macro_f1", "validation_recall_5"]
    results = []
    for candidate, group in frame.groupby("candidate", sort=False):
        record = {"candidate": candidate, "n_seeds": len(group),
                  "optimizer": group.iloc[0]["optimizer"], "momentum": float(group.iloc[0]["momentum"]),
                  "learning_rate": float(group.iloc[0]["learning_rate"]),
                  "median_best_epoch": float(group.best_epoch.median())}
        for metric in metrics:
            record[metric + "_mean"] = float(group[metric].mean())
            record[metric + "_std"] = float(group[metric].std(ddof=1))
            record[metric + "_min"] = float(group[metric].min())
            record[metric + "_max"] = float(group[metric].max())
        results.append(record)
    return results


def reuse(current, sources, dataset_hash):
    """Only reuse complete, exactly matching experiments on the current CSV."""
    target = Path(current["output"]["directory"])
    required = ["summary.json", "history.csv", "best_weights.npz", "split_indices.npz", "validation_predictions.csv"]
    for source in sources:
        if not all((source / name).is_file() for name in required):
            continue
        saved = json.loads((source / "config.json").read_text())
        summary = json.loads((source / "summary.json").read_text())
        if comparable(saved) != comparable(current) or summary["dataset_sha256"] != dataset_hash:
            continue
        shutil.copytree(source, target)
        (target / "config.json").write_text(json.dumps(current, indent=2) + "\n")
        print(f"Reused verified matching configuration: {source} -> {target}", flush=True)
        return summary, str(source.resolve())
    return None, None


def run_study(config):
    import hashlib
    runs = configurations(config)
    if Path(config["data"]["path"]).name != "digits.csv":
        raise ValueError("Only digits.csv may be used for development")
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "study_config.json").write_text(json.dumps(config, indent=2) + "\n")
    sources = []
    if config["study"].get("reuse_directory"):
        sources = sorted(p.parent for p in Path(config["study"]["reuse_directory"]).glob("*/config.json"))
    digest = hashlib.sha256(Path(config["data"]["path"]).read_bytes()).hexdigest()
    rows, initial_by_seed = [], {}
    reference_split = None
    for current in runs:
        directory = Path(current["output"]["directory"])
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite an earlier run: {directory}")
        summary, source = reuse(current, sources, digest)
        if summary is None:
            summary = run(current)
        assert summary["dataset_sha256"] == digest
        with np.load(directory / "split_indices.npz") as split:
            indices = (split["train"].copy(), split["validation"].copy())
        if reference_split is None:
            reference_split = indices
        else:
            for actual, expected in zip(indices, reference_split):
                np.testing.assert_array_equal(actual, expected)
        seed = current["seed"]
        if seed in initial_by_seed:
            assert initial_by_seed[seed] == summary["initial_validation"], "Paired initialization differs"
        else:
            initial_by_seed[seed] = summary["initial_validation"]
        spec = current["optimizer"]
        candidate = f"{spec['name']}_lr{spec['lr']:g}_m{spec.get('momentum', 0):g}"
        metric = summary["validation"]
        history = pd.read_csv(directory / "history.csv")
        reached = history.loc[history.validation_accuracy >= 0.95, "epoch"]
        rows.append({"candidate": candidate, "seed": seed, "optimizer": spec["name"],
                     "learning_rate": spec["lr"], "momentum": spec.get("momentum", 0),
                     "best_epoch": summary["best_epoch"], "train_accuracy": summary["train"]["accuracy"],
                     "validation_accuracy": metric["accuracy"], "validation_loss": metric["loss"],
                     "validation_macro_f1": metric["macro_f1"], "validation_recall_5": metric["per_class"][5]["recall"],
                     "first_epoch_at_95_percent": int(reached.iloc[0]) if len(reached) else None,
                     "run_directory": directory.name, "reused_from": source})
        pd.DataFrame(rows).to_csv(output / "runs.csv", index=False)
    results = aggregate(rows)
    pd.DataFrame(results).to_csv(output / "aggregate.csv", index=False)
    winner = max(results, key=lambda r: (r["validation_accuracy_mean"], -r["validation_loss_mean"]))
    result = {"selection": "highest mean validation accuracy across seeds, tie: lowest mean validation loss",
              "winner": winner, "aggregates": results, "runs": rows,
              "sample_std_ddof": 1, "fixed_split_verified": True, "paired_initial_metrics_verified": True,
              "dataset_sha256": digest, "external_test_evaluated": False,
              "scope": "Training randomness on one reused development split; not independent test performance"}
    (output / "study_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    frame = pd.DataFrame(rows)
    for ax, metric, title in zip(axes, ["validation_accuracy", "validation_recall_5"],
                               ["Accuracy de validación", "Recall del 5 en validación"]):
        for seed in config["study"]["seeds"]:
            group = frame[frame.seed == seed].set_index("candidate")
            ax.plot(range(len(results)), [100 * group.loc[r["candidate"], metric] for r in results],
                    marker="o", label=f"Semilla {seed}", alpha=0.8)
        ax.set(xticks=range(len(results)), xticklabels=["SGD" if r["optimizer"] == "sgd" else f"Momentum {r['momentum']:g}" for r in results],
               title=title, ylabel="%")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "seed_comparison.png", dpi=150)
    plt.close(fig)
    print(pd.DataFrame(results).to_string(index=False), flush=True)
    return result


if __name__ == "__main__":
    run_study(config_from_cli(STUDY_DEFAULTS))
