"""Particiones independientes y protocolo fijo para capacidad del VAE RGB."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.factory import build_dataset
from src.data.art import FAMILIES
from src.options import ConfigurationError, Options

BASE = "ej2_art_1_single"
CONFIGS = {4: BASE, 8: "ej2_capacity_art_latent8", 16: "ej2_capacity_art_latent16"}
SEEDS = (7, 17, 27)
PROBE_SEEDS = {"validation": 2026111401, "reserved_final": 2026111402}


def read_json(path):
    if not path.is_file():
        raise ConfigurationError(f"Falta {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_hash(samples):
    return hashlib.sha256(samples.astype("<f4").tobytes()).hexdigest()


def expected_configs(root):
    base = read_json(root / f"configs/{BASE}.json")
    if (base["dataset"]["type"] != "geometric_art" or base["dataset"].get("objects", 1) != 1
            or set(base["dataset"].get("families", FAMILIES)) != set(FAMILIES)
            or base["model"]["type"] != "variational" or base["model"]["latent_size"] != 4
            or base["seeds"] != list(SEEDS) or "output" in base or "name" in base):
        raise ConfigurationError("La comparación requiere la base VAE RGB de un objeto y semillas 7,17,27")
    configs = {}
    for latent, name in CONFIGS.items():
        config = copy.deepcopy(base)
        config["model"]["latent_size"] = latent
        configs[latent] = config
        path = root / f"configs/{name}.json"
        if path.is_file() and read_json(path) != config:
            raise ConfigurationError(f"Cambió un factor adicional al latente en {path}")
    return configs


def prepare_capacity(root, count=512):
    root = Path(root)
    if type(count) is not int or count < 8 or count % 4:
        raise ConfigurationError("La cantidad reservada debe ser múltiplo de 4 y al menos 8")
    configs = expected_configs(root)
    for latent, config in configs.items():
        path = root / f"configs/{CONFIGS[latent]}.json"
        if not path.exists():
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    directory = root / "results/art_capacity_data"
    protocol = {"version": 1, "latents": list(CONFIGS), "model_seeds": list(SEEDS),
                "probe_count": count, "probe_seeds": PROBE_SEEDS,
                "configuration_hashes": {str(k): file_hash(root / f"configs/{v}.json") for k, v in CONFIGS.items()},
                "selection": "smallest latent within 0.01 absolute validation macro IoU of best mean over seeds",
                "iou_tolerance": .01, "mask_contrast_fraction": .5,
                "boundary_tolerance_px": 1., "empty_contour_penalty": "image diagonal",
                "selection_scope": "reconstruction geometry, not certification of generative quality"}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["protocol"] != protocol:
            raise ConfigurationError("El protocolo de capacidad ya está fijado; no cambiarlo dentro del mismo estudio")
        for split, item in manifest["partitions"].items():
            if not (directory / f"{split}.npz").is_file() or file_hash(directory / f"{split}.npz") != item["file_sha256"]:
                raise ConfigurationError(f"Partición alterada: {split}")
        return directory
    dataset = build_dataset(Options(configs[4]["dataset"], "dataset"), root)
    seen = {row.tobytes() for row in np.vstack([dataset.samples, dataset.generative_study.test_samples])}
    directory.mkdir(parents=True, exist_ok=True)
    partitions = {}
    for split, seed in PROBE_SEEDS.items():
        options = {**configs[4]["dataset"], "data_seed": seed, "train_count": count, "test_count": 4}
        # El loader genera datos; aquí ninguna fila se entrega a un entrenador.
        probe = build_dataset(Options(options, "probe"), root)
        samples = probe.samples
        fingerprints = {row.tobytes() for row in samples}
        if len(fingerprints) != count or seen.intersection(fingerprints):
            raise ConfigurationError("Las particiones de capacidad comparten imágenes exactas")
        seen.update(fingerprints)
        records = [{**r, "label": f"{split}_{i:05d}"}
                   for i, r in enumerate(probe.generative_study.train_records)]
        path = directory / f"{split}.npz"
        np.savez_compressed(path, samples=samples, records_json=json.dumps(records))
        partitions[split] = {"count": count, "seed": seed, "samples_sha256": sample_hash(samples),
                             "file_sha256": file_hash(path)}
    manifest = {"protocol": protocol, "partitions": partitions,
                "training_sha256": dataset.generative_study.metadata["train_sha256"],
                "previous_test_sha256": dataset.generative_study.metadata["test_sha256"],
                "interpretation": "Independent same-distribution probes. Original test was already inspected. "
                                  "No final reconstruction is evaluated before validation selection is frozen."}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Protocolo y particiones fijados en {directory}", flush=True)
    return directory


def load_probe(directory, split):
    manifest = read_json(directory / "manifest.json")
    path = directory / f"{split}.npz"
    if file_hash(path) != manifest["partitions"][split]["file_sha256"]:
        raise ConfigurationError(f"Partición alterada: {split}")
    with np.load(path, allow_pickle=False) as saved:
        return saved["samples"], json.loads(str(saved["records_json"]))
