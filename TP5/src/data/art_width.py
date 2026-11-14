"""Diseño factorial de tipo de autoencoder y ancho, con datos RGB nuevos de evaluación."""

import copy
import json
from pathlib import Path

import numpy as np

from src.data.art_capacity import read_json, file_hash, sample_hash, load_probe
from src.data.factory import build_dataset
from src.options import ConfigurationError, Options

BASE = "ej2_capacity_art_latent8"
SEEDS = (7, 17, 27)
VARIANTS = {
    "vae_small": (BASE, "VAE [128,32]"),
    "ae_small": ("ej2_width_art_ae_small", "AE [128,32]"),
    "vae_wide": ("ej2_width_art_vae_wide", "VAE [256,128]"),
    "ae_wide": ("ej2_width_art_ae_wide", "AE [256,128]"),
}
PROBE_SEEDS = {"validation": 2026111501, "reserved_final": 2026111502}
CONTRASTS = {"width_ae": ("ae_wide", "ae_small"), "width_vae": ("vae_wide", "vae_small"),
             "ae_minus_vae_small": ("ae_small", "vae_small"),
             "ae_minus_vae_wide": ("ae_wide", "vae_wide")}


def configurations(root, wide_layers=(256, 128)):
    base = read_json(root / f"configs/{BASE}.json")
    if (base["model"]["type"] != "variational" or base["model"]["latent_size"] != 8
            or base["dataset"]["type"] != "geometric_art" or base["dataset"].get("objects", 1) != 1
            or base["seeds"] != list(SEEDS) or "output" in base or "name" in base):
        raise ConfigurationError("Se necesita la base VAE RGB de latente 8 y semillas 7,17,27")
    result = {}
    for variant in VARIANTS:
        config = copy.deepcopy(base)
        if variant.endswith("wide"):
            config["model"]["hidden_layers"] = list(wide_layers)
        if variant.startswith("ae_"):
            config["model"]["type"] = "autoencoder"
            config["model"].pop("kl")
        result[variant] = config
        path = root / f"configs/{VARIANTS[variant][0]}.json"
        if path.exists() and read_json(path) != config:
            raise ConfigurationError(f"La variante cambió factores ajenos al protocolo: {path}")
    return result


def prepare_width(root, count=512, wide_layers=(256, 128)):
    root = Path(root)
    if type(count) is not int or count < 8 or count % 4:
        raise ConfigurationError("La cantidad reservada debe ser múltiplo de 4, al menos 8")
    configs = configurations(root, wide_layers)
    for variant, config in configs.items():
        path = root / f"configs/{VARIANTS[variant][0]}.json"
        if not path.exists():
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    directory = root / "results/art_width_data"
    protocol = {"version": 1, "probe_count": count, "probe_seeds": PROBE_SEEDS,
                "model_seeds": list(SEEDS), "wide_layers": list(wide_layers),
                "configuration_hashes": {v: file_hash(root / f"configs/{item[0]}.json") for v, item in VARIANTS.items()},
                "contrasts": CONTRASTS, "iou_tolerance": .01,
                "selection": "within each model type, choose small if within 0.01 of best validation macro IoU",
                "final_scope": "all four predeclared conditions and contrasts, no retuning",
                "generation": {"seed": 1234, "count": 64, "ae": "Gaussian fitted to training codes only",
                               "vae": "standard normal prior", "ae_standard": "supplementary diagnostic"}}
    path = directory / "manifest.json"
    if path.exists():
        manifest = read_json(path)
        # JSON convierte los pares de los contrastes de tuplas a listas.
        if manifest["protocol"] != json.loads(json.dumps(protocol)):
            raise ConfigurationError("El protocolo de ancho ya está fijado")
        for split, item in manifest["partitions"].items():
            if file_hash(directory / f"{split}.npz") != item["file_sha256"]:
                raise ConfigurationError(f"Partición alterada: {split}")
        return directory
    training = build_dataset(Options(configs["vae_small"]["dataset"], "dataset"), root)
    excluded = [training.samples, training.generative_study.test_samples]
    prior = root / "results/art_capacity_data"
    if (prior / "manifest.json").exists():
        excluded.extend(load_probe(prior, split)[0] for split in PROBE_SEEDS)
    seen = {row.tobytes() for samples in excluded for row in samples}
    directory.mkdir(parents=True, exist_ok=True)
    partitions = {}
    for split, seed in PROBE_SEEDS.items():
        probe = build_dataset(Options({**configs["vae_small"]["dataset"], "train_count": count,
                                      "test_count": 4, "data_seed": seed}, "probe"), root)
        samples = probe.samples
        fingerprints = {row.tobytes() for row in samples}
        if len(fingerprints) != count or seen.intersection(fingerprints):
            raise ConfigurationError("Las referencias nuevas comparten imágenes con particiones anteriores")
        seen.update(fingerprints)
        records = [{**r, "label": f"width_{split}_{i:05d}"}
                   for i, r in enumerate(probe.generative_study.train_records)]
        output = directory / f"{split}.npz"
        np.savez_compressed(output, samples=samples, records_json=json.dumps(records))
        partitions[split] = {"count": count, "samples_sha256": sample_hash(samples), "file_sha256": file_hash(output)}
    manifest = {"protocol": protocol, "partitions": partitions,
                "training_sha256": training.generative_study.metadata["train_sha256"],
                "previous_test_sha256": training.generative_study.metadata["test_sha256"],
                "previous_capacity_manifest_sha256": file_hash(prior / "manifest.json") if (prior / "manifest.json").exists() else None,
                "interpretation": "Parameters are evaluation metadata, not training inputs. "
                                  "AE/VAE comparison changes sampling, variance head and KL together."}
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Diseño y particiones de ancho fijados: {directory}", flush=True)
    return directory
