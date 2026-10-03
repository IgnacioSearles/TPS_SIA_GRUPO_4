"""Run a list of local experiment configs in order and resume between jobs."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_queue(config_path: str | Path) -> None:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    queue_copy = output / "queue_config.json"
    state_path = output / "queue_state.json"
    if queue_copy.exists() and queue_copy.read_text() != json.dumps(config, indent=2) + "\n":
        raise ValueError(f"Queue config differs from existing {queue_copy}; use a new output directory")
    queue_copy.write_text(json.dumps(config, indent=2) + "\n")
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["queue_sha256"] != _digest(queue_copy):
            raise ValueError("Queue definition changed; use a new queue output directory")
        current_job_hashes = {job["name"]: _digest(Path(job["config"])) for job in config["jobs"]}
        if state.get("job_config_sha256") != current_job_hashes:
            raise ValueError("A job config changed; use a new queue output directory")
    else:
        state = {"queue_sha256": _digest(queue_copy),
                 "job_config_sha256": {job["name"]: _digest(Path(job["config"])) for job in config["jobs"]},
                 "completed": [], "active": None}
    for job in config["jobs"]:
        if job["name"] in state["completed"]:
            continue
        state["active"] = job["name"]
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        command = [sys.executable, "-m", job["module"], job["config"]]
        print(f"\n=== Iniciando {job['name']} ===", flush=True)
        subprocess.run(command, check=True)
        state["completed"].append(job["name"])
        state["active"] = None
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        print(f"=== Terminado {job['name']} ===", flush=True)
    print("Cola completa.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="experiments/digits/configs/queue.json")
    run_queue(parser.parse_args().config)
