"""Learning-rate schedules: the learning rate to use in each epoch of a run.

A schedule is a declarative spec, e.g. {"name": "cosine", "final_fraction": 0.01}.
`None` means a constant learning rate, the behavior of every study before schedules
existed.
"""

import math
from typing import Any

SCHEDULES = ("constant", "cosine")


def learning_rate_at(schedule: dict[str, Any] | None, base_lr: float, epoch: int, total_epochs: int) -> float:
    """Learning rate for `epoch` (1-based) of a run lasting `total_epochs`.

    Cosine decays smoothly from `base_lr` in the first epoch to
    `final_fraction * base_lr` in the last one.
    """
    if not 1 <= epoch <= total_epochs:
        raise ValueError(f"epoch must be in [1, {total_epochs}], got {epoch}")
    name = "constant" if schedule is None else schedule.get("name")
    if name == "constant":
        return base_lr
    if name == "cosine":
        final_fraction = schedule.get("final_fraction", 0.0)
        if not 0 <= final_fraction <= 1:
            raise ValueError(f"final_fraction must be in [0, 1], got {final_fraction}")
        progress = (epoch - 1) / max(total_epochs - 1, 1)
        fraction = final_fraction + (1 - final_fraction) * 0.5 * (1 + math.cos(math.pi * progress))
        return base_lr * fraction
    raise ValueError(f"Unknown learning-rate schedule {name!r}; available: {SCHEDULES}")
