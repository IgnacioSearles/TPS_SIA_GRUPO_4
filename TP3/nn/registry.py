"""Name -> class lookup so components can be built from JSON configs."""

from typing import Any, Callable

REGISTRIES: dict[str, dict[str, type]] = {
    "activation": {},
    "loss": {},
    "optimizer": {},
    "initializer": {},
    "scheduler": {},
    "callback": {},
}

Spec = str | dict[str, Any]


def register(kind: str, name: str) -> Callable[[type], type]:
    """Class decorator that makes `cls` buildable as `build(kind, name)`."""
    if kind not in REGISTRIES:
        raise ValueError(f"Unknown component kind '{kind}'. Valid kinds: {list(REGISTRIES)}")

    def decorator(cls: type) -> type:
        if name in REGISTRIES[kind]:
            raise ValueError(f"{kind} '{name}' is already registered")
        REGISTRIES[kind][name] = cls
        return cls

    return decorator


def build(kind: str, spec: Spec, **dependencies: Any) -> Any:
    """Instantiate a registered component.

    `spec` is either a bare name ("step") or a dict with a "name" key plus
    constructor arguments ({"name": "sgd", "lr": 0.1}). `dependencies` are
    runtime objects that do not belong in a config file, such as the shared rng.
    """
    spec = {"name": spec} if isinstance(spec, str) else dict(spec)
    name = spec.pop("name", None)
    if name is None:
        raise ValueError(f"{kind} spec is missing a 'name': {spec}")
    if name not in REGISTRIES[kind]:
        raise ValueError(f"Unknown {kind} '{name}'. Available: {sorted(REGISTRIES[kind])}")
    return REGISTRIES[kind][name](**spec, **dependencies)
