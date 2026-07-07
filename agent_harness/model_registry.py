from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path

from .compare import ProviderSpec


@dataclass
class ModelConfig:
    name: str
    provider: str
    command: str
    env: dict[str, str]


def load_model_configs(path: Path) -> dict[str, ModelConfig]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    models: dict[str, ModelConfig] = {}
    for name, item in data.get("models", {}).items():
        models[name] = ModelConfig(
            name=name,
            provider=str(item.get("provider", name)),
            command=str(item["command"]),
            env={str(key): str(value) for key, value in item.get("env", {}).items()},
        )
    return models


def provider_specs_from_models(path: Path, names: list[str]) -> list[ProviderSpec]:
    configs = load_model_configs(path)
    missing = [name for name in names if name not in configs]
    if missing:
        raise ValueError(f"Unknown model config(s): {', '.join(missing)}")
    specs: list[ProviderSpec] = []
    for name in names:
        config = configs[name]
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in config.env.items())
        command = f"{prefix} {config.command}".strip()
        specs.append(ProviderSpec(name=name, command=command))
    return specs
