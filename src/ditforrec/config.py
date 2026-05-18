from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigNode(dict):
    def __getattr__(self, item: str) -> Any:
        value = self[item]
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            value = ConfigNode(value)
            self[item] = value
        return value


def load_config(path: str | Path) -> ConfigNode:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return ConfigNode(data)
