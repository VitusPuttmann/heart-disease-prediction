"""
Function for leading yaml files.
"""

from typing import Any
import yaml


def load_yaml(file_path: str) -> dict[str, Any]:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
