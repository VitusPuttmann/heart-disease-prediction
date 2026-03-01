"""
Dataclass interface for ML models.
"""

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True, slots=True)
class ModelSpec:
    params: Mapping[str, Any]
    fit: Callable[..., Any]
    evaluate: Callable[..., Any]
    predict: Callable[..., Any]
    store: Callable[..., Any]
