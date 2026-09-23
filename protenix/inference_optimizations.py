"""Opt-in, source-only inference optimizations for the Protenix v2 runner.

Only random parameter initialization is bypassed during model construction.
The runner requires a strict checkpoint load before it can use that model.
Constant initializers remain active, including non-persistent buffers.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from typing import Iterator


_MODE_VARIABLE = "PROTENIX_PORTABLE_OPTIMIZATIONS"
_FEATURE_VARIABLE = "_PROTENIX_PORTABLE_FEATURES"  # Benchmark ablation only.
_FEATURES = frozenset({
    "lazy_init", "release_prediction",
})


@dataclass(frozen=True)
class PortablePolicy:
    mode: str
    features: frozenset[str]

    def enabled(self, feature: str) -> bool:
        return feature in self.features


def resolve_policy(configs=None) -> PortablePolicy:
    mode = os.environ.get(_MODE_VARIABLE, "off").strip().lower()
    if mode not in {"off", "portable"}:
        raise ValueError(f"{_MODE_VARIABLE} must be 'off' or 'portable', got {mode!r}")
    if mode == "off":
        return PortablePolicy(mode, frozenset())
    raw_features = os.environ.get(_FEATURE_VARIABLE)
    features = (
        frozenset(feature.strip() for feature in raw_features.split(",") if feature.strip())
        if raw_features is not None else _FEATURES
    )
    unknown = features - _FEATURES
    if unknown:
        raise ValueError(f"unknown {_FEATURE_VARIABLE}: {', '.join(sorted(unknown))}")
    if configs is not None and "lazy_init" in features:
        if configs.model_name != "protenix-v2" or not configs.load_strict:
            raise ValueError("portable lazy_init requires protenix-v2 and load_strict=true")
    return PortablePolicy(mode, features)


@contextmanager
def skip_random_initialization() -> Iterator[tuple[str, ...]]:
    """Skip only random initializers while building a fully loaded v2 model.

    Both Protenix's imported alias and its defining module must be patched:
    the former initializes OpenfoldLinear, while the latter initializes
    triangular layers. Everything is restored even if construction fails.
    """
    import torch.nn.init as init
    import protenix.model.modules.primitives as primitives
    import protenix.model.triangular.layers as triangular

    def unchanged(tensor, *args, **kwargs):
        return tensor

    def no_random_init(*args, **kwargs):
        return None

    targets = [
        (init, name, unchanged)
        for name in (
            "trunc_normal_", "normal_", "xavier_uniform_", "xavier_normal_",
            "kaiming_uniform_", "kaiming_normal_", "uniform_",
        )
    ]
    targets.extend([
        (triangular, "trunc_normal_init_", no_random_init),
        (primitives, "trunc_normal_init_", no_random_init),
    ])
    saved = [(module, name, getattr(module, name)) for module, name, _ in targets]
    try:
        for module, name, replacement in targets:
            setattr(module, name, replacement)
        yield tuple(f"{module.__name__}.{name}" for module, name, _ in targets)
    finally:
        for module, name, original in reversed(saved):
            setattr(module, name, original)
