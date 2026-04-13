"""Runtime adaptive config selection for Triton kernels.

Loads the ConfigDatabase offline and builds an O(1) lookup table indexed
by (operator, shape_bucket, hardware).  At inference time the launcher
calls ``selector.select(**shape_kwargs)`` which buckets the shape and
returns the highest-throughput config, falling back to curated defaults.

Thread-safe: the lookup table is built once at init and is read-only
thereafter; no locks required for concurrent selects.
"""

from __future__ import annotations

import platform
import threading
from pathlib import Path
from typing import Any

from .triton_kernels import ConfigDatabase
from .triton_operators import REGISTRY, TritonOperatorSpec

_DEFAULT_DB_PATHS = [
    Path(".noeris/triton-configs.json"),
    Path(".noeris/colab-configs.json"),
]


def _detect_hardware() -> str:
    """Best-effort GPU name detection without requiring torch at import time."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return platform.processor() or "unknown"


class AdaptiveConfigSelector:
    """Selects optimal Triton config at runtime based on input shape.

    The lookup table maps (shape_bucket) -> best config dict.  It is
    populated from ConfigDatabase entries for the given operator+hardware.
    Missing buckets fall back to the first curated config in the operator
    spec.
    """

    def __init__(
        self,
        operator_name: str,
        hardware: str | None = None,
        db_paths: list[Path] | None = None,
    ) -> None:
        self._spec: TritonOperatorSpec = REGISTRY.get(operator_name)
        self._hardware: str = hardware or _detect_hardware()
        self._table: dict[str, dict[str, int]] = {}
        self._fallback: dict[str, int] = self._spec.curated_configs[0]

        # Load all available databases and merge (later DBs overwrite
        # earlier ones only if they have higher throughput for a bucket).
        paths = db_paths if db_paths is not None else _DEFAULT_DB_PATHS
        best_tflops: dict[str, float] = {}
        for db_path in paths:
            db_path = Path(db_path)
            if not db_path.exists():
                continue
            db = ConfigDatabase(path=db_path)
            for key, record in db.records.items():
                if not key.startswith(f"{self._spec.name}:"):
                    continue
                if self._hardware and not key.endswith(f":{self._hardware}"):
                    continue
                bucket = key.split(":")[1] if ":" in key else key
                if record.best_tflops > best_tflops.get(bucket, 0.0):
                    config = db.get_best_config(
                        shape=record.shape,
                        hardware=self._hardware,
                        operator=self._spec.name,
                        bucket=bucket,
                    )
                    if config is not None:
                        self._table[bucket] = config
                        best_tflops[bucket] = record.best_tflops

    @property
    def operator_name(self) -> str:
        return self._spec.name

    @property
    def hardware(self) -> str:
        return self._hardware

    @property
    def known_buckets(self) -> list[str]:
        return sorted(self._table.keys())

    def select(self, **shape_kwargs: Any) -> dict[str, int]:
        """Return the best config for the given runtime shape.

        Keyword arguments are passed to the operator's ``shape_bucket_fn``
        to determine the bucket, then the lookup table is consulted.
        Falls back to the curated default when no data exists.
        """
        bucket = self._spec.shape_bucket_fn(shape_kwargs)
        return self._table.get(bucket, self._fallback)


# ---- Module-level selector cache (thread-safe, lazy) ----

_selector_cache: dict[str, AdaptiveConfigSelector] = {}
_cache_lock = threading.Lock()


def get_selector(
    operator_name: str,
    hardware: str | None = None,
    db_paths: list[Path] | None = None,
) -> AdaptiveConfigSelector:
    """Return a cached AdaptiveConfigSelector for the given operator.

    The first call constructs the selector; subsequent calls return the
    same instance.  Thread-safe via a module-level lock.
    """
    cache_key = f"{operator_name}:{hardware or 'auto'}"
    if cache_key in _selector_cache:
        return _selector_cache[cache_key]
    with _cache_lock:
        if cache_key not in _selector_cache:
            _selector_cache[cache_key] = AdaptiveConfigSelector(
                operator_name=operator_name,
                hardware=hardware,
                db_paths=db_paths,
            )
    return _selector_cache[cache_key]
