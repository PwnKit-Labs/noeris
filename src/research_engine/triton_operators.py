"""Common protocol for parameterized Triton operators.

Each operator (matmul, softmax, rmsnorm, layernorm, etc.) provides:
- A parameter space for kernel tuning
- Curated starting configs
- Shape buckets for workload-aware selection
- A Triton kernel source generator
- A benchmark script generator
- A bucket key function for classifying shapes

The rest of the infrastructure (ConfigDatabase, LLM proposer, Modal runner,
CLI) works uniformly across all operators by dispatching through the registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass(slots=True)
class TritonOperatorSpec:
    """Specification for a parameterized Triton operator.

    Attributes:
        name: Operator identifier (matmul, softmax, rmsnorm, layernorm).
        param_space: Dict of param name -> list of allowed values.
        curated_configs: Hand-picked starting configs.
        shape_buckets: List of shape dicts (operator-specific shape format).
        metric_name: "tflops" for compute-bound, "gb_per_s" for memory-bound.
        config_id_fn: Function mapping a config dict to a stable string ID.
        shape_bucket_fn: Function mapping a shape dict to a bucket name.
        benchmark_script_fn: Function generating the self-contained benchmark.
        grid_generator_fn: Function producing the systematic config grid.
    """

    name: str
    param_space: dict[str, list[int]]
    curated_configs: list[dict[str, int]]
    shape_buckets: list[dict[str, object]]
    metric_name: str
    config_id_fn: Callable[[dict[str, int]], str]
    shape_bucket_fn: Callable[[dict[str, int]], str]
    benchmark_script_fn: Callable[[list[dict[str, int]], list[dict]], str]
    grid_generator_fn: Callable[..., list[dict[str, int]]]
    shared_memory_check_fn: Callable[[dict[str, int]], bool] = field(
        default=lambda config: True,
    )
    description: str = ""


class OperatorRegistry:
    """Registry mapping operator names to their specs."""

    def __init__(self) -> None:
        self._operators: dict[str, TritonOperatorSpec] = {}

    def register(self, spec: TritonOperatorSpec) -> None:
        self._operators[spec.name] = spec

    def get(self, name: str) -> TritonOperatorSpec:
        if name not in self._operators:
            raise KeyError(
                f"Unknown operator: {name!r}. "
                f"Available: {sorted(self._operators.keys())}"
            )
        return self._operators[name]

    def names(self) -> list[str]:
        return sorted(self._operators.keys())


REGISTRY = OperatorRegistry()


def register_operator(spec: TritonOperatorSpec) -> TritonOperatorSpec:
    """Decorator-friendly operator registration."""
    REGISTRY.register(spec)
    return spec
