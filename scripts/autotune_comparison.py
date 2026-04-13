#!/usr/bin/env python3
"""Autotune comparison — DEPRECATED, see convergence_experiment.py.

The original version of this script compared @triton.autotune vs Noeris bandit
vs fixed curated configs. However, the @triton.autotune path produced inflated
numbers because autotune's internal ``do_bench`` does not flush L2 cache between
trials. Our "fair timer" fix only re-timed the *winner* — autotune still used
do_bench internally to *select* the config, so the comparison was apples-to-oranges.

The correct comparison is: exhaustive grid search (all configs, fair timer) vs
bandit (few configs, fair timer) vs fixed curated (1 config, fair timer). This is
exactly what ``scripts/convergence_experiment.py`` does. Results are in
``docs/results/t4-convergence-experiment.json``.

Key finding from convergence experiment:
  - Bandit reaches >=98% of exhaustive-search optimal in 1 iteration (6 configs)
  - 8.3x reduction in GPU cost vs exhaustive search

Usage::

    # Run the convergence experiment directly:
    python3 scripts/convergence_experiment.py [--output convergence.json]
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    print("autotune_comparison.py is deprecated.")
    print("Running convergence_experiment.py instead...\n")
    script = str(__import__("pathlib").Path(__file__).resolve().parent / "convergence_experiment.py")
    return subprocess.run([sys.executable, script] + sys.argv[1:]).returncode


if __name__ == "__main__":
    sys.exit(main())
