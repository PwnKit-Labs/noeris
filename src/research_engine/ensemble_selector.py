"""Ensemble config selector combining cost model and Thompson-sampling bandit.

Motivation
----------
Cross-run ablation studies show that the cost model and the bandit are
*complementary* predictors:

- **Cost model** wins on attention and softmax (compute-bound operators where
  config structure — tile sizes, num_warps — has a strong, learnable effect
  on throughput).
- **Bandit** wins on matmul (where empirical reward histories from prior runs
  are rich enough to dominate feature-based prediction).
- They *tie* on cross_entropy.

Rather than picking one winner globally, this module interleaves their picks
in strict alternation: slot 1 goes to the cost model's top candidate, slot 2
to the bandit's top Thompson sample, slot 3 to the cost model's second pick,
and so on.  This is deliberately *not* a weighted-voting or score-fusion
scheme.

Why alternation instead of weighted voting?
-------------------------------------------
Weighted voting requires a calibrated shared score scale — non-trivial because
the cost model produces TFLOPS predictions while the bandit emits Beta
posterior samples (values in [0, 1]).  Normalizing and combining them adds a
tuning knob that would itself need ablation.  Alternation is parameter-free
and ensures the budget is split fairly regardless of scale differences.  If
one selector consistently dominates, an online meta-learner can adjust the
split in a future iteration; alternation is the right *prior* given the
evidence.

Usage::

    from research_engine.ensemble_selector import select_configs_ensemble

    configs = select_configs_ensemble(
        spec=spec,
        database=database,
        hardware="A100",
        shapes=shapes,
        max_configs=8,
        proposed_configs=llm_proposals,      # optional
        cost_model=cost_model,               # optional CostModel
        bandit_selector=bandit,              # optional BanditSelector
    )

If ``cost_model`` is None the ensemble falls back to bandit-only (equivalent
to ``BanditSelector.select_configs``).  If ``bandit_selector`` is None the
ensemble falls back to cost-model-only (equivalent to
``select_configs_for_operator`` with a cost model).  If both are None the
ensemble falls back to the frontier-slot baseline.
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

# Number of grid candidates to draw from the cost-model ranker before
# interleaving.  Must be large enough to give the cost model a meaningful
# pool to reorder, but bounded to keep selection latency low.
_COST_MODEL_CANDIDATES: int = 40


def _collect_tested_ids(
    database: Any,
    *,
    operator: str,
    hardware: str,
) -> set[str]:
    """Return the set of config_id strings already benchmarked.

    Mirrors the logic in ``select_configs_for_operator`` and
    ``BanditSelector.select_configs`` to avoid duplicating untested-grid
    filtering across the two selectors.
    """
    tested: set[str] = set()
    if database is None:
        return tested
    for key, record in database.records.items():
        parts = key.split(":")
        if len(parts) == 3:
            rec_op, _, rec_hw = parts
        elif len(parts) == 2:
            rec_op, rec_hw = "matmul", parts[0]
        else:
            continue
        if operator and rec_op != operator:
            continue
        if hardware and rec_hw != hardware:
            continue
        for result in record.results:
            tested.add(result.get("config_id", ""))
    return tested


def _cost_model_ranked(
    *,
    spec: Any,
    database: Any,
    hardware: str,
    shapes: list[dict],
    tested_ids: set[str],
    cost_model: Any,
    n_candidates: int = _COST_MODEL_CANDIDATES,
) -> list[dict[str, int]]:
    """Return untested grid candidates ranked by cost model prediction.

    Falls back to natural grid order when ``cost_model`` is None.
    """
    try:
        grid = spec.grid_generator_fn(include_curated=False, max_configs=n_candidates)
    except TypeError:
        grid = spec.grid_generator_fn(max_configs=n_candidates)

    untested = [c for c in grid if spec.config_id_fn(c) not in tested_ids]

    if cost_model is not None and untested:
        ranked = cost_model.rank_configs(
            configs=untested,
            shapes=shapes,
            hardware=hardware,
            operator=spec.name,
            top_k=None,
        )
        return [cfg for cfg, _ in ranked]

    return untested


def _bandit_ranked(
    *,
    spec: Any,
    database: Any,
    hardware: str,
    shapes: list[dict],
    tested_ids: set[str],
    bandit_selector: Any,
) -> list[dict[str, int]]:
    """Return untested candidates ranked by Thompson sampling.

    Reloads posteriors from the database (idempotent) and collects ranked
    configs across all target-shape buckets, deduplicating and filtering
    already-tested IDs.
    """
    bandit_selector.load_from_database(
        database, operator=spec.name, hardware=hardware
    )

    ranked: list[dict[str, int]] = []
    seen: set[str] = set()
    buckets_seen: set[str] = set()

    for shape in shapes:
        bucket = spec.shape_bucket_fn(shape)
        if bucket in buckets_seen:
            continue
        buckets_seen.add(bucket)
        for cfg in bandit_selector.ranked_configs_for_bucket(
            operator=spec.name, bucket=bucket, hardware=hardware
        ):
            cid = spec.config_id_fn(cfg)
            if cid not in seen and cid not in tested_ids:
                seen.add(cid)
                ranked.append(cfg)

    return ranked


def select_configs_ensemble(
    *,
    spec: Any,  # TritonOperatorSpec
    database: Any,  # ConfigDatabase
    hardware: str,
    shapes: list[dict],
    max_configs: int = 8,
    proposed_configs: list[dict[str, int]] | None = None,
    cost_model: Any = None,  # Optional CostModel
    bandit_selector: Any = None,  # Optional BanditSelector
    cost_model_candidates: int = _COST_MODEL_CANDIDATES,
    include_curated: bool = True,
) -> list[dict[str, int]]:
    """Select configs by interleaving cost-model and bandit picks.

    Slot allocation (in priority order):

    1. **Incumbent** — best known config from the database for the target
       shapes (carries forward the best result already found).
    2. **LLM-proposed configs** — any ``proposed_configs`` passed in, in order.
    3. **Curated starting configs** not yet benchmarked (optional, controlled
       by ``include_curated``).
    4. **Alternating ensemble** — cost-model and bandit picks are interleaved:

       - Sub-slot 4a: best cost-model candidate (rank 1)
       - Sub-slot 4b: best bandit Thompson sample (rank 1)
       - Sub-slot 4a: second cost-model candidate (rank 2)
       - Sub-slot 4b: second bandit Thompson sample (rank 2)
       - ...

       Configs that both selectors agree on are deduplicated — each config_id
       appears at most once in the output regardless of which selector picked
       it.  If one selector is exhausted before the budget is filled, the
       remaining slots are backfilled from the other.

    Args:
        spec: TritonOperatorSpec providing config_id_fn, shape_bucket_fn,
            grid_generator_fn, curated_configs, and name.
        database: ConfigDatabase holding past benchmark results.
        hardware: GPU identifier string (e.g. "A100", "H100").
        shapes: Target shape dicts (operator-specific format).
        max_configs: Maximum number of configs to return.
        proposed_configs: Optional LLM-generated configs (go into slot 2).
        cost_model: Optional trained CostModel.  When None, the cost-model
            interleave slots are skipped and the bandit fills the full budget.
        bandit_selector: Optional BanditSelector.  When None, the bandit
            interleave slots are skipped and the cost model fills the full
            budget.
        cost_model_candidates: How many grid candidates to draw for the cost
            model to score (default 40, same as ``select_configs_for_operator``).
        include_curated: Whether to include curated starter configs in slot 3.

    Returns:
        Ordered list of config dicts (highest-priority first), length <=
        ``max_configs``, with no duplicate config_ids.
    """
    selected: list[dict[str, int]] = []
    seen_ids: set[str] = set()

    def _add(config: dict[str, int]) -> bool:
        """Add config to selection if unique and within budget."""
        cid = spec.config_id_fn(config)
        if cid in seen_ids or len(selected) >= max_configs:
            return False
        seen_ids.add(cid)
        selected.append(config)
        return True

    # ------------------------------------------------------------------
    # Slot 1: Incumbent — best known config from the database
    # ------------------------------------------------------------------
    if database is not None and shapes:
        for shape in shapes:
            bucket = spec.shape_bucket_fn(shape)
            best = database.get_best_config(
                shape=shape,
                hardware=hardware,
                operator=spec.name,
                bucket=bucket,
            )
            if best is not None:
                _add(best)
                break

    # ------------------------------------------------------------------
    # Slot 2: LLM-proposed configs
    # ------------------------------------------------------------------
    for config in proposed_configs or []:
        _add(config)

    # ------------------------------------------------------------------
    # Slot 3: Curated configs not yet benchmarked
    # ------------------------------------------------------------------
    tested_ids = _collect_tested_ids(database, operator=spec.name, hardware=hardware)

    if include_curated:
        for config in spec.curated_configs:
            if spec.config_id_fn(config) not in tested_ids:
                _add(config)

    if len(selected) >= max_configs:
        return selected

    # ------------------------------------------------------------------
    # Slot 4: Alternating ensemble — cost model vs bandit
    # ------------------------------------------------------------------
    # Build ranked candidate lists from each selector (excluding already-seen IDs).
    cm_candidates: list[dict[str, int]] = []
    bandit_candidates: list[dict[str, int]] = []

    if cost_model is not None:
        cm_candidates = _cost_model_ranked(
            spec=spec,
            database=database,
            hardware=hardware,
            shapes=shapes,
            tested_ids=tested_ids,
            cost_model=cost_model,
            n_candidates=cost_model_candidates,
        )

    if bandit_selector is not None:
        bandit_candidates = _bandit_ranked(
            spec=spec,
            database=database,
            hardware=hardware,
            shapes=shapes,
            tested_ids=tested_ids,
            bandit_selector=bandit_selector,
        )

    # Fallback when one or both selectors are absent: draw from grid directly.
    if not cm_candidates and not bandit_candidates:
        # Neither selector provided candidates — fall through to grid
        try:
            grid = spec.grid_generator_fn(
                include_curated=False, max_configs=cost_model_candidates
            )
        except TypeError:
            grid = spec.grid_generator_fn(max_configs=cost_model_candidates)
        for cfg in grid:
            if spec.config_id_fn(cfg) not in tested_ids:
                _add(cfg)
        return selected

    # Interleave cost-model and bandit picks in strict alternation.
    # Deduplication is handled by _add() which checks seen_ids.
    cm_iter = iter(cm_candidates)
    bandit_iter = iter(bandit_candidates)
    cm_exhausted = cost_model is None  # treat absent selector as exhausted
    bandit_exhausted = bandit_selector is None

    while len(selected) < max_configs:
        advanced_any = False

        # Sub-slot A: cost model
        if not cm_exhausted:
            cfg = next(cm_iter, None)
            if cfg is None:
                cm_exhausted = True
            else:
                _add(cfg)
                advanced_any = True

        # Sub-slot B: bandit
        if not bandit_exhausted and len(selected) < max_configs:
            cfg = next(bandit_iter, None)
            if cfg is None:
                bandit_exhausted = True
            else:
                _add(cfg)
                advanced_any = True

        # Both exhausted — break to avoid infinite loop
        if not advanced_any:
            break

    LOGGER.debug(
        "select_configs_ensemble: selected %d configs (max=%d, cm=%s, bandit=%s)",
        len(selected),
        max_configs,
        cost_model is not None,
        bandit_selector is not None,
    )
    return selected
