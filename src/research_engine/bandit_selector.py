"""Thompson-sampling bandit config selector for Triton kernel search.

This module provides a second algorithmic axis alongside the LLM proposer and
the learned cost model. Where the cost model predicts performance from config
features, and the LLM proposer generates novel configs from prior insights, the
bandit selector exploits accumulated empirical reward signal via Bayesian
posterior updates.

Algorithm: Thompson Sampling with Beta(alpha, beta) posteriors per
(operator, shape_bucket, hardware, config_id) cell. A config's "success"
is defined as landing in the top-K% of measured throughputs for its bucket.
At selection time we draw one sample from each config's posterior and rank by
sample value, giving a principled exploration-exploitation tradeoff:
- Under-measured configs have diffuse priors (uniform Beta(1,1)) -> high
  exploration probability.
- Well-measured winners converge toward Beta(N_wins+1, 1) -> near-certain
  selection.

This differs from the cost model (regression on config features) and the LLM
proposer (generative search guided by insights) in that it makes no assumptions
about config structure — it is purely empirical, learning from reward histories
stored in ConfigDatabase.records.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

# Top-K% threshold: a config is a "success" if its throughput is at or above
# this percentile within its (operator, bucket, hardware) cohort.
_DEFAULT_TOP_K_PERCENT: float = 30.0

# Beta prior hyperparameters for an unseen config — weakly optimistic to
# encourage exploration of novel configs.
_PRIOR_ALPHA: float = 1.0
_PRIOR_BETA: float = 1.0


@dataclass
class _BetaArm:
    """Beta posterior for one config in one bucket cell.

    Attributes:
        config_id: Stable string ID (e.g. "bm128_bn256_bk64_gm8_w8_s3").
        config: Full parameter dict (kept for return value assembly).
        alpha: Beta distribution alpha = successes + prior_alpha.
        beta: Beta distribution beta = failures + prior_beta.
    """

    config_id: str
    config: dict[str, int]
    alpha: float = _PRIOR_ALPHA
    beta: float = _PRIOR_BETA

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one Thompson sample from the posterior."""
        return float(rng.beta(self.alpha, self.beta))

    def update(self, *, success: bool) -> None:
        """Incorporate one new observation into the posterior."""
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    @property
    def expected_reward(self) -> float:
        """Posterior mean E[theta] = alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def total_observations(self) -> int:
        """Number of observations seen so far (excluding prior)."""
        return int(round(self.alpha + self.beta - _PRIOR_ALPHA - _PRIOR_BETA))


@dataclass
class BanditSelector:
    """Thompson-sampling bandit over config_id per (operator, bucket, hardware).

    Each distinct (operator, bucket, hardware) triple is an independent bandit
    problem. Within each bandit we maintain a Beta posterior per config_id and
    select configs proportional to posterior Thompson samples.

    Usage::

        selector = BanditSelector()
        selector.load_from_database(database, operator="matmul", hardware="A100")
        configs = selector.select_configs(
            spec=spec,
            database=database,
            hardware="A100",
            shapes=shapes,
            max_configs=8,
        )

    The ``select_configs`` signature intentionally mirrors
    ``select_configs_for_operator`` so the CLI can route through either
    without changing call sites.
    """

    top_k_percent: float = _DEFAULT_TOP_K_PERCENT
    seed: int | None = None
    # When True, fall back to spec.curated_configs as the warm-start seed
    # set if a bucket has no history. When False (the default and the
    # research-correct setting), warm-start from a uniform random sample of
    # the operator's grid instead — the bandit then learns feasibility and
    # quality from scratch with no hand-curated prior.
    use_curated_seeds: bool = False
    # Number of random configs to draw on a cold-start bucket. Roughly
    # matches the historical curated count for each operator.
    warm_start_size: int = 8

    # Internal state: cell_key -> {config_id -> _BetaArm}
    _arms: dict[str, dict[str, _BetaArm]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _cell_key(self, operator: str, bucket: str, hardware: str) -> str:
        return f"{operator}:{bucket}:{hardware}"

    def _get_or_create_arm(
        self,
        cell: str,
        config_id: str,
        config: dict[str, int],
    ) -> _BetaArm:
        if cell not in self._arms:
            self._arms[cell] = {}
        if config_id not in self._arms[cell]:
            self._arms[cell][config_id] = _BetaArm(
                config_id=config_id,
                config=config,
            )
        return self._arms[cell][config_id]

    def load_from_database(
        self,
        database: Any,  # ConfigDatabase
        *,
        operator: str = "",
        hardware: str = "",
    ) -> None:
        """Replay all stored results into Beta posteriors.

        For each (operator, bucket, hardware) cell we:
        1. Compute the top-K% throughput threshold over the *correct* runs
           in that cell (failures contribute zero throughput and so cannot
           be in the top-K, but they still need an arm).
        2. For every recorded result (correct **or** failed) update the
           corresponding arm:
             - failures (correct=False or tflops<=0)  -> success=False
             - correct results above the cell threshold -> success=True
             - correct results below the threshold      -> success=False
        This makes failed runs first-class evidence: a config that crashes
        or produces wrong output gets its Beta posterior pushed toward
        zero, exactly like a config that ran correctly but slowly. The
        method is idempotent — arm state is cleared at the start.
        """
        self._arms.clear()

        for key, record in database.records.items():
            # Parse record key format: "operator:bucket:hardware" (current)
            # or legacy "bucket:hardware" (two-part).
            parts = key.split(":")
            if len(parts) == 3:
                rec_operator, rec_bucket, rec_hardware = parts
            elif len(parts) == 2:
                rec_operator, rec_bucket, rec_hardware = "matmul", parts[0], parts[1]
            else:
                continue

            if operator and rec_operator != operator:
                continue
            if hardware and rec_hardware != hardware:
                continue

            cell = self._cell_key(rec_operator, rec_bucket, rec_hardware)

            # Compute the cohort top-K% threshold over the successful runs
            # only (failed runs have no meaningful throughput).
            correct_results = [
                r for r in record.results
                if r.get("correct") and r.get("tflops") is not None and r["tflops"] > 0
            ]
            if correct_results:
                tflops_values = [r["tflops"] for r in correct_results]
                threshold = float(
                    np.percentile(tflops_values, 100.0 - self.top_k_percent)
                )
            else:
                # No successful runs in this cell yet — every observation we
                # have is a failure, so the threshold is irrelevant; all
                # arms will be marked failure below.
                threshold = float("inf")

            for result in record.results:
                cid = result.get("config_id", "")
                config = result.get("config", {})
                if not cid or not config:
                    continue
                arm = self._get_or_create_arm(cell, cid, config)
                tflops = result.get("tflops") or 0.0
                is_correct = bool(result.get("correct"))
                if not is_correct or tflops <= 0:
                    # Failed run: legitimate reward=0 sample.
                    arm.update(success=False)
                else:
                    arm.update(success=tflops >= threshold)

        total_arms = sum(len(v) for v in self._arms.values())
        LOGGER.debug(
            "BanditSelector loaded %d arms across %d cells",
            total_arms,
            len(self._arms),
        )

    # ------------------------------------------------------------------
    # Config selection
    # ------------------------------------------------------------------

    def ranked_configs_for_bucket(
        self,
        *,
        operator: str,
        bucket: str,
        hardware: str,
    ) -> list[dict[str, int]]:
        """Return configs for a bucket sorted by Thompson sample (descending).

        Configs with no prior observations default to Beta(1, 1) giving a
        uniform draw — pure exploration for unseen arms.
        """
        cell = self._cell_key(operator, bucket, hardware)
        arms = self._arms.get(cell, {})
        if not arms:
            return []

        samples = [(arm.sample(self._rng), arm.config) for arm in arms.values()]
        samples.sort(key=lambda x: -x[0])
        return [config for _, config in samples]

    def select_configs(
        self,
        *,
        spec: Any,  # TritonOperatorSpec
        database: Any,  # ConfigDatabase
        hardware: str,
        shapes: list[dict],
        max_configs: int = 8,
        proposed_configs: list[dict[str, int]] | None = None,
        use_curated_seeds: bool | None = None,
    ) -> list[dict[str, int]]:
        """Select configs using Thompson sampling.

        Slot allocation:
        1. Thompson-sampled configs from empirical posteriors (per bucket).
        2. LLM-proposed configs (if provided).
        3. Cold-start warm-up: if a target bucket has *no* history, draw
           ``warm_start_size`` configs uniformly at random from
           ``spec.grid_generator_fn`` (or, if ``use_curated_seeds=True``,
           from ``spec.curated_configs``). The bandit therefore learns
           feasibility and quality from scratch with no hand-curated prior.
        4. Random grid exploration for any remaining slots.

        Args:
            spec: TritonOperatorSpec providing curated_configs, config_id_fn,
                shape_bucket_fn, and grid_generator_fn.
            database: ConfigDatabase from which reward history is loaded.
            hardware: GPU identifier string (e.g. "A100").
            shapes: Target shape dicts for the run (operator-specific format).
            max_configs: Maximum number of configs to return.
            proposed_configs: Optional pre-computed LLM proposals.
            use_curated_seeds: Override ``self.use_curated_seeds`` for this
                call. ``None`` (default) uses the instance setting.

        Returns:
            Ordered list of config dicts (highest-priority first).
        """
        # Reload posteriors fresh from the database each call so we always
        # reflect the latest observed rewards without manual state management.
        self.load_from_database(database, operator=spec.name, hardware=hardware)

        if use_curated_seeds is None:
            use_curated_seeds = self.use_curated_seeds

        selected: list[dict[str, int]] = []
        seen_ids: set[str] = set()

        def _add(config: dict[str, int]) -> bool:
            cid = spec.config_id_fn(config)
            if cid in seen_ids or len(selected) >= max_configs:
                return False
            seen_ids.add(cid)
            selected.append(config)
            return True

        # Slot 1: Thompson-sampled configs (one draw per bucket for the
        # target shapes). Track which buckets have prior arms so we know
        # which need a cold-start warm-up below.
        bucket_order: list[str] = []
        bucket_has_history: dict[str, bool] = {}
        for shape in shapes:
            bucket = spec.shape_bucket_fn(shape)
            if bucket in bucket_has_history:
                continue
            bucket_order.append(bucket)
            cell = self._cell_key(spec.name, bucket, hardware)
            bucket_has_history[bucket] = bool(self._arms.get(cell))

            ranked = self.ranked_configs_for_bucket(
                operator=spec.name, bucket=bucket, hardware=hardware,
            )
            for config in ranked:
                _add(config)
                if len(selected) >= max_configs:
                    break

        # Slot 2: LLM-proposed configs
        for config in proposed_configs or []:
            _add(config)

        # Determine already-tested IDs for this operator+hardware pair
        # (correct *or* failed — both are evidence for the bandit and we
        # don't want to re-issue them on a warm-start).
        tested_ids: set[str] = set()
        for key, record in database.records.items():
            parts = key.split(":")
            if len(parts) == 3:
                rec_op, _, rec_hw = parts
            elif len(parts) == 2:
                rec_op, rec_hw = "matmul", parts[0]
            else:
                continue
            if rec_op != spec.name:
                continue
            if hardware and rec_hw != hardware:
                continue
            for result in record.results:
                tested_ids.add(result.get("config_id", ""))

        # Slot 3: cold-start warm-up. Only fires for buckets that have
        # zero arms (i.e. no history at all). The default behaviour is to
        # sample uniformly at random from the operator's grid; with
        # ``use_curated_seeds=True`` callers can opt back into the legacy
        # curated bootstrap for ablations.
        cold_start_buckets = [b for b in bucket_order if not bucket_has_history[b]]
        if cold_start_buckets and len(selected) < max_configs:
            if use_curated_seeds:
                seed_pool = list(spec.curated_configs)
            else:
                try:
                    seed_pool = spec.grid_generator_fn(
                        include_curated=False, max_configs=500,
                    )
                except TypeError:
                    seed_pool = spec.grid_generator_fn(max_configs=500)
                seed_pool = [
                    c for c in seed_pool
                    if spec.config_id_fn(c) not in tested_ids
                ]
                # Deterministic shuffle so reproducibility is preserved.
                # Use the same RNG that powers Thompson sampling — it was
                # seeded from ``self.seed`` in __post_init__.
                indices = list(range(len(seed_pool)))
                self._rng.shuffle(indices)
                seed_pool = [seed_pool[i] for i in indices]

            n_to_take = max(0, min(self.warm_start_size, max_configs - len(selected)))
            for config in seed_pool[:n_to_take]:
                _add(config)

        # Slot 4: random grid exploration for any remaining slots
        if len(selected) < max_configs:
            try:
                grid = spec.grid_generator_fn(include_curated=False, max_configs=500)
            except TypeError:
                grid = spec.grid_generator_fn(max_configs=500)
            untested = [c for c in grid if spec.config_id_fn(c) not in tested_ids]
            # Use the seeded RNG so the whole selection is reproducible.
            indices = list(range(len(untested)))
            self._rng.shuffle(indices)
            for i in indices:
                _add(untested[i])

        return selected

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def posterior_summary(
        self,
        *,
        operator: str,
        bucket: str,
        hardware: str,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Return the top-N arms by expected reward for inspection/debugging."""
        cell = self._cell_key(operator, bucket, hardware)
        arms = list(self._arms.get(cell, {}).values())
        arms.sort(key=lambda a: -a.expected_reward)
        return [
            {
                "config_id": arm.config_id,
                "alpha": round(arm.alpha, 2),
                "beta": round(arm.beta, 2),
                "expected_reward": round(arm.expected_reward, 4),
                "total_observations": arm.total_observations,
            }
            for arm in arms[:top_n]
        ]
