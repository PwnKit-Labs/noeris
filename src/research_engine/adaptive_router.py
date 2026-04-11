"""Adaptive router that learns which selector to use per operator.

Motivation
----------
Cross-run experiments show that cost_model and bandit are *not* interchangeable:

- **bandit** wins on matmul by a wide margin (+134% vs cost_model's +45%).
- **cost_model** wins on attention (+66% vs bandit's +59%).
- On rmsnorm and cross_entropy both tie.

A naive 50/50 ensemble (ensemble_selector.py) fails on matmul: it halves
the bandit's budget and lands at cost_model's performance level (~+45%).

The right fix is to route adaptively: maintain a *bandit over selectors*,
where each "meta-arm" is one of the three selection strategies. After each
iteration we observe whether the best metric improved, and update the
chosen arm's Beta posterior accordingly. Future arm selections are drawn
via Thompson sampling over those posteriors, giving a principled
exploration-exploitation tradeoff over strategies themselves.

Arms
----
- ``cost_model`` — cost-model-ranked grid candidates
- ``bandit``     — Thompson-sampling BanditSelector
- ``baseline``   — frontier-slot exploration (no model)

Usage::

    from research_engine.adaptive_router import AdaptiveRouter

    router = AdaptiveRouter(seed=42)
    router.load_from_database(database)

    # At config-selection time:
    configs, chosen_arm = router.select_configs(
        spec=spec,
        database=database,
        hardware="A100",
        shapes=shapes,
        max_configs=8,
        cost_model=cost_model,
        bandit_selector=bandit,
    )

    # After benchmarking:
    improved = new_best > old_best
    router.record_outcome(operator="matmul", arm=chosen_arm, improvement=improved)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

# The three arms available to the router.
ARM_COST_MODEL = "cost_model"
ARM_BANDIT = "bandit"
ARM_BASELINE = "baseline"

_ALL_ARMS = (ARM_COST_MODEL, ARM_BANDIT, ARM_BASELINE)

# Prior hyperparameters for each arm's Beta posterior.  Beta(1, 1) = uniform,
# which gives equal initial selection probability for all arms.
_PRIOR_ALPHA: float = 1.0
_PRIOR_BETA: float = 1.0


@dataclass
class _ArmPosterior:
    """Beta posterior for one (operator, arm) pair.

    Attributes:
        arm: The selector arm name.
        alpha: Beta alpha parameter = prior + successes.
        beta: Beta beta parameter = prior + failures.
    """

    arm: str
    alpha: float = _PRIOR_ALPHA
    beta: float = _PRIOR_BETA

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one Thompson sample from Beta(alpha, beta)."""
        return float(rng.beta(self.alpha, self.beta))

    def update(self, *, success: bool) -> None:
        """Update posterior with one new observation.

        ``success=True``  → this arm improved the best metric → alpha += 1
        ``success=False`` → no improvement                    → beta  += 1
        """
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
        """Number of observations incorporated beyond the prior."""
        return int(round(self.alpha + self.beta - _PRIOR_ALPHA - _PRIOR_BETA))


@dataclass
class AdaptiveRouter:
    """Bandit-over-selectors: learns which selection strategy wins per operator.

    Each (operator, arm) pair maintains an independent Beta posterior.  At
    selection time we Thompson-sample one arm per operator and delegate the
    full config budget to that arm.  After observing real throughput we update
    the chosen arm's posterior.

    Posteriors are initialised from the database using ``selector`` annotations
    stored in result records (field ``run_id`` prefix or explicit ``selector``
    field).  Records without attribution are ignored (the prior Beta(1,1)
    applies uniformly).

    Attributes:
        seed: Optional RNG seed for reproducibility.
    """

    seed: int | None = None

    # Internal: operator -> arm_name -> _ArmPosterior
    _posteriors: dict[str, dict[str, _ArmPosterior]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _get_or_create_arm(self, operator: str, arm: str) -> _ArmPosterior:
        if operator not in self._posteriors:
            self._posteriors[operator] = {}
        if arm not in self._posteriors[operator]:
            self._posteriors[operator][arm] = _ArmPosterior(arm=arm)
        return self._posteriors[operator][arm]

    def load_from_database(self, database: Any) -> None:
        """Replay historical selector attribution from the ConfigDatabase.

        For each result record that carries a ``selector`` field we update the
        corresponding arm's posterior.  The ``selector`` field should be one of
        the ARM_* constants (``"cost_model"``, ``"bandit"``, ``"baseline"``).

        If no results carry selector attribution the router starts from the
        uniform prior Beta(1, 1) for all arms and operators, which is the
        correct Bayesian starting point.

        This method resets all posteriors before replaying — calling it twice
        on the same database is idempotent.
        """
        self._posteriors.clear()

        attributed = 0
        for key, record in database.records.items():
            # Parse operator from key format "operator:bucket:hardware"
            parts = key.split(":")
            if len(parts) == 3:
                operator = parts[0]
            elif len(parts) == 2:
                operator = "matmul"
            else:
                continue

            for result in record.results:
                selector_label = result.get("selector", "")
                if selector_label not in _ALL_ARMS:
                    # No attribution — skip; prior remains.
                    continue
                improved = result.get("selector_improved", None)
                if improved is None:
                    # Attribution present but no outcome recorded — skip.
                    continue
                arm = self._get_or_create_arm(operator, selector_label)
                arm.update(success=bool(improved))
                attributed += 1

        LOGGER.debug(
            "AdaptiveRouter loaded %d attributed outcomes across %d operators",
            attributed,
            len(self._posteriors),
        )

    # ------------------------------------------------------------------
    # Arm selection
    # ------------------------------------------------------------------

    def select_arm(self, operator: str) -> str:
        """Thompson-sample one arm for the given operator.

        Draws one sample from each arm's Beta posterior and returns the arm
        with the highest sample.  Operators with no history use the flat
        Beta(1, 1) prior, giving uniform selection probability initially.

        Args:
            operator: Operator name (e.g. ``"matmul"``).

        Returns:
            One of ``ARM_COST_MODEL``, ``ARM_BANDIT``, or ``ARM_BASELINE``.
        """
        samples: list[tuple[float, str]] = []
        for arm_name in _ALL_ARMS:
            posterior = self._posteriors.get(operator, {}).get(arm_name)
            if posterior is None:
                # Use the prior Beta(1, 1) inline — don't persist yet.
                sample = float(self._rng.beta(_PRIOR_ALPHA, _PRIOR_BETA))
            else:
                sample = posterior.sample(self._rng)
            samples.append((sample, arm_name))

        samples.sort(key=lambda x: -x[0])
        chosen = samples[0][1]
        LOGGER.debug("AdaptiveRouter: operator=%s, chose arm=%s", operator, chosen)
        return chosen

    # ------------------------------------------------------------------
    # Config selection
    # ------------------------------------------------------------------

    def select_configs(
        self,
        *,
        spec: Any,
        database: Any,
        hardware: str,
        shapes: list[dict],
        max_configs: int = 8,
        cost_model: Any = None,
        bandit_selector: Any = None,
        proposed_configs: list[dict[str, int]] | None = None,
    ) -> tuple[list[dict[str, int]], str]:
        """Select configs by routing to the Thompson-sampled arm.

        Picks one arm via Thompson sampling, then delegates the *entire*
        config budget to that arm's underlying selector.  The routing
        decision is returned alongside the configs so the caller can update
        the arm posterior after observing real throughput.

        Args:
            spec: TritonOperatorSpec for the target operator.
            database: ConfigDatabase holding past benchmark results.
            hardware: GPU identifier string (e.g. ``"A100"``).
            shapes: Target shape dicts (operator-specific format).
            max_configs: Maximum number of configs to return.
            cost_model: Optional trained CostModel.
            bandit_selector: Optional BanditSelector instance.
            proposed_configs: Optional LLM-generated configs (passed through
                to the underlying selector).

        Returns:
            A 2-tuple ``(configs, chosen_arm)`` where ``chosen_arm`` is the
            ARM_* constant identifying which selector was used.
        """
        from research_engine.triton_operators import select_configs_for_operator

        chosen_arm = self.select_arm(spec.name)

        if chosen_arm == ARM_BANDIT and bandit_selector is not None:
            configs = bandit_selector.select_configs(
                spec=spec,
                database=database,
                hardware=hardware,
                shapes=shapes,
                max_configs=max_configs,
                proposed_configs=proposed_configs,
            )

        elif chosen_arm == ARM_COST_MODEL and cost_model is not None:
            configs = select_configs_for_operator(
                spec=spec,
                database=database,
                hardware=hardware,
                shapes=shapes,
                max_configs=max_configs,
                proposed_configs=proposed_configs,
                cost_model=cost_model,
                include_curated=True,
            )

        else:
            # ARM_BASELINE, or a requested arm whose dependency is missing —
            # fall back to the frontier-slot selector (no model, no bandit).
            configs = select_configs_for_operator(
                spec=spec,
                database=database,
                hardware=hardware,
                shapes=shapes,
                max_configs=max_configs,
                proposed_configs=proposed_configs,
                cost_model=None,
                include_curated=True,
            )
            # Normalise the arm label so the caller updates the right posterior.
            chosen_arm = ARM_BASELINE

        return configs, chosen_arm

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        *,
        operator: str,
        arm: str,
        improvement: bool,
    ) -> None:
        """Update the Beta posterior for (operator, arm) based on observed outcome.

        The update rule is simple and principled:
        - ``improvement=True``  → the arm's selection led to a new best metric
          for this operator iteration → success → alpha += 1.
        - ``improvement=False`` → the metric did not improve over the previous
          best → failure → beta += 1.

        "Improvement" is defined as: ``new_best_metric > previous_best_metric``
        for the operator's shape buckets in the current run.  The caller is
        responsible for computing this comparison; the router only stores the
        Beta posterior update.

        Args:
            operator: Operator name (must match the one passed to select_arm).
            arm: One of the ARM_* constants (as returned by select_configs).
            improvement: Whether the best metric improved over the prior best.
        """
        if arm not in _ALL_ARMS:
            LOGGER.warning(
                "AdaptiveRouter.record_outcome: unknown arm %r — ignoring", arm
            )
            return

        posterior = self._get_or_create_arm(operator, arm)
        posterior.update(success=improvement)

        LOGGER.debug(
            "AdaptiveRouter: operator=%s arm=%s improvement=%s -> alpha=%.1f beta=%.1f",
            operator,
            arm,
            improvement,
            posterior.alpha,
            posterior.beta,
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def posterior_summary(self, operator: str) -> list[dict]:
        """Return a sorted summary of arm posteriors for an operator.

        Useful for logging and debugging.  Arms with no history are shown with
        their prior parameters.

        Args:
            operator: Operator name.

        Returns:
            List of dicts sorted by ``expected_reward`` descending.
        """
        rows = []
        for arm_name in _ALL_ARMS:
            posterior = self._posteriors.get(operator, {}).get(arm_name)
            if posterior is None:
                alpha, beta = _PRIOR_ALPHA, _PRIOR_BETA
                obs = 0
            else:
                alpha = posterior.alpha
                beta = posterior.beta
                obs = posterior.total_observations
            rows.append(
                {
                    "arm": arm_name,
                    "alpha": round(alpha, 2),
                    "beta": round(beta, 2),
                    "expected_reward": round(alpha / (alpha + beta), 4),
                    "total_observations": obs,
                }
            )
        rows.sort(key=lambda r: -r["expected_reward"])
        return rows
