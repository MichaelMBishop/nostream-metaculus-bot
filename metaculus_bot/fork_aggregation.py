"""Fork-only: a median-led blend of the median and the mean, in place of the plain median.

``FORK_BLEND_AGGREGATION_ENABLED`` turns it on (off by default, so upstream behaviour and upstream's
tests are unchanged). When on, every path that would combine members by MEDIAN combines them by
``BLEND_MEDIAN_WEIGHT * median + (1 - BLEND_MEDIAN_WEIGHT) * mean`` instead: per probability for
binary, per option for multiple choice (then clamped and renormalized like the median), and
pointwise in CDF space for numeric, discrete and date questions. With three members and a weight of
0.7, that is weights 0.1 / 0.8 / 0.1 on the low, middle and high member.

The hook is ``AggregationPipeline._numeric_combine_strategy``, the per-question seam all three
median paths (base combine, simple combine, stacker fallback) pass through; it maps MEDIAN to
``AggregationStrategy.BLEND``, so ``NUMERIC_AGGREGATE`` logs ``method=blend``. Per-bin Mantic
members keep upstream's MEAN. Background and worked examples:
``scratch/spread_analysis_2026-09-30/`` (gitignored) and FORK.md.

Constants live here rather than in ``constants.py`` so upstream syncs never conflict over them.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

import numpy as np
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.constants import env_flag_enabled

# Private helpers: the blend must line members up and legalize the result exactly as
# aggregate_numeric does. A rename upstream fails this import, and CI, loudly.
from metaculus_bot.numeric.utils import (
    _canonical_cdf_length,
    _cdf_heights_on_canonical_grid,
    _postprocess_ensemble_cdf,
)

FORK_BLEND_AGGREGATION_ENABLED_ENV: str = "FORK_BLEND_AGGREGATION_ENABLED"
# Weight on the median; the rest goes to the mean. 0.7 moves the aggregate 30% of the way from
# the median toward the mean (a = 0.1 on each extreme member when there are three).
BLEND_MEDIAN_WEIGHT: float = 0.7
BLEND_METHOD_LABEL: str = "blend"


def blend_aggregation_enabled() -> bool:
    return env_flag_enabled(FORK_BLEND_AGGREGATION_ENABLED_ENV)


def blend_values(values: Sequence[float]) -> float:
    """``BLEND_MEDIAN_WEIGHT * median + (1 - BLEND_MEDIAN_WEIGHT) * mean`` of ``values``."""
    if not values:
        raise ValueError("Cannot blend an empty list of values")
    return BLEND_MEDIAN_WEIGHT * statistics.median(values) + (1.0 - BLEND_MEDIAN_WEIGHT) * statistics.fmean(values)


def aggregate_binary_blend(predictions: Sequence[float]) -> float:
    """The blend of binary forecasts, rounded to three decimals like the median."""
    return round(blend_values(list(predictions)), 3)


def aggregate_numeric_blend(
    predictions: Sequence[NumericDistribution],
    question: NumericQuestion,
) -> NumericDistribution:
    """Pointwise blend of member CDF heights on the canonical grid, legalized like the median and mean."""
    if not predictions:
        raise ValueError("Cannot aggregate empty list of numeric predictions")
    n_points = _canonical_cdf_length(question)
    heights = np.vstack(
        [
            _cdf_heights_on_canonical_grid(prediction, n_points, question, index)
            for index, prediction in enumerate(predictions)
        ]
    )
    p_vals = BLEND_MEDIAN_WEIGHT * np.median(heights, axis=0) + (1.0 - BLEND_MEDIAN_WEIGHT) * heights.mean(axis=0)
    return _postprocess_ensemble_cdf(p_vals, question, method_label=BLEND_METHOD_LABEL)
