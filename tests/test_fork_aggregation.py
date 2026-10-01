"""Fork-only: the median/mean blend (``metaculus_bot/fork_aggregation.py``) and its one pipeline hook.

``TestProductionPathPublishesTheBlend`` and ``TestTournamentWorkflowTurnsItOn`` are the sync guards:
an upstream merge that drops the hook, routes a median around ``_numeric_combine_strategy`` or loses
the workflow flag fails here instead of silently publishing the median again.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml
from forecasting_tools import BinaryQuestion, GeneralLlm, MultipleChoiceQuestion, NumericQuestion, PredictedOptionList
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile

from metaculus_bot.aggregation_pipeline import AggregationPipeline
from metaculus_bot.aggregation_strategies import (
    AggregationStrategy,
    combine_binary_predictions,
    combine_multiple_choice_predictions,
)
from metaculus_bot.fork_aggregation import (
    FORK_BLEND_AGGREGATION_ENABLED_ENV,
    aggregate_binary_blend,
    aggregate_numeric_blend,
    blend_values,
)
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric.utils import aggregate_numeric
from tests.mantic_fakes import load_preseason_date_question
from tests.pipeline_test_helpers import assert_server_accepts_cdf, cdf_heights, make_real_numeric_question, on_mantic

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOURNAMENT_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "run_bot_on_tournament.yaml"

# Upstream q44802 (summer 2026): two members near 0.58 and one dissenter at 0.33.
_OUTLIER_MEMBERS = [0.59, 0.58, 0.33]
# 0.7 * median 0.58 + 0.3 * mean 0.50
_OUTLIER_BLEND = 0.556

# Upstream q45092 (summer 2026), raw member ballots.
_MC_OPTIONS = ["0 or 1", "2 or 3", "4", "5", ">5"]
_MC_MEMBERS = [
    [0.12, 0.57, 0.15, 0.06, 0.10],
    [0.03, 0.81, 0.13, 0.02, 0.01],
    [0.02, 0.83, 0.11, 0.03, 0.01],
]


@pytest.fixture
def blend_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FORK_BLEND_AGGREGATION_ENABLED_ENV, "true")


def _binary_question(qid: int = 100) -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will X happen?",
        id_of_question=qid,
        page_url=f"https://metaculus.com/questions/{qid}",
        api_json={"type": "binary"},
    )


def _mc_question(qid: int = 200) -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Which will happen?",
        id_of_question=qid,
        page_url=f"https://metaculus.com/questions/{qid}",
        options=_MC_OPTIONS,
        api_json={"type": "multiple_choice"},
    )


def _mc_members() -> list[PredictedOptionList]:
    return [
        PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=name, probability=p) for name, p in zip(_MC_OPTIONS, ballot, strict=True)
            ]
        )
        for ballot in _MC_MEMBERS
    ]


def _percentile_members(question: NumericQuestion) -> list[NumericDistribution]:
    """Three percentile members with different centers and widths, so median and mean differ."""
    span = question.upper_bound - question.lower_bound
    members = []
    for center, width in ((0.40, 0.30), (0.45, 0.20), (0.60, 0.50)):
        declared = [
            Percentile(percentile=p, value=question.lower_bound + span * (center + width * (p - 0.5)))
            for p in STANDARD_PERCENTILES
        ]
        sanitized, zero_point = sanitize_percentiles(declared, question, model_name="test-model")
        members.append(build_numeric_distribution(sanitized, question, zero_point, model_name="test-model"))
    return members


def _pipeline() -> AggregationPipeline:
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    return AggregationPipeline(
        strategy=AggregationStrategy.CONDITIONAL_STACKING,
        stacker_llm=test_llm,
        parser_llm=test_llm,
        stacking_fallback_on_failure=True,
        stacking_randomize_order=False,
        stacking_spread_thresholds={"binary": 0.15, "mc": 0.20, "numeric": 0.15},
        discrete_integer_votes=defaultdict(list),
    )


def _option_probs(prediction: PredictedOptionList) -> dict[str, float]:
    return {o.option_name: o.probability for o in prediction.predicted_options}


class TestBlendMath:
    def test_one_outlier_moves_thirty_percent_of_the_way_to_the_mean(self) -> None:
        assert aggregate_binary_blend(_OUTLIER_MEMBERS) == pytest.approx(_OUTLIER_BLEND)

    def test_three_members_weigh_low_middle_high_by_one_eight_one(self) -> None:
        low, mid, high = 0.1, 0.5, 0.6
        assert blend_values([high, low, mid]) == pytest.approx(0.1 * low + 0.8 * mid + 0.1 * high)

    @pytest.mark.parametrize(
        "values",
        [[0.2, 0.3, 0.9], [0.05, 0.4, 0.45], [0.5, 0.5, 0.5], [0.1, 0.2, 0.3, 0.7, 0.8, 0.95], [0.3, 0.6]],
    )
    def test_lands_between_median_and_mean(self, values: list[float]) -> None:
        median, mean = float(np.median(values)), float(np.mean(values))
        assert min(median, mean) - 1e-12 <= blend_values(values) <= max(median, mean) + 1e-12

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            blend_values([])

    def test_binary_dispatch(self) -> None:
        assert combine_binary_predictions(_OUTLIER_MEMBERS, AggregationStrategy.BLEND) == pytest.approx(_OUTLIER_BLEND)

    def test_mc_blends_each_option_then_renormalizes(self) -> None:
        blended = _option_probs(combine_multiple_choice_predictions(_mc_members(), AggregationStrategy.BLEND))
        median = _option_probs(combine_multiple_choice_predictions(_mc_members(), AggregationStrategy.MEDIAN))
        assert sum(blended.values()) == pytest.approx(1.0)
        # ">5": median 0.01, mean 0.04, blend 0.019 before renormalizing by the blended total 1.007.
        assert blended[">5"] == pytest.approx(0.019 / 1.007, abs=1e-3)
        assert blended[">5"] > median[">5"]
        assert blended["2 or 3"] < median["2 or 3"]

    def test_mc_still_rejects_mismatched_options(self) -> None:
        other = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="X", probability=0.5),
                PredictedOption(option_name="Y", probability=0.5),
            ]
        )
        with pytest.raises(ValueError, match="same option names"):
            combine_multiple_choice_predictions([*_mc_members(), other], AggregationStrategy.BLEND)

    def test_numeric_cdf_lies_between_median_and_mean_and_is_legal(self) -> None:
        question = make_real_numeric_question()
        members = _percentile_members(question)
        median = cdf_heights(aggregate_numeric(members, question, "median"))
        mean = cdf_heights(aggregate_numeric(members, question, "mean"))
        blend = cdf_heights(aggregate_numeric_blend(members, question))

        assert np.all(blend >= np.minimum(median, mean) - 1e-6)
        assert np.all(blend <= np.maximum(median, mean) + 1e-6)
        assert np.abs(blend - median).max() > 1e-3
        assert_server_accepts_cdf(
            blend,
            cdf_size=question.cdf_size,
            open_lower=question.open_lower_bound,
            open_upper=question.open_upper_bound,
        )


class TestPipelineHook:
    @pytest.mark.parametrize(
        "question",
        [_binary_question(), _mc_question(), make_real_numeric_question()],
        ids=["binary", "multiple-choice", "numeric"],
    )
    def test_flag_off_keeps_the_median(self, question) -> None:
        assert _pipeline()._numeric_combine_strategy(question, AggregationStrategy.MEDIAN) is AggregationStrategy.MEDIAN

    def test_explicit_false_keeps_the_median(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FORK_BLEND_AGGREGATION_ENABLED_ENV, "false")
        result = _pipeline()._numeric_combine_strategy(_binary_question(), AggregationStrategy.MEDIAN)
        assert result is AggregationStrategy.MEDIAN

    @pytest.mark.usefixtures("blend_on")
    @pytest.mark.parametrize(
        "question",
        [_binary_question(), _mc_question(), make_real_numeric_question()],
        ids=["binary", "multiple-choice", "numeric"],
    )
    def test_flag_on_turns_the_median_into_the_blend(self, question) -> None:
        assert _pipeline()._numeric_combine_strategy(question, AggregationStrategy.MEDIAN) is AggregationStrategy.BLEND

    @pytest.mark.usefixtures("blend_on")
    def test_flag_on_leaves_a_mean_alone(self) -> None:
        result = _pipeline()._numeric_combine_strategy(_binary_question(), AggregationStrategy.MEAN)
        assert result is AggregationStrategy.MEAN

    @pytest.mark.usefixtures("blend_on")
    def test_per_bin_mantic_members_keep_upstreams_mean(self) -> None:
        question = on_mantic(load_preseason_date_question())
        result = _pipeline()._numeric_combine_strategy(question, AggregationStrategy.MEDIAN)
        assert result is AggregationStrategy.MEAN


@pytest.mark.usefixtures("blend_on")
class TestProductionPathPublishesTheBlend:
    """The production strategy (CONDITIONAL_STACKING, stacking off) publishes the blend on every median path."""

    def test_binary_base_combine(self) -> None:
        combined = _pipeline().base_combine(cast("list[PredictionTypes]", _OUTLIER_MEMBERS), _binary_question())
        assert combined == pytest.approx(_OUTLIER_BLEND)

    def test_binary_median_fallback(self) -> None:
        combined = _pipeline()._median_fallback(cast("list[PredictionTypes]", _OUTLIER_MEMBERS), _binary_question())
        assert combined == pytest.approx(_OUTLIER_BLEND)

    def test_mc_base_combine(self) -> None:
        combined = _pipeline().base_combine(cast("list[PredictionTypes]", _mc_members()), _mc_question())
        assert isinstance(combined, PredictedOptionList)
        expected = combine_multiple_choice_predictions(_mc_members(), AggregationStrategy.BLEND)
        assert _option_probs(combined) == pytest.approx(_option_probs(expected))

    def test_numeric_base_combine_publishes_and_records_the_blend(self) -> None:
        pipeline = _pipeline()
        question = make_real_numeric_question()
        members = _percentile_members(question)

        combined = pipeline.base_combine(cast("list[PredictionTypes]", members), question)

        assert isinstance(combined, NumericDistribution)
        assert np.array_equal(cdf_heights(combined), cdf_heights(aggregate_numeric_blend(members, question)))
        assert pipeline.numeric_combine_methods == {question.id_of_question: "blend"}


class TestTournamentWorkflowTurnsItOn:
    def test_the_run_bot_step_sets_the_flag(self) -> None:
        workflow = yaml.safe_load(_TOURNAMENT_WORKFLOW.read_text())
        envs = [step.get("env") or {} for job in workflow["jobs"].values() for step in job.get("steps", [])]
        assert any(str(env.get(FORK_BLEND_AGGREGATION_ENABLED_ENV)).lower() == "true" for env in envs)
