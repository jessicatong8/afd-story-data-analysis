from pathlib import Path

import pandas as pd

from src.completeness import (
    apply_completion_rules,
    apply_storybook_overrides,
    load_completion_rules,
    load_storybook_overrides,
)
from src.joining import join_sources
from src.qualtrics import clean_qualtrics_csv
from src.reporting import summary_metrics

INPUT_DATA = Path(__file__).parents[1] / "input-data"
CONFIG = Path(__file__).parents[1] / "config"
PRETEST = next(INPUT_DATA.glob("*Pre-Test*.csv"))
POSTTEST = next(INPUT_DATA.glob("*Post-Test*.csv"))


def test_sample_pipeline_counts_match_expected_study_metrics():
    pretest = clean_qualtrics_csv(PRETEST, "pretest")
    posttest = clean_qualtrics_csv(POSTTEST, "posttest")

    # Build an offline Supabase fixture from valid pre-test IDs. The three
    # outage overrides are intentionally absent, and four confirmed rows are
    # incomplete so the expected storybook total is 28 + 3 inferred = 31.
    override_ids = {
        override["participant_id"]
        for override in load_storybook_overrides(CONFIG / "storybook_overrides.yaml")
    }
    empty_storybook = pd.DataFrame(
        {
            "participant_id": pd.Series(dtype="string"),
            "book_completed": pd.Series(dtype="boolean"),
            "game_completed": pd.Series(dtype="boolean"),
        }
    )
    preliminary = join_sources(pretest.data, posttest.data, empty_storybook)
    preliminary = apply_completion_rules(
        preliminary, load_completion_rules(CONFIG / "completion_rules.yaml")
    )
    pretest_ids = set(pretest.data["participant_id"])
    posttest_complete_ids = set(
        preliminary.loc[
            preliminary["posttest_complete"], "participant_id"
        ].dropna()
    )
    confirmed_complete_ids = sorted(posttest_complete_ids - override_ids)[:27]
    confirmed_true_nonposttest_id = sorted(
        pretest_ids - set(confirmed_complete_ids) - override_ids
    )[0]
    remaining_ids = sorted(
        pretest_ids
        - set(confirmed_complete_ids)
        - {confirmed_true_nonposttest_id}
        - override_ids
    )[:4]
    storybook_ids = (
        confirmed_complete_ids + [confirmed_true_nonposttest_id] + remaining_ids
    )
    storybook = pd.DataFrame(
        {
            "participant_id": storybook_ids,
            "book_completed": [True] * 28 + [False] * 4,
            "game_completed": [True] * 28 + [False] * 4,
        }
    )

    joined = join_sources(pretest.data, posttest.data, storybook)
    rules = load_completion_rules(CONFIG / "completion_rules.yaml")
    joined = apply_completion_rules(joined, rules)
    joined = apply_storybook_overrides(
        joined, load_storybook_overrides(CONFIG / "storybook_overrides.yaml")
    )
    joined = apply_completion_rules(joined, rules)
    metrics = summary_metrics(joined)

    assert metrics["completed_pretest_participants"] == 36
    assert metrics["completed_posttest_participants"] == 30
    assert metrics["completed_storybook_participants"] == 31
    assert metrics["fully_completed_participants"] == 30
    assert metrics["confirmed_storybook_participants"] == 32
