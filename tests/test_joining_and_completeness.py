from pathlib import Path

import pandas as pd

from src.completeness import apply_completion_rules, load_completion_rules, participant_completion_summary
from src.joining import join_sources
from src.qualtrics import clean_qualtrics_csv

INPUT_DATA = Path(__file__).parents[1] / "input-data"
PRETEST = next(INPUT_DATA.glob("*Pre-Test*.csv"))
POSTTEST = next(INPUT_DATA.glob("*Post-Test*.csv"))
RULES = Path(__file__).parents[1] / "config" / "completion_rules.yaml"


def _cleaned_sources():
    return (
        clean_qualtrics_csv(PRETEST, "pretest").data,
        clean_qualtrics_csv(POSTTEST, "posttest").data,
    )


def test_full_outer_join_retains_duplicate_posttest_rows():
    pretest, posttest = _cleaned_sources()
    storybook = pd.DataFrame(
        {
            "participant_id": ["57822002", "storybook-only"],
            "book_completed": [True, False],
        }
    )

    joined = join_sources(pretest, posttest, storybook)

    assert len(joined) >= len(pretest)
    assert joined["posttest__participant_id"].notna().sum() == len(posttest)
    assert joined["storybook_present"].sum() == 1
    assert "storybook-only" not in set(joined["participant_id"].dropna())
    assert "missing_multiple_sources" in set(joined["join_status"])


def test_join_excludes_blank_storybook_ids():
    pretest = pd.DataFrame({"participant_id": ["participant-1"]})
    posttest = pd.DataFrame({"participant_id": ["participant-1"]})
    storybook = pd.DataFrame(
        {
            "participant_id": ["participant-1", "", None, "not-in-pretest"],
            "book_completed": [True, True, True, True],
        }
    )

    joined = join_sources(pretest, posttest, storybook)

    assert joined["storybook_present"].sum() == 1
    assert set(joined["participant_id"].dropna()) == {"participant-1"}


def test_completion_rules_and_participant_summary():
    pretest, posttest = _cleaned_sources()
    storybook = pd.DataFrame(
        {"participant_id": ["57822002"], "book_completed": [True]}
    )
    joined = join_sources(pretest, posttest, storybook)
    completed = apply_completion_rules(joined, load_completion_rules(RULES))
    summary = participant_completion_summary(completed)

    assert "fully_completed" in completed.columns
    assert summary["participant_id"].is_unique
    assert summary["fully_completed"].dtype == bool
