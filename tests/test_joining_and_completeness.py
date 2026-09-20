from pathlib import Path

import pandas as pd

from src.completeness import apply_completion_rules, load_completion_rules, participant_completion_summary
from src.joining import join_sources
from src.qualtrics import clean_qualtrics_csv
from src.reporting import summary_metrics

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
    source_columns = [
        column
        for column in joined.columns
        if column.startswith(("pretest__", "storybook__", "posttest__"))
    ]
    assert source_columns[0].startswith("pretest__")
    assert next(index for index, column in enumerate(source_columns) if column.startswith("storybook__")) > 0
    assert next(index for index, column in enumerate(source_columns) if column.startswith("posttest__")) > next(
        index for index, column in enumerate(source_columns) if column.startswith("storybook__")
    )


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


def test_pretest_completion_reports_missing_sections():
    frame = pd.DataFrame(
        {
            "participant_id": ["complete", "missing"],
            "pretest__Country": ["US", "US"],
            "pretest__CAge": ["8", ""],
            "pretest__PSDQ_1": ["1", ""],
        }
    )
    rules = {
        "pretest": {
            "sections": {
                "demographics": ["Country", "CAge"],
                "sdq_scale": ["PSDQ_1"],
            },
            "completion_mode": "all_non_empty",
        },
        "posttest": {"required_columns": ["participant_id"]},
        "storybook": {"required_columns": ["participant_id"]},
    }

    result = apply_completion_rules(frame, rules)

    assert result.loc[0, "pretest_complete"]
    assert pd.isna(result.loc[0, "pretest_missing_sections"])
    assert result.loc[1, "pretest_missing_sections"] == "Demographics, SDQ Scale"


def test_posttest_completion_requires_all_sections_and_reports_missing_sections():
    frame = pd.DataFrame(
        {
            "participant_id": ["complete", "missing"],
            "posttest__CEnjoyment": ["1", "1"],
            "posttest__CUnderstanding": ["1", ""],
            "posttest__CLearning-LL": ["1", "1"],
            "posttest__C L L-WordsChange": ["1", "1"],
            "posttest__PEnjoyment": ["1", "1"],
            "posttest__PUnderstanding": ["1", "1"],
            "posttest__PLearning-LL": ["1", "1"],
        }
    )
    frame = frame.rename(columns={"posttest__C L L-WordsChange": "posttest__CLL-WordsChange"})
    rules = {
        "pretest": {"required_columns": ["participant_id"]},
        "posttest": {
            "sections": {
                "child_program_eval": ["CEnjoyment", "CUnderstanding"],
                "child_learning": ["CLearning-LL"],
                "child_love_language_change": ["CLL-WordsChange"],
                "parent_program_eval": ["PEnjoyment", "PUnderstanding"],
                "parent_learning": ["PLearning-LL"],
            }
        },
        "storybook": {"required_columns": ["participant_id"]},
    }

    result = apply_completion_rules(frame, rules)

    assert result.loc[0, "posttest_complete"]
    assert pd.isna(result.loc[0, "posttest_missing_sections"])
    assert not result.loc[1, "posttest_complete"]
    assert result.loc[1, "posttest_missing_sections"] == "Child Program Eval"


def test_summary_metrics_counts_completed_pretest_participants():
    joined = pd.DataFrame(
        {
            "participant_id": ["one", "one", "two"],
            "pretest_present": [True, True, True],
            "posttest_present": [False, False, False],
            "storybook_present": [False, False, False],
            "pretest_complete": [True, True, False],
            "posttest_complete": [True, True, False],
            "storybook_complete": [False, False, False],
            "fully_completed": [False, False, False],
        }
    )

    metrics = summary_metrics(joined)

    assert metrics["completed_pretest_participants"] == 1
    assert metrics["completed_posttest_participants"] == 1
