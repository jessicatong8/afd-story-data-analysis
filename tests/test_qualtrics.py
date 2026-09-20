from pathlib import Path

import pandas as pd

from src.qualtrics import (
    DEFAULT_CUTOFF,
    clean_qualtrics_csv,
    filter_by_date_range,
    load_qualtrics_csv,
)

INPUT_DATA = Path(__file__).parents[1] / "input-data"
PRETEST = next(INPUT_DATA.glob("*Pre-Test*.csv"))
POSTTEST = next(INPUT_DATA.glob("*Post-Test*.csv"))


def test_load_qualtrics_removes_question_and_import_rows():
    frame = load_qualtrics_csv(PRETEST)

    assert len(frame) == 83
    assert frame.iloc[0]["StartDate"] != "Start Date"
    assert not frame.iloc[0]["StartDate"].startswith('{"ImportId"')


def test_cleaning_filters_previews_and_before_cutoff_rows():
    result = clean_qualtrics_csv(PRETEST, "pretest")

    assert len(result.data) == 73
    assert not {"_has_participant_id", "_is_preview", "_before_collection_cutoff", "_exclusion_reason"}.intersection(result.data.columns)
    assert result.data["participant_id"].dtype == "string"
    assert result.data["_parsed_start_date"].dt.tz is not None
    assert str(result.data["_parsed_start_date"].dt.tz) == "America/Los_Angeles"


def test_cleaning_retains_posttest_duplicate_rows_and_flags_ids():
    result = clean_qualtrics_csv(POSTTEST, "posttest")

    assert len(result.data) == 34
    assert result.data["_is_duplicate_participant_id"].sum() == 4
    assert len(result.duplicate_ids) == 2


def test_cutoff_is_timezone_aware():
    assert DEFAULT_CUTOFF == pd.Timestamp("2026-03-29 19:45:41", tz="America/Los_Angeles")


def test_date_range_filter_is_inclusive_for_calendar_days():
    result = clean_qualtrics_csv(PRETEST, "pretest")

    filtered = filter_by_date_range(
        result.data,
        start_date=pd.Timestamp("2026-09-19").date(),
        end_date=pd.Timestamp("2026-09-19").date(),
    )

    assert len(filtered) == 2
    assert filtered["_parsed_start_date"].dt.date.eq(pd.Timestamp("2026-09-19").date()).all()
