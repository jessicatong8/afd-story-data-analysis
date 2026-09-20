from pathlib import Path

import pandas as pd

from src.qualtrics import DEFAULT_CUTOFF, clean_qualtrics_csv, load_qualtrics_csv

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
    assert len(result.excluded) == 10
    assert not result.data["_is_preview"].any()
    assert not result.data["_before_collection_cutoff"].any()
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
