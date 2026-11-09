import pandas as pd
import pytest

from src.completeness import participant_completion_summary
from src.reporting import (
    average_total_study_time,
    completed_participant_details,
    completed_participant_times,
    duration_summary,
    duration_distribution,
    format_duration,
    total_study_time_summary,
)

def test_completed_participant_details_are_unique_and_sorted_by_date():
    joined = pd.DataFrame(
        {
            "participant_id": ["one", "one", "two", "three"],
            "fully_completed": [True, True, True, False],
            "pretest__StartDate": [
                "2026-04-01 10:00:00",
                "2026-04-01 10:00:00",
                "2026-05-01 10:00:00",
                "2026-06-01 10:00:00",
            ],
            "pretest__City": ["City A", "City A", "City B", "City C"],
            "pretest__State": ["State A", "State A", "State B", "State C"],
            "pretest__Country": ["US", "US", "US", "US"],
            "pretest__CAge": ["8", "8", "9", "10"],
            "pretest__PEthnicity": ["A", "A", "B", "C"],
            "pretest__CEthnicity": ["D", "D", "E", "F"],
            "pretest__Referal Source": ["School", "School", "Friend", "Social media"],
        }
    )

    result = completed_participant_details(joined)

    assert result["participant_id"].tolist() == ["two", "one"]
    assert result["referral_source"].tolist() == ["Friend", "School"]
    assert result["location"].tolist() == ["City B, State B, US", "City A, State A, US"]


def test_completed_participant_times_sums_all_study_components():
    joined = pd.DataFrame(
        {
            "participant_id": ["one", "two"],
            "fully_completed": [True, False],
            "pretest__Duration (in seconds)": [120, 90],
            "storybook__book_time_sec": [300, 200],
            "storybook__game_time_sec": [180, 100],
            "posttest__Duration (in seconds)": [240, 150],
        }
    )

    result = completed_participant_times(joined)

    assert result["participant_id"].tolist() == ["one"]
    assert result.loc[0, "total_study_time_sec"] == 840


def test_completed_participant_times_excludes_duration_outlier_ids_only():
    joined = pd.DataFrame(
        {
            "participant_id": ["one", "two"],
            "fully_completed": [True, True],
            "pretest__Duration (in seconds)": [120, 9999],
            "storybook__book_time_sec": [300, 300],
            "storybook__game_time_sec": [180, 180],
            "posttest__Duration (in seconds)": [240, 240],
        }
    )

    result = completed_participant_times(joined, {"two"})

    assert result["participant_id"].tolist() == ["one"]


def test_duration_outlier_does_not_change_fully_completed_status():
    joined = pd.DataFrame(
        {
            "participant_id": ["outlier"],
            "pretest_complete": [True],
            "posttest_complete": [True],
            "storybook_complete": [True],
            "fully_completed": [True],
            "pretest__Duration (in seconds)": [5000],
            "storybook__book_time_sec": [300],
            "storybook__game_time_sec": [180],
            "posttest__Duration (in seconds)": [240],
        }
    )

    summary = participant_completion_summary(joined)
    timing = completed_participant_times(joined, {"outlier"})

    assert summary.loc[0, "fully_completed"]
    assert timing.empty


def test_duration_visualization_helpers_format_average_and_bins():
    times = pd.DataFrame({"total_study_time_sec": [840, 960], "part": [120, 360]})

    assert format_duration(840) == "14:00"
    assert format_duration(65) == "1:05"
    assert average_total_study_time(times) == 900
    assert total_study_time_summary(times) == {
        "average": 900.0,
        "mean": 900.0,
        "median": 900.0,
        "max": 960.0,
    }
    distribution = duration_distribution(times, "part")
    assert distribution["time_range"].tolist() == ["0:00-4:59", "5:00-9:59"]
    assert distribution["participants"].tolist() == [1, 1]


def test_duration_summary_reports_means_and_sample_standard_deviations():
    times = pd.DataFrame(
        {
            "pretest_duration_sec": [60, 180],
            "storybook_book_time_sec": [300, 600],
            "storybook_game_time_sec": [120, 240],
            "posttest_duration_sec": [60, 120],
            "total_study_time_sec": [540, 1140],
        }
    )

    summary = duration_summary(times)

    assert list(summary) == ["Pre-test", "Book", "Game", "Post-test", "Total study"]
    assert summary["Pre-test"]["mean"] == 120
    assert summary["Pre-test"]["sd"] == pytest.approx(84.8528)
    assert summary["Total study"]["mean"] == 840
    assert summary["Total study"]["sd"] == pytest.approx(424.2641)


def test_duration_distribution_is_sorted_by_numeric_bucket_start():
    times = pd.DataFrame({"part": [610, 10, 305, 1_200]})

    result = duration_distribution(times, "part")

    assert result["time_range"].tolist() == [
        "0:00-4:59",
        "5:00-9:59",
        "10:00-14:59",
        "15:00-19:59",
        "20:00-24:59",
    ]
