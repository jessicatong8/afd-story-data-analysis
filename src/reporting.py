from __future__ import annotations

import pandas as pd

from .completeness import participant_completion_summary

REFERRAL_COLUMN = "pretest__Referal Source"


def summary_metrics(joined: pd.DataFrame) -> dict[str, int]:
    participant_rows = participant_completion_summary(joined)
    return {
        "unique_participants": int(joined["participant_id"].dropna().replace("", pd.NA).nunique()),
        "pretest_participants": int(_unique_present(joined, "pretest_present")),
        "completed_pretest_participants": int(participant_rows["pretest_complete"].sum())
        if "pretest_complete" in participant_rows
        else 0,
        "posttest_participants": int(_unique_present(joined, "posttest_present")),
        "completed_posttest_participants": int(participant_rows["posttest_complete"].sum())
        if "posttest_complete" in participant_rows
        else 0,
        "storybook_participants": int(_unique_present(joined, "storybook_present")),
        "confirmed_storybook_participants": int(
            _unique_source_participants(joined, "supabase_confirmed")
        ),
        "inferred_storybook_participants": int(
            _unique_source_participants(joined, "inferred_from_posttest")
        ),
        "completed_storybook_participants": int(participant_rows["storybook_complete"].sum())
        if "storybook_complete" in participant_rows
        else 0,
        "fully_completed_participants": int(participant_rows["fully_completed"].sum())
        if "fully_completed" in participant_rows
        else 0,
        "duplicate_joined_rows": int(joined.get("posttest___is_duplicate_participant_id", pd.Series(dtype=bool)).sum())
        + int(joined.get("pretest___is_duplicate_participant_id", pd.Series(dtype=bool)).sum()),
    }


def _unique_present(frame: pd.DataFrame, presence_column: str) -> int:
    return int(frame.loc[frame[presence_column], "participant_id"].dropna().replace("", pd.NA).nunique())


def _unique_source_participants(frame: pd.DataFrame, source: str) -> int:
    if "storybook_completion_source" not in frame.columns:
        return 0
    return int(
        frame.loc[
            frame["storybook_completion_source"].eq(source), "participant_id"
        ]
        .dropna()
        .replace("", pd.NA)
        .nunique()
    )


def completed_participant_details(joined: pd.DataFrame) -> pd.DataFrame:
    """Return one demographic detail row per fully completed participant."""
    source_columns = {
        "start_date": "pretest__StartDate",
        "city": "pretest__City",
        "state": "pretest__State",
        "country": "pretest__Country",
        "child_age": "pretest__CAge",
        "parent_ethnicity": "pretest__PEthnicity",
        "child_ethnicity": "pretest__CEthnicity",
        "referral_source": REFERRAL_COLUMN,
    }
    output_columns = [
        "participant_id",
        "start_date",
        "location",
        "referral_source",
        "child_age",
        "parent_ethnicity",
        "child_ethnicity",
    ]
    required_columns = ["fully_completed", "participant_id", *source_columns.values()]
    if not all(column in joined.columns for column in required_columns):
        return pd.DataFrame(columns=output_columns)

    completed = joined.loc[
        joined["fully_completed"]
        & joined["participant_id"].notna()
        & joined["participant_id"].ne(""),
        required_columns,
    ].copy()
    if completed.empty:
        return pd.DataFrame(columns=output_columns)

    completed = completed.rename(
        columns={value: key for key, value in source_columns.items()}
    )
    completed["start_date"] = pd.to_datetime(
        completed["start_date"], errors="coerce", format="mixed"
    )
    completed = completed.sort_values(
        "start_date", ascending=True, na_position="first"
    )
    completed = completed.drop_duplicates("participant_id", keep="last")

    def clean_value(value: object, fallback: str = "Missing/Unknown") -> str:
        if pd.isna(value) or str(value).strip() == "":
            return fallback
        return str(value).strip()

    completed["location"] = completed.apply(
        lambda row: ", ".join(
            value
            for value in [
                clean_value(row["city"], ""),
                clean_value(row["state"], ""),
                clean_value(row["country"], ""),
            ]
            if value
        )
        or "Missing/Unknown",
        axis=1,
    )
    for column in [
        "referral_source",
        "child_age",
        "parent_ethnicity",
        "child_ethnicity",
    ]:
        completed[column] = completed[column].map(clean_value)

    return completed.sort_values(
        "start_date", ascending=False, na_position="last"
    )[output_columns].reset_index(drop=True)


def completed_participant_times(
    joined: pd.DataFrame,
    excluded_participant_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Return duration breakdowns and total time for fully completed participants."""
    duration_columns = {
        "pretest_duration_sec": "pretest__Duration (in seconds)",
        "storybook_book_time_sec": "storybook__book_time_sec",
        "storybook_game_time_sec": "storybook__game_time_sec",
        "posttest_duration_sec": "posttest__Duration (in seconds)",
    }
    output_columns = [
        "participant_id",
        *duration_columns.keys(),
        "total_study_time_sec",
    ]
    if "fully_completed" not in joined.columns or "participant_id" not in joined.columns:
        return pd.DataFrame(columns=output_columns)

    if not all(column in joined.columns for column in duration_columns.values()):
        return pd.DataFrame(columns=output_columns)

    completed = joined.loc[
        joined["fully_completed"]
        & joined["participant_id"].notna()
        & joined["participant_id"].ne(""),
        ["participant_id", *duration_columns.values()],
    ].copy()
    if completed.empty:
        return pd.DataFrame(columns=output_columns)

    if excluded_participant_ids:
        completed = completed.loc[
            ~completed["participant_id"].isin(excluded_participant_ids)
        ].copy()
        if completed.empty:
            return pd.DataFrame(columns=output_columns)

    completed = completed.rename(
        columns={column: output for output, column in duration_columns.items()}
    )
    for column in duration_columns:
        completed[column] = pd.to_numeric(completed[column], errors="coerce")

    completed = completed.groupby("participant_id", as_index=False).max(numeric_only=True)
    duration_output_columns = list(duration_columns)
    completed["total_study_time_sec"] = completed[duration_output_columns].sum(
        axis=1, min_count=len(duration_output_columns)
    )
    return completed[output_columns].sort_values("participant_id").reset_index(drop=True)


def format_duration(seconds: float | int | None) -> str:
    """Format a duration as whole minutes and seconds."""
    if seconds is None or pd.isna(seconds):
        return "N/A"
    total_seconds = max(0, int(round(float(seconds))))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    return f"{minutes}:{remaining_seconds:02d}"


def average_total_study_time(times: pd.DataFrame) -> float | None:
    if "total_study_time_sec" not in times.columns:
        return None
    values = pd.to_numeric(times["total_study_time_sec"], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def total_study_time_summary(times: pd.DataFrame) -> dict[str, float | None]:
    """Return average, mean, median, and maximum total study time in seconds."""
    if "total_study_time_sec" not in times.columns:
        values = pd.Series(dtype="float64")
    else:
        values = pd.to_numeric(times["total_study_time_sec"], errors="coerce").dropna()
    if values.empty:
        return {"average": None, "mean": None, "median": None, "max": None}
    return {
        "average": float(values.mean()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def duration_summary(times: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Return mean and sample standard deviation for each study duration."""
    duration_columns = {
        "Pre-test": "pretest_duration_sec",
        "Book": "storybook_book_time_sec",
        "Game": "storybook_game_time_sec",
        "Post-test": "posttest_duration_sec",
        "Total study": "total_study_time_sec",
    }
    summary = {}
    for label, column in duration_columns.items():
        values = pd.to_numeric(
            times[column] if column in times.columns else pd.Series(dtype="float64"),
            errors="coerce",
        ).dropna()
        summary[label] = {
            "mean": float(values.mean()) if not values.empty else None,
            "sd": float(values.std()) if len(values) > 1 else None,
        }
    return summary


def duration_distribution(
    times: pd.DataFrame, duration_column: str, bin_seconds: int = 300
) -> pd.DataFrame:
    """Build five-minute duration bins for a Streamlit bar chart."""
    if duration_column not in times.columns:
        return pd.DataFrame(columns=["time_range", "participants"])
    values = pd.to_numeric(times[duration_column], errors="coerce").dropna()
    if values.empty:
        return pd.DataFrame(columns=["time_range", "participants"])

    max_seconds = max(float(values.max()), 0)
    upper_bound = (int(max_seconds // bin_seconds) + 1) * bin_seconds
    if upper_bound == 0:
        upper_bound = bin_seconds
    edges = list(range(0, upper_bound + bin_seconds, bin_seconds))
    buckets = pd.cut(values, bins=edges, right=False, include_lowest=True)
    counts = buckets.value_counts(sort=False)
    distribution = pd.DataFrame(
        {
            "bucket_start": [interval.left for interval in counts.index],
            "bucket_end": [interval.right for interval in counts.index],
            "participants": counts.to_numpy(),
        }
    ).sort_values("bucket_start", kind="stable")
    distribution["time_range"] = distribution.apply(
        lambda row: (
            f"{format_duration(row['bucket_start'])}-"
            f"{format_duration(row['bucket_end'] - 1)}"
        ),
        axis=1,
    )
    return distribution[["time_range", "participants"]].reset_index(drop=True)


def _resolve_referral_source(values: pd.Series) -> str:
    unique_values = sorted(set(values.dropna()))
    if not unique_values:
        return "Missing/Unknown"
    if len(unique_values) == 1:
        return unique_values[0]
    return "Multiple/Review"
