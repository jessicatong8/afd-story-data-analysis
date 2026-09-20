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


def _resolve_referral_source(values: pd.Series) -> str:
    unique_values = sorted(set(values.dropna()))
    if not unique_values:
        return "Missing/Unknown"
    if len(unique_values) == 1:
        return unique_values[0]
    return "Multiple/Review"
