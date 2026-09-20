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
        "storybook_participants": int(_unique_present(joined, "storybook_present")),
        "fully_completed_participants": int(participant_rows["fully_completed"].sum())
        if "fully_completed" in participant_rows
        else 0,
        "duplicate_joined_rows": int(joined.get("posttest___is_duplicate_participant_id", pd.Series(dtype=bool)).sum())
        + int(joined.get("pretest___is_duplicate_participant_id", pd.Series(dtype=bool)).sum()),
    }


def _unique_present(frame: pd.DataFrame, presence_column: str) -> int:
    return int(frame.loc[frame[presence_column], "participant_id"].dropna().replace("", pd.NA).nunique())


def referral_sources_for_completed(joined: pd.DataFrame) -> pd.DataFrame:
    """Count one referral source per fully completed participant."""
    if REFERRAL_COLUMN not in joined.columns:
        return pd.DataFrame(columns=["referral_source", "participants", "percentage"])

    completed = joined.loc[
        joined["fully_completed"]
        & joined["participant_id"].notna()
        & joined["participant_id"].ne(""),
        ["participant_id", REFERRAL_COLUMN],
    ].copy()
    if completed.empty:
        return pd.DataFrame(columns=["referral_source", "participants", "percentage"])

    completed["referral_source"] = (
        completed[REFERRAL_COLUMN].astype("string").str.strip().replace("", pd.NA).fillna("Missing/Unknown")
    )
    per_participant = completed.groupby("participant_id", as_index=False).agg(
        referral_source=("referral_source", _resolve_referral_source)
    )
    counts = per_participant["referral_source"].value_counts().rename_axis("referral_source").reset_index(name="participants")
    counts["percentage"] = counts["participants"] / len(per_participant) * 100
    return counts


def _resolve_referral_source(values: pd.Series) -> str:
    unique_values = sorted(set(values.dropna()))
    if not unique_values:
        return "Missing/Unknown"
    if len(unique_values) == 1:
        return unique_values[0]
    return "Multiple/Review"
