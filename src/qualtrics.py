from __future__ import annotations

from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO, TextIO

import pandas as pd

from .models import CleaningResult

DEFAULT_TIMEZONE = "America/Los_Angeles"
DEFAULT_CUTOFF = pd.Timestamp("2026-03-29 19:45:41", tz=DEFAULT_TIMEZONE)


def _read_csv_source(source: BinaryIO | TextIO | str | Path) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        return pd.read_csv(source, dtype="string", keep_default_na=False)
    raw = source.read()
    if isinstance(raw, str):
        return pd.read_csv(StringIO(raw), dtype="string", keep_default_na=False)
    return pd.read_csv(BytesIO(raw), dtype="string", keep_default_na=False)


def _column_lookup(columns: pd.Index) -> dict[str, str]:
    return {str(column).strip().casefold(): str(column) for column in columns}


def _find_column(frame: pd.DataFrame, name: str) -> str:
    column = _column_lookup(frame.columns).get(name.casefold())
    if column is None:
        raise ValueError(f"Qualtrics file is missing required column: {name}")
    return column


def _is_metadata_row(value: object) -> bool:
    text = str(value).strip()
    return text == "Start Date" or text.startswith('{"ImportId"')


def _parse_start_dates(values: pd.Series, timezone: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(timezone)
    return parsed.dt.tz_convert(timezone)


def _as_cutoff(value: datetime | pd.Timestamp | str, timezone: str) -> pd.Timestamp:
    cutoff = pd.Timestamp(value)
    if cutoff.tzinfo is None:
        return cutoff.tz_localize(timezone)
    return cutoff.tz_convert(timezone)


def load_qualtrics_csv(source: BinaryIO | TextIO | str | Path) -> pd.DataFrame:
    """Load a Qualtrics export and remove its question and ImportId rows."""
    frame = _read_csv_source(source)
    start_column = _find_column(frame, "StartDate")
    metadata_mask = frame[start_column].map(_is_metadata_row)
    return frame.loc[~metadata_mask].reset_index(drop=True)


def clean_qualtrics_csv(
    source: BinaryIO | TextIO | str | Path,
    source_name: str,
    cutoff: datetime | pd.Timestamp | str = DEFAULT_CUTOFF,
    timezone: str = DEFAULT_TIMEZONE,
) -> CleaningResult:
    """Filter a Qualtrics export while retaining audit information."""
    frame = load_qualtrics_csv(source).copy()
    status_column = _find_column(frame, "status")
    start_column = _find_column(frame, "StartDate")
    participant_column = _find_column(frame, "participant_id")

    frame[participant_column] = frame[participant_column].astype("string").str.strip()
    frame["participant_id"] = frame[participant_column]
    frame["_source"] = source_name
    frame["_parsed_start_date"] = _parse_start_dates(frame[start_column], timezone)
    is_preview = (
        frame[status_column].astype("string").str.strip().str.casefold()
        == "survey preview"
    )
    before_collection_cutoff = frame["_parsed_start_date"] < _as_cutoff(
        cutoff, timezone
    )
    duplicate_mask = frame["participant_id"].ne("") & frame["participant_id"].notna()
    duplicate_mask &= frame["participant_id"].duplicated(keep=False)
    frame["_is_duplicate_participant_id"] = duplicate_mask

    eligible_mask = ~is_preview & ~before_collection_cutoff
    data = frame.loc[eligible_mask].copy()
    duplicate_ids = (
        data.loc[data["_is_duplicate_participant_id"], ["participant_id"]]
        .drop_duplicates()
        .sort_values("participant_id")
        .reset_index(drop=True)
    )

    return CleaningResult(
        data=data.reset_index(drop=True),
        source=source_name,
        duplicate_ids=duplicate_ids,
    )
