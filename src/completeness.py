from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_completion_rules(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        rules = yaml.safe_load(file) or {}
    return rules


def _is_non_empty(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.notna()
    return values.notna() & values.astype("string").str.strip().ne("")


def _column_for_source(frame: pd.DataFrame, source: str, column: str) -> str:
    if column == "participant_id":
        return column
    source_column = f"{source}__{column}"
    if source_column in frame.columns:
        return source_column
    if column in frame.columns:
        return column
    raise ValueError(f"Completion rule references missing column: {source}.{column}")


def apply_completion_rules(frame: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
    """Add row-level completion flags from configurable required columns."""
    result = frame.copy()
    source_flags: list[str] = []
    for source in ("pretest", "posttest", "storybook"):
        source_rule = rules.get(source, {})
        required_columns = source_rule.get("required_columns", [])
        if source_rule.get("completion_mode", "all_non_empty") != "all_non_empty":
            raise ValueError(f"Unsupported completion mode for {source}")

        checks = []
        for column in required_columns:
            resolved_column = _column_for_source(result, source, column)
            checks.append(_is_non_empty(result[resolved_column]))
        flag_name = f"{source}_complete"
        result[flag_name] = (
            pd.concat(checks, axis=1).all(axis=1)
            if checks
            else pd.Series(False, index=result.index)
        )
        source_flags.append(flag_name)

    result["fully_completed"] = result[source_flags].all(axis=1)
    return result


def participant_completion_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse row-level results to one completion record per participant."""
    valid = frame[frame["participant_id"].notna() & frame["participant_id"].ne("")].copy()
    if valid.empty:
        return pd.DataFrame(columns=["participant_id", "fully_completed"])
    aggregations = {
        column: "max"
        for column in ["pretest_complete", "posttest_complete", "storybook_complete", "fully_completed"]
        if column in valid.columns
    }
    summary = valid.groupby("participant_id", as_index=False).agg(aggregations)
    for column in aggregations:
        summary[column] = summary[column].fillna(False).astype(bool)
    return summary
