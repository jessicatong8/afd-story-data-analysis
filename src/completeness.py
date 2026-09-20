from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .joining import refresh_join_status

PRETEST_SECTION_LABELS = {
    "demographics": "Demographics",
    "cprs_scale": "CPRS Scale",
    "sdq_scale": "SDQ Scale",
    "parent_afd_scale": "Parent AFD Scale",
    "parent_love_general": "Parent Love General",
    "parent_love_language_scale": "Parent Love Language Scale",
    "child_afd_scale": "Child AFD Scale",
    "child_love_general": "Child Love General",
    "child_love_language_scale": "Child Love Language Scale",
}

POSTTEST_SECTION_LABELS = {
    "child_program_eval": "Child Program Eval",
    "child_learning": "Child Learning",
    "child_love_language_change": "Child Love Language Change",
    "parent_program_eval": "Parent Program Eval",
    "parent_learning": "Parent Learning",
}


def load_completion_rules(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        rules = yaml.safe_load(file) or {}
    return rules


def load_storybook_overrides(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config.get("overrides", [])


def _is_non_empty(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.notna()
    return values.notna() & values.astype("string").str.strip().ne("")


def _is_true(values: pd.Series) -> pd.Series:
    return values.astype("boolean").fillna(False).eq(True)


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
    """Add row-level completion flags and missing-section details."""
    result = frame.copy()
    source_flags: list[str] = []
    for source in ("pretest", "posttest", "storybook"):
        source_rule = rules.get(source, {})
        if source_rule.get("completion_mode", "all_non_empty") != "all_non_empty":
            raise ValueError(f"Unsupported completion mode for {source}")

        sections = source_rule.get("sections")
        if sections is None:
            sections = {"overall": source_rule.get("required_columns", [])}

        section_flags: list[str] = []
        for section, required_columns in sections.items():
            checks = []
            for column in required_columns:
                resolved_column = _column_for_source(result, source, column)
                checks.append(_is_non_empty(result[resolved_column]))
            if section == "overall":
                for column in source_rule.get("required_true_columns", []):
                    resolved_column = _column_for_source(result, source, column)
                    checks.append(_is_true(result[resolved_column]))
            flag_name = f"{source}_{section}_complete"
            result[flag_name] = (
                pd.concat(checks, axis=1).all(axis=1)
                if checks
                else pd.Series(False, index=result.index)
            )
            section_flags.append(flag_name)

        flag_name = f"{source}_complete"
        result[flag_name] = result[section_flags].all(axis=1)
        source_flags.append(flag_name)

        if source in {"pretest", "posttest"}:
            section_names = list(sections)
            labels = (
                PRETEST_SECTION_LABELS
                if source == "pretest"
                else POSTTEST_SECTION_LABELS
            )
            result[f"{source}_missing_sections"] = result.apply(
                lambda row: ", ".join(
                    labels.get(
                        section, section.replace("_", " ").title()
                    )
                    for section, section_flag in zip(section_names, section_flags)
                    if not row[section_flag]
                )
                or pd.NA,
                axis=1,
            )

    result["fully_completed"] = result[source_flags].all(axis=1)
    return result


def apply_storybook_overrides(
    frame: pd.DataFrame, overrides: list[dict[str, str]]
) -> pd.DataFrame:
    """Infer storybook completion for approved post-test-based overrides."""
    result = frame.copy()
    result["storybook_completion_source"] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )
    result["storybook_completion_reason"] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )
    result["storybook_metrics_available"] = result["storybook_present"].astype(bool)
    result.loc[result["storybook_present"], "storybook_completion_source"] = (
        "supabase_confirmed"
    )

    if not overrides:
        return refresh_join_status(result)

    for override in overrides:
        participant_id = str(override["participant_id"]).strip()
        eligible = (
            result["participant_id"].astype("string").eq(participant_id)
            & result["posttest_complete"].astype(bool)
            & ~result["storybook_present"].astype(bool)
        )
        result.loc[eligible, "storybook__book_completed"] = True
        result.loc[eligible, "storybook__game_completed"] = True
        result.loc[eligible, "storybook_present"] = True
        result.loc[eligible, "storybook_completion_source"] = "inferred_from_posttest"
        result.loc[eligible, "storybook_completion_reason"] = override.get("reason", "")
        result.loc[eligible, "storybook_metrics_available"] = False

    return refresh_join_status(result)


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
