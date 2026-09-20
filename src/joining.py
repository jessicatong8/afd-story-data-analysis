from __future__ import annotations

from functools import reduce

import pandas as pd

SOURCE_NAMES = ("pretest", "posttest", "storybook")


def _prepare_source(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    prepared = frame.copy()
    if "participant_id" not in prepared.columns:
        raise ValueError(f"{source} data is missing participant_id")

    prepared["participant_id"] = prepared["participant_id"].astype("string").str.strip()
    prepared["__join_key"] = prepared["participant_id"]
    missing = prepared["__join_key"].isna() | prepared["__join_key"].eq("")
    prepared.loc[missing, "__join_key"] = [
        f"__missing_{source}_{index}" for index in prepared.index[missing]
    ]
    prepared[f"__{source}_present"] = True

    rename = {
        column: f"{source}__{column}"
        for column in prepared.columns
        if column not in {"__join_key", f"__{source}_present"}
    }
    return prepared.rename(columns=rename)


def _coalesce_participant_id(joined: pd.DataFrame) -> pd.Series:
    id_columns = [
        column
        for column in joined.columns
        if column == "participant_id" or column.endswith("__participant_id")
    ]
    result = pd.Series(pd.NA, index=joined.index, dtype="string")
    for column in id_columns:
        values = joined[column].astype("string").str.strip()
        result = result.fillna(values.mask(values.eq("")))
    return result


def join_sources(
    pretest: pd.DataFrame,
    posttest: pd.DataFrame,
    storybook: pd.DataFrame,
) -> pd.DataFrame:
    """Join sources while retaining survey rows and matched storybook metrics."""
    pretest_ids = (
        pretest["participant_id"].astype("string").str.strip()
        if "participant_id" in pretest.columns
        else pd.Series(dtype="string")
    )
    valid_pretest_ids = set(pretest_ids[pretest_ids.notna() & pretest_ids.ne("")])
    storybook_ids = storybook["participant_id"].astype("string").str.strip()
    storybook = storybook.loc[
        storybook_ids.isin(valid_pretest_ids)
    ].copy()

    sources = {
        "pretest": _prepare_source(pretest, "pretest"),
        "storybook": _prepare_source(storybook, "storybook"),
        "posttest": _prepare_source(posttest, "posttest"),
    }
    joined = reduce(
        lambda left, right: left.merge(right, on="__join_key", how="outer"),
        sources.values(),
    )
    joined["participant_id"] = _coalesce_participant_id(joined)
    joined = joined.drop(columns=["__join_key"], errors="ignore")

    for source in SOURCE_NAMES:
        presence_column = f"__{source}_present"
        joined[f"{source}_present"] = joined.get(presence_column, False).fillna(False).astype(bool)
        joined = joined.drop(columns=[presence_column], errors="ignore")

    presence = joined[[f"{source}_present" for source in SOURCE_NAMES]]
    joined["join_status"] = presence.apply(_join_status, axis=1)
    return joined


def _join_status(row: pd.Series) -> str:
    present = {source for source in SOURCE_NAMES if row[f"{source}_present"]}
    if len(present) == len(SOURCE_NAMES):
        return "matched_all_sources"
    missing = [source for source in SOURCE_NAMES if source not in present]
    if len(missing) > 1:
        return "missing_multiple_sources"
    return "missing_" + "_and_".join(missing)
