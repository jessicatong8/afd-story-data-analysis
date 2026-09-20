from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from src.completeness import (
    apply_completion_rules,
    apply_storybook_overrides,
    load_completion_rules,
    load_storybook_overrides,
)
from src.joining import join_sources
from src.qualtrics import (
    DEFAULT_CUTOFF,
    DEFAULT_TIMEZONE,
    clean_qualtrics_csv,
    filter_cleaning_result_by_date_range,
)
from src.reporting import completed_participant_details, summary_metrics
from src.supabase_client import create_supabase_client, fetch_participants

load_dotenv()

RULES_PATH = Path(__file__).parent / "config" / "completion_rules.yaml"
STORYBOOK_OVERRIDES_PATH = Path(__file__).parent / "config" / "storybook_overrides.yaml"


def _secret(name: str) -> str | None:
    try:
        return st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        return None

st.set_page_config(page_title="AFD Storybook Study Data Analysis", layout="wide")
st.title("AFD Storybook Study Data Analysis")
st.caption("Cleans and filters Qualtrics survey data and storybook metrics and joins by participant ID. Completion is defined by valid responses to key survey questions (essential demographics, completion of scales) and book and game completion. Filter by date range to see only responses collected during a specific time period. Download the joined dataset or the fully completed dataset for further analysis.")

with st.sidebar:
    st.header("Data sources")
    pretest_upload = st.file_uploader("Pre-test Qualtrics CSV", type="csv")
    posttest_upload = st.file_uploader("Post-test Qualtrics CSV", type="csv")
    use_date_slice = st.checkbox("Filter by date range", value=False)
    st.caption(f"Responses before the start of data collection on {DEFAULT_CUTOFF} are excluded.")

if not pretest_upload or not posttest_upload:
    st.info("Upload both Qualtrics CSV files to begin.")
    st.stop()

try:
    pretest_result = clean_qualtrics_csv(pretest_upload, "pretest")
    posttest_result = clean_qualtrics_csv(posttest_upload, "posttest")
except (ValueError, TypeError) as error:
    st.error(f"Could not process the Qualtrics files: {error}")
    st.stop()

if use_date_slice:
    available_dates = pd.concat(
        [
            pretest_result.data["_parsed_start_date"],
            posttest_result.data["_parsed_start_date"],
        ]
    ).dropna()
    min_available_date = available_dates.min().date()
    max_available_date = available_dates.max().date()
    with st.sidebar:
        selected_start_date = st.date_input(
            "Start date",
            value=min_available_date,
            min_value=min_available_date,
            max_value=max_available_date,
        )
        selected_end_date = st.date_input(
            "End date",
            value=max_available_date,
            min_value=min_available_date,
            max_value=max_available_date,
        )
    try:
        pretest_result = filter_cleaning_result_by_date_range(
            pretest_result,
            selected_start_date,
            selected_end_date,
            DEFAULT_TIMEZONE,
        )
        posttest_result = filter_cleaning_result_by_date_range(
            posttest_result,
            selected_start_date,
            selected_end_date,
            DEFAULT_TIMEZONE,
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()

supabase_url = os.getenv("SUPABASE_URL") or _secret("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY") or _secret("SUPABASE_KEY")
if supabase_url and supabase_key:
    try:
        storybook = fetch_participants(create_supabase_client(supabase_url, supabase_key))
    except Exception as error:
        st.error(f"Could not load Supabase participant metrics: {error}")
        st.stop()
else:
    st.warning("Supabase credentials are not configured. Storybook metrics are currently empty.")
    storybook = pd.DataFrame(
        columns=["participant_id", "book_completed", "game_completed"]
    )

joined = join_sources(pretest_result.data, posttest_result.data, storybook)
completion_rules = load_completion_rules(RULES_PATH)
try:
    joined = apply_completion_rules(joined, completion_rules)
    joined = apply_storybook_overrides(
        joined, load_storybook_overrides(STORYBOOK_OVERRIDES_PATH)
    )
    joined = apply_completion_rules(joined, completion_rules)
except ValueError as error:
    st.error(f"Completion configuration error: {error}")
    st.stop()

metrics = summary_metrics(joined)
metric_columns = st.columns(8)
for column, (label, value) in zip(
    metric_columns,
    [
        # ("Unique participants", metrics["unique_participants"]),
        ("Pre-test participants", metrics["pretest_participants"]),
        ("Completed pre-test", metrics["completed_pretest_participants"]),
        # ("Post-test participants", metrics["posttest_participants"]),
        ("Completed post-test", metrics["completed_posttest_participants"]),
        # ("Storybook participants", metrics["storybook_participants"]),
        ("Completed storybook", metrics["completed_storybook_participants"]),
        ("Fully completed", metrics["fully_completed_participants"]),
    ],
):
    column.metric(label, value)

st.subheader("Data quality")
quality_columns = st.columns(4)
quality_columns[0].metric("Pre-test duplicate IDs", len(pretest_result.duplicate_ids))
quality_columns[1].metric("Post-test duplicate IDs", len(posttest_result.duplicate_ids))
quality_columns[2].metric(
    "Supabase storybook", metrics["confirmed_storybook_participants"]
)
quality_columns[3].metric(
    "Inferred storybook", metrics["inferred_storybook_participants"]
)

st.subheader("Completed participant details")
participant_details = completed_participant_details(joined)
if participant_details.empty:
    st.info("No fully completed participants match the current completion rules.")
else:
    st.dataframe(
        participant_details,
        width="stretch",
        hide_index=True,
    )

with st.expander("Duplicate participant IDs"):
    duplicates = pd.concat(
        [
            pretest_result.duplicate_ids.assign(source="pretest"),
            posttest_result.duplicate_ids.assign(source="posttest"),
        ],
        ignore_index=True,
    )
    st.dataframe(duplicates, width="stretch", hide_index=True)

with st.expander("Joined dataset preview"):
    st.dataframe(joined, width="stretch", hide_index=True)

st.download_button(
    "Download joined dataset",
    data=joined.to_csv(index=False).encode("utf-8"),
    file_name="afd_story_joined_dataset.csv",
    mime="text/csv",
)

fully_completed = joined.loc[joined["fully_completed"]].copy()

with st.expander("Fully completed dataset preview"):
    st.dataframe(fully_completed, width="stretch", hide_index=True)

st.download_button(
    "Download fully completed dataset",
    data=fully_completed.to_csv(index=False).encode("utf-8"),
    file_name="afd_story_fully_completed_dataset.csv",
    mime="text/csv",
)
