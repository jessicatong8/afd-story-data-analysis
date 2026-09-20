from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from src.completeness import apply_completion_rules, load_completion_rules
from src.joining import join_sources
from src.qualtrics import DEFAULT_CUTOFF, clean_qualtrics_csv
from src.reporting import referral_sources_for_completed, summary_metrics
from src.supabase_client import create_supabase_client, fetch_participants

load_dotenv()

RULES_PATH = Path(__file__).parent / "config" / "completion_rules.yaml"


def _secret(name: str) -> str | None:
    try:
        return st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        return None

st.set_page_config(page_title="AFD Story Data Analysis", layout="wide")
st.title("AFD Story Data Analysis")
st.caption("Process Qualtrics surveys and participant storybook metrics in memory.")

with st.sidebar:
    st.header("Data sources")
    pretest_upload = st.file_uploader("Pre-test Qualtrics CSV", type="csv")
    posttest_upload = st.file_uploader("Post-test Qualtrics CSV", type="csv")
    st.caption(f"Responses before {DEFAULT_CUTOFF} are excluded.")

if not pretest_upload or not posttest_upload:
    st.info("Upload both Qualtrics CSV files to begin.")
    st.stop()

try:
    pretest_result = clean_qualtrics_csv(pretest_upload, "pretest")
    posttest_result = clean_qualtrics_csv(posttest_upload, "posttest")
except (ValueError, TypeError) as error:
    st.error(f"Could not process the Qualtrics files: {error}")
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
    storybook = pd.DataFrame(columns=["participant_id", "book_completed"])

joined = join_sources(pretest_result.data, posttest_result.data, storybook)
try:
    joined = apply_completion_rules(joined, load_completion_rules(RULES_PATH))
except ValueError as error:
    st.error(f"Completion configuration error: {error}")
    st.stop()

metrics = summary_metrics(joined)
metric_columns = st.columns(5)
for column, (label, value) in zip(
    metric_columns,
    [
        ("Unique participants", metrics["unique_participants"]),
        ("Pre-test participants", metrics["pretest_participants"]),
        ("Post-test participants", metrics["posttest_participants"]),
        ("Storybook participants", metrics["storybook_participants"]),
        ("Fully completed", metrics["fully_completed_participants"]),
    ],
):
    column.metric(label, value)

st.subheader("Data quality")
quality_columns = st.columns(4)
quality_columns[0].metric("Pre-test excluded", len(pretest_result.excluded))
quality_columns[1].metric("Post-test excluded", len(posttest_result.excluded))
quality_columns[2].metric("Pre-test duplicate IDs", len(pretest_result.duplicate_ids))
quality_columns[3].metric("Post-test duplicate IDs", len(posttest_result.duplicate_ids))

st.subheader("Referral sources for fully completed participants")
referral_table = referral_sources_for_completed(joined)
if referral_table.empty:
    st.info("No fully completed participants match the current completion rules.")
else:
    st.dataframe(
        referral_table.style.format({"percentage": "{:.1f}%"}),
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

excluded = pd.concat(
    [pretest_result.excluded.assign(source="pretest"), posttest_result.excluded.assign(source="posttest")],
    ignore_index=True,
)
st.download_button(
    "Download filtering audit",
    data=excluded.to_csv(index=False).encode("utf-8"),
    file_name="afd_story_filtering_audit.csv",
    mime="text/csv",
)
