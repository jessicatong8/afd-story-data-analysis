from __future__ import annotations

import os
from typing import Any

import pandas as pd
from supabase import Client, create_client

TABLE_NAME = "participants"
TABLE_COLUMNS = [
    "participant_id",
    "game_score",
    "book_time_sec",
    "book_completed",
    "game_time_sec",
    "game_completed",
    "created_at",
    "updated_at",
]


def create_supabase_client(url: str | None = None, key: str | None = None) -> Client:
    """Create a Supabase client from explicit values or environment variables."""
    resolved_url = url or os.getenv("SUPABASE_URL")
    resolved_key = key or os.getenv("SUPABASE_KEY")
    if not resolved_url or not resolved_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be configured")
    return create_client(resolved_url, resolved_key)


def fetch_participants(client: Client) -> pd.DataFrame:
    """Fetch the participant-level story metrics without persisting them."""
    response = client.table(TABLE_NAME).select(",".join(TABLE_COLUMNS)).execute()
    frame = pd.DataFrame(response.data)
    if frame.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)

    frame["participant_id"] = frame["participant_id"].astype("string").str.strip()
    duplicate_ids = frame["participant_id"].duplicated(keep=False)
    if duplicate_ids.any():
        duplicate_values = frame.loc[duplicate_ids, "participant_id"].dropna().unique()
        raise ValueError(
            "Supabase participants table must contain one row per participant; "
            f"duplicate IDs found: {list(duplicate_values)}"
        )
    return frame


def fetch_participants_from_env() -> pd.DataFrame:
    return fetch_participants(create_supabase_client())
