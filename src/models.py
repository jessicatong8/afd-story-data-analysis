from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CleaningResult:
    """Cleaned responses plus duplicate and duration-outlier review data."""

    data: pd.DataFrame
    source: str
    duplicate_ids: pd.DataFrame
    duration_outliers: pd.DataFrame
