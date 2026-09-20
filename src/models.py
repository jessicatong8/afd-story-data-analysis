from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CleaningResult:
    """Cleaned eligible responses and duplicate IDs requiring review."""

    data: pd.DataFrame
    source: str
    duplicate_ids: pd.DataFrame
