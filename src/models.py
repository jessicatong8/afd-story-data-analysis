from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CleaningResult:
    """Cleaned eligible responses plus rows excluded during processing."""

    data: pd.DataFrame
    excluded: pd.DataFrame
    source: str
    duplicate_ids: pd.DataFrame
