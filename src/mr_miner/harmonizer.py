import pandas as pd
from pathlib import Path


class Harmonizer:
    def __init__(
        self,
        cache_dir: Path | str | None = None,
        tolerance: float = 0.08,
        action: int = 2,
    ):
        """
        Initialize harmonizer

        Args:
            cache_dir: Optional directory to cache harmonized data
            tolerance: Allele frequency matching tolerance
            action: Harmonization action level (1-3)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.tolerance = tolerance
        self.action = action

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_harmonized_data(
        self, harmonized_data: pd.DataFrame, output_fpath: Path | str
    ) -> None:
        """
        Save harmonized data to file

        Args:
            harmonized_data: DataFrame with harmonized data
            output_fpath: Path to save harmonized data
        """
        harmonized_data.to_csv(output_fpath, sep="\t", index=False)
