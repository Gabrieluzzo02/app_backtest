import pandas as pd


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple percentage returns from prices."""
    return prices.pct_change().dropna(how="all")
