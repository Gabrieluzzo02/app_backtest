import pandas as pd


def apply_coefficients_last_row(df: pd.DataFrame, alpha: float, coeff: pd.Series) -> float:
    """Apply alpha + beta coefficients to the last row of a DataFrame."""
    x_last = df.iloc[-1].reindex(coeff.index)
    return float(alpha + (x_last * coeff).sum())
