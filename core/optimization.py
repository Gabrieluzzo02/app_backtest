import numpy as np
import pandas as pd
from scipy.optimize import minimize


def markowitz_min_variance(
    returns: pd.DataFrame,
    annualization_factor: int = 252,
    allow_short: bool = False,
) -> pd.Series:
    """Return Markowitz minimum-variance portfolio weights."""
    clean_returns = returns.dropna(how="any")
    if clean_returns.empty:
        raise ValueError("Returns DataFrame is empty after dropping NaN values.")

    cov_matrix = clean_returns.cov() * annualization_factor
    n_assets = len(cov_matrix.columns)

    if n_assets == 0:
        raise ValueError("No assets available for optimization.")

    def portfolio_volatility(weights: np.ndarray) -> float:
        return float(np.sqrt(weights.T @ cov_matrix.values @ weights))

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = None if allow_short else tuple((0.0, 1.0) for _ in range(n_assets))
    initial_weights = np.ones(n_assets) / n_assets

    result = minimize(
        portfolio_volatility,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return pd.Series(result.x, index=clean_returns.columns, name="weight")


# Alias compatible with your original notebook function name.
def markowitz(returns: pd.DataFrame, annualization_factor: int = 252) -> pd.Series:
    return markowitz_min_variance(returns, annualization_factor=annualization_factor)


def format_weights(weights: pd.Series, threshold: float = 1e-6) -> pd.Series:
    """Clean tiny numerical noise, remove near-zero weights, and normalize."""
    weights = weights.copy().astype(float).clip(lower=0.0)
    weights = weights[weights > threshold]

    if weights.empty:
        raise ValueError("All weights are zero after cleaning.")

    weights = weights / weights.sum()
    return weights.round(6)


def format_weights_percent(weights: pd.Series, threshold: float = 1e-6) -> pd.Series:
    """Return weights in percentage format."""
    return (format_weights(weights, threshold=threshold) * 100).round(2)


def equal_weight(tickers: list[str]) -> pd.Series:
    """Return equal weights for a list of tickers."""
    if not tickers:
        return pd.Series(dtype=float)
    return pd.Series(1 / len(tickers), index=tickers, name="weight")


def inverse_volatility_weights(
    returns: pd.DataFrame,
    annualization_factor: int = 252,
) -> pd.Series:
    """Return inverse-volatility weights."""
    vol = returns.std() * np.sqrt(annualization_factor)
    inv_vol = 1 / vol.replace(0, np.nan)
    weights = inv_vol / inv_vol.sum()
    return weights.dropna().rename("weight")


def apply_weights_to_returns(returns_test: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Apply asset weights to a returns DataFrame and return portfolio returns."""
    weights = weights.copy()
    available = [t for t in weights.index if t in returns_test.columns]

    if not available:
        raise ValueError("No overlap between returns columns and weights index.")

    aligned_returns = returns_test[available].dropna(how="all")
    aligned_weights = weights.loc[available]
    aligned_weights = aligned_weights / aligned_weights.sum()

    portfolio_returns = aligned_returns.dot(aligned_weights)
    portfolio_returns.name = "portfolio_returns"
    return portfolio_returns
