import pandas as pd

from core.optimization import (
    apply_weights_to_returns,
    equal_weight,
    format_weights,
    inverse_volatility_weights,
    markowitz,
)


def run_static_portfolio(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Apply fixed weights to all returns."""
    return apply_weights_to_returns(returns, weights)


def run_rolling_markowitz(
    prices: pd.DataFrame,
    start_year: int,
    end_year: int,
    train_window_years: int = 5,
    annualization_factor: int = 12,
) -> tuple[pd.Series, pd.DataFrame]:
    """Rolling out-of-sample Markowitz backtest by calendar year."""
    returns_total = []
    weights_history = []

    for year in range(start_year, end_year + 1):
        train_start = year - train_window_years
        train_end = year - 1

        train_prices = prices.loc[str(train_start): str(train_end)]
        test_prices = prices.loc[str(year)]

        if train_prices.empty or test_prices.empty:
            continue

        returns_train = train_prices.pct_change().dropna()
        returns_test = test_prices.pct_change().dropna()

        if returns_train.empty or returns_test.empty:
            continue

        raw_weights = markowitz(returns_train, annualization_factor=annualization_factor)
        weights = format_weights(raw_weights)

        portfolio_returns = apply_weights_to_returns(returns_test, weights)
        returns_total.append(portfolio_returns)

        weights_row = weights.to_dict()
        weights_row["Year"] = year
        weights_history.append(weights_row)

    if not returns_total:
        return pd.Series(dtype=float), pd.DataFrame()

    returns_series = pd.concat(returns_total).sort_index()
    weights_df = pd.DataFrame(weights_history).set_index("Year").fillna(0)
    return returns_series, weights_df


def run_inverse_volatility(
    returns: pd.DataFrame,
    annualization_factor: int = 252,
) -> tuple[pd.Series, pd.Series]:
    """One-shot inverse volatility portfolio."""
    weights = inverse_volatility_weights(returns, annualization_factor=annualization_factor)
    portfolio_returns = apply_weights_to_returns(returns, weights)
    return portfolio_returns, weights


def run_equal_weight(returns: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """One-shot equal-weight portfolio."""
    weights = equal_weight(list(returns.columns))
    portfolio_returns = apply_weights_to_returns(returns, weights)
    return portfolio_returns, weights
