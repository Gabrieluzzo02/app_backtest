import numpy as np
import pandas as pd


def equity_curve_from_returns(returns: pd.Series, initial_capital: float = 1.0) -> pd.Series:
    """Convert return series to equity curve."""
    equity = (1 + returns.fillna(0)).cumprod() * initial_capital
    equity.name = "equity_curve"
    return equity


def portfolio_stats_from_returns(
    returns: pd.Series,
    annualization_factor: int = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    """Calculate common portfolio metrics from return series."""
    returns = returns.dropna()
    if returns.empty:
        return {}

    equity = equity_curve_from_returns(returns)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) > 1 else 0.0
    volatility = returns.std() * np.sqrt(annualization_factor)

    excess_periodic = returns - (risk_free_rate / annualization_factor)
    sharpe = np.nan
    if excess_periodic.std() != 0:
        sharpe = excess_periodic.mean() / excess_periodic.std() * np.sqrt(annualization_factor)

    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
    }


def portfolio_stats(series: pd.Series) -> dict:
    """Compatibility function from your notebook: stats from a price/equity series."""
    returns = series.pct_change().dropna()
    if returns.empty:
        return {}
    cumulative_return = (series.iloc[-1] / series.iloc[0]) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else np.nan
    drawdown = series / series.cummax() - 1
    max_dd = drawdown.min()
    return {
        "Return": cumulative_return,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
    }
