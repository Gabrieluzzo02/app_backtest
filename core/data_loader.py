from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st
import yfinance as yf

from core.asset_search import prepare_asset_universe


@st.cache_data(show_spinner=False)
def load_asset_universe(path: str | Path = "data/assets_universe.csv") -> pd.DataFrame:
    """Load the investable universe used by the Streamlit selector."""
    assets = pd.read_csv(path)
    return prepare_asset_universe(assets)


@st.cache_data(show_spinner=True)
def download_prices(
    tickers: Iterable[str],
    start_date,
    end_date,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance via yfinance."""
    tickers = list(dict.fromkeys([str(t).strip().upper() for t in tickers if str(t).strip()]))

    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        elif "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"]
        else:
            raise ValueError("Could not find Close or Adj Close in downloaded data.")
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.dropna(how="all")
    prices = prices.ffill()
    return prices


def resample_prices(prices: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Convert daily prices to monthly prices if requested."""
    if prices.empty:
        return prices
    if frequency == "Mensile":
        return prices.resample("MS").last().ffill()
    return prices
