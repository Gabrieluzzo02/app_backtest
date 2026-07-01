import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable, Tuple

import pandas as pd


REQUIRED_COLUMNS = ["name", "ticker", "yahoo_ticker", "country", "sector"]


def normalize_text(text: object) -> str:
    """Normalize text for robust search: lowercase, no accents, clean spaces."""
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s\.\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def similarity(a: object, b: object) -> float:
    """Return string similarity between 0 and 1."""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def prepare_asset_universe(assets: pd.DataFrame) -> pd.DataFrame:
    """Prepare an asset DataFrame for search and display in Streamlit."""
    assets = assets.copy()

    for col in REQUIRED_COLUMNS:
        if col not in assets.columns:
            assets[col] = ""

    assets["name"] = assets["name"].fillna("").astype(str)
    assets["ticker"] = assets["ticker"].fillna("").astype(str).str.upper()
    assets["yahoo_ticker"] = assets["yahoo_ticker"].fillna(assets["ticker"]).astype(str).str.upper()
    assets["country"] = assets["country"].fillna("").astype(str)
    assets["sector"] = assets["sector"].fillna("").astype(str)

    assets["display_label"] = (
        assets["name"]
        + " ("
        + assets["yahoo_ticker"]
        + ") - "
        + assets["country"]
        + " - "
        + assets["sector"]
    )

    assets["search_text"] = (
        assets["name"].astype(str)
        + " "
        + assets["ticker"].astype(str)
        + " "
        + assets["yahoo_ticker"].astype(str)
        + " "
        + assets["country"].astype(str)
        + " "
        + assets["sector"].astype(str)
    ).apply(normalize_text)

    assets = assets.drop_duplicates(subset=["yahoo_ticker"]).reset_index(drop=True)
    return assets


def parse_company_names(user_input: str) -> list[str]:
    """Convert 'Apple, Microsoft, Enel' into ['Apple', 'Microsoft', 'Enel']."""
    if not user_input:
        return []
    return [x.strip() for x in user_input.split(",") if x.strip()]


def search_assets(query: str, assets: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Search assets by company name, ticker, country or sector."""
    query_norm = normalize_text(query)
    if not query_norm:
        return pd.DataFrame()

    prepared = assets.copy()
    if "search_text" not in prepared.columns:
        prepared = prepare_asset_universe(prepared)

    prepared["contains_score"] = prepared["search_text"].apply(
        lambda x: 1.0 if query_norm in x else 0.0
    )
    prepared["name_score"] = prepared["name"].apply(lambda x: similarity(query_norm, x))
    prepared["ticker_score"] = prepared["yahoo_ticker"].apply(lambda x: similarity(query_norm, x))

    # Ticker exact match should dominate all other results.
    prepared["exact_ticker_score"] = prepared["yahoo_ticker"].apply(
        lambda x: 3.0 if normalize_text(x) == query_norm else 0.0
    )

    prepared["score"] = (
        prepared["exact_ticker_score"]
        + prepared["contains_score"] * 2.0
        + prepared["name_score"]
        + prepared["ticker_score"]
    )

    results = prepared.sort_values("score", ascending=False)
    results = results[results["score"] > 0.25]

    output_cols = [
        "name",
        "ticker",
        "yahoo_ticker",
        "country",
        "sector",
        "display_label",
        "score",
    ]
    existing = [c for c in output_cols if c in results.columns]
    return results.head(top_n)[existing].reset_index(drop=True)


def names_to_tickers(company_names: Iterable[str], assets: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """
    Simple conversion: company names -> yahoo tickers.

    This is useful in notebooks/Colab. For the Streamlit app, prefer `search_assets()`
    because it lets users choose among ambiguous matches.
    """
    tickers: list[str] = []
    not_found: list[str] = []

    prepared = assets.copy()
    if "search_text" not in prepared.columns:
        prepared = prepare_asset_universe(prepared)

    for company in company_names:
        matches = search_assets(company, prepared, top_n=1)
        if matches.empty:
            not_found.append(company)
            continue
        ticker = matches.iloc[0]["yahoo_ticker"]
        if ticker not in tickers:
            tickers.append(ticker)

    return tickers, not_found


def names_to_tickers_with_matches(
    company_names: Iterable[str], assets: pd.DataFrame
) -> tuple[list[str], pd.DataFrame, list[str]]:
    """Return tickers, a match table, and names that were not found."""
    rows = []
    not_found = []

    prepared = prepare_asset_universe(assets)

    for company in company_names:
        matches = search_assets(company, prepared, top_n=1)
        if matches.empty:
            not_found.append(company)
            continue
        best = matches.iloc[0]
        rows.append(
            {
                "input": company,
                "matched_name": best.get("name"),
                "yahoo_ticker": best.get("yahoo_ticker"),
                "country": best.get("country"),
                "sector": best.get("sector"),
                "score": best.get("score"),
            }
        )

    matches_df = pd.DataFrame(rows)
    tickers = matches_df["yahoo_ticker"].tolist() if not matches_df.empty else []
    return tickers, matches_df, not_found
