import pandas as pd
import streamlit as st

from core.asset_search import names_to_tickers_with_matches, parse_company_names, search_assets


def _add_tickers(tickers: list[str]) -> None:
    """Add tickers to Streamlit session state without duplicates."""
    if "selected_tickers" not in st.session_state:
        st.session_state.selected_tickers = []

    for ticker in tickers:
        if ticker and ticker not in st.session_state.selected_tickers:
            st.session_state.selected_tickers.append(ticker)


def asset_selector_sidebar(assets: pd.DataFrame) -> list[str]:
    """Sidebar UI to search, select and remove tickers from the portfolio."""
    if "selected_tickers" not in st.session_state:
        st.session_state.selected_tickers = []

    st.sidebar.header("Asset selector")

    selector_mode = st.sidebar.radio(
        "Metodo di selezione",
        [
            "Ricerca guidata",
            "Lista nomi aziende",
            "Menu con filtri",
        ],
    )

    filtered_assets = assets.copy()

    # Common filters
    with st.sidebar.expander("Filtri universo investibile", expanded=False):
        sectors = sorted([x for x in assets["sector"].dropna().unique() if str(x).strip()])
        countries = sorted([x for x in assets["country"].dropna().unique() if str(x).strip()])
        regions = sorted([x for x in assets.get("market_region", pd.Series(dtype=str)).dropna().unique() if str(x).strip()])

        selected_sectors = st.multiselect("Settore", sectors)
        selected_countries = st.multiselect("Paese", countries)
        selected_regions = st.multiselect("Area mercato", regions)

        if selected_sectors:
            filtered_assets = filtered_assets[filtered_assets["sector"].isin(selected_sectors)]
        if selected_countries:
            filtered_assets = filtered_assets[filtered_assets["country"].isin(selected_countries)]
        if selected_regions and "market_region" in filtered_assets.columns:
            filtered_assets = filtered_assets[filtered_assets["market_region"].isin(selected_regions)]

    if selector_mode == "Ricerca guidata":
        query = st.sidebar.text_input(
            "Cerca titolo",
            placeholder="Es. Apple, Microsoft, Enel, ASML...",
        )

        if query:
            results = search_assets(query, filtered_assets, top_n=10)

            if results.empty:
                st.sidebar.warning("Nessun risultato trovato.")
            else:
                selected_idx = st.sidebar.selectbox(
                    "Risultati trovati",
                    options=list(range(len(results))),
                    format_func=lambda i: results.loc[i, "display_label"],
                )

                selected_asset = results.loc[selected_idx]

                st.sidebar.caption("Titolo selezionato")
                st.sidebar.write(selected_asset["display_label"])

                if st.sidebar.button("Aggiungi al portafoglio"):
                    _add_tickers([selected_asset["yahoo_ticker"]])
                    st.sidebar.success(f"{selected_asset['yahoo_ticker']} aggiunto.")

    elif selector_mode == "Lista nomi aziende":
        user_input = st.sidebar.text_area(
            "Scrivi nomi separati da virgola",
            placeholder="Apple, Microsoft, Enel",
            height=90,
        )

        if user_input:
            company_names = parse_company_names(user_input)
            tickers, matches_df, not_found = names_to_tickers_with_matches(company_names, filtered_assets)

            if not matches_df.empty:
                st.sidebar.caption("Match trovati")
                st.sidebar.dataframe(matches_df, use_container_width=True, hide_index=True)

            if not_found:
                st.sidebar.warning(f"Non trovati: {', '.join(not_found)}")

            if tickers and st.sidebar.button("Aggiungi ticker trovati"):
                _add_tickers(tickers)
                st.sidebar.success("Ticker aggiunti al portafoglio.")

    else:
        ticker_to_label = dict(zip(filtered_assets["yahoo_ticker"], filtered_assets["display_label"]))

        selected = st.sidebar.multiselect(
            "Scegli asset dal menu",
            options=filtered_assets["yahoo_ticker"].tolist(),
            format_func=lambda t: ticker_to_label.get(t, t),
        )

        if selected and st.sidebar.button("Aggiungi selezionati"):
            _add_tickers(selected)
            st.sidebar.success("Asset aggiunti.")

    st.sidebar.divider()
    st.sidebar.subheader("Portafoglio")

    if st.session_state.selected_tickers:
        st.sidebar.write(st.session_state.selected_tickers)

        ticker_to_remove = st.sidebar.selectbox(
            "Rimuovi ticker",
            options=st.session_state.selected_tickers,
        )

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Rimuovi"):
                st.session_state.selected_tickers.remove(ticker_to_remove)
                st.rerun()
        with col2:
            if st.button("Svuota"):
                st.session_state.selected_tickers = []
                st.rerun()
    else:
        st.sidebar.info("Nessun ticker selezionato.")

    return st.session_state.selected_tickers
