import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from core.backtest import run_equal_weight, run_inverse_volatility, run_rolling_markowitz
from core.data_loader import download_prices, load_asset_universe, resample_prices
from core.metrics import equity_curve_from_returns, portfolio_stats_from_returns
from core.optimization import apply_weights_to_returns, format_weights, format_weights_percent, markowitz
from core.returns import calculate_returns
from ui.asset_selector import asset_selector_sidebar


st.set_page_config(
    page_title="Backtesting App",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Backtesting App")
st.caption("Prototipo Streamlit: selezione asset, download prezzi, Markowitz e backtest base.")

assets = load_asset_universe("data/assets_universe.csv")
selected_tickers = asset_selector_sidebar(assets)

with st.sidebar:
    st.divider()
    st.header("Parametri backtest")

    start_date = st.date_input("Data inizio", value=dt.date(2015, 1, 1))
    end_date = st.date_input("Data fine", value=dt.date.today())

    frequency = st.radio("Frequenza dati", ["Giornaliera", "Mensile"], index=1)
    annualization_factor = 12 if frequency == "Mensile" else 252

    initial_capital = st.number_input(
        "Capitale iniziale",
        min_value=1.0,
        value=10_000.0,
        step=1_000.0,
    )

    model = st.selectbox(
        "Modello",
        ["Markowitz - Min Variance", "Equal Weight", "Inverse Volatility", "Rolling Markowitz OOS"],
    )

    train_window = st.number_input(
        "Finestra training rolling (anni)",
        min_value=1,
        max_value=20,
        value=5,
        disabled=(model != "Rolling Markowitz OOS"),
    )

    run_button = st.button("Esegui backtest", type="primary")


if not selected_tickers:
    st.info("Seleziona almeno un asset dalla sidebar per iniziare.")
    st.stop()

st.subheader("Ticker selezionati")
st.write(selected_tickers)

if run_button:
    if start_date >= end_date:
        st.error("La data di inizio deve essere precedente alla data di fine.")
        st.stop()

    with st.spinner("Download prezzi e calcolo backtest..."):
        prices_raw = download_prices(selected_tickers, start_date, end_date)
        prices = resample_prices(prices_raw, frequency)

    if prices.empty:
        st.error("Nessun dato scaricato. Controlla i ticker o le date selezionate.")
        st.stop()

    returns = calculate_returns(prices)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Prezzi",
        "Pesi",
        "Equity curve",
        "Metriche",
    ])

    with tab1:
        st.write("Prezzi usati dal modello")
        st.dataframe(prices.tail(20), use_container_width=True)
        fig_prices = px.line(prices, title="Prezzi storici")
        st.plotly_chart(fig_prices, use_container_width=True)

    if model == "Markowitz - Min Variance":
        raw_weights = markowitz(returns, annualization_factor=annualization_factor)
        weights = format_weights(raw_weights)
        portfolio_returns = apply_weights_to_returns(returns, weights)
        weights_history = pd.DataFrame()

    elif model == "Equal Weight":
        portfolio_returns, weights = run_equal_weight(returns)
        weights_history = pd.DataFrame()

    elif model == "Inverse Volatility":
        portfolio_returns, weights = run_inverse_volatility(
            returns,
            annualization_factor=annualization_factor,
        )
        weights = format_weights(weights)
        weights_history = pd.DataFrame()

    else:
        start_year = start_date.year + int(train_window)
        end_year = end_date.year
        portfolio_returns, weights_history = run_rolling_markowitz(
            prices,
            start_year=start_year,
            end_year=end_year,
            train_window_years=int(train_window),
            annualization_factor=annualization_factor,
        )

        if portfolio_returns.empty:
            st.error("Backtest rolling vuoto. Aumenta il periodo storico o riduci la finestra training.")
            st.stop()

        weights = weights_history.iloc[-1].replace(0, pd.NA).dropna()

    equity_curve = equity_curve_from_returns(portfolio_returns, initial_capital=initial_capital)
    stats = portfolio_stats_from_returns(
        portfolio_returns,
        annualization_factor=annualization_factor,
    )

    with tab2:
        st.write("Pesi finali")
        weights_percent = format_weights_percent(weights)
        weights_df = weights_percent.rename("Peso %").to_frame()
        st.dataframe(weights_df, use_container_width=True)

        fig_weights = px.bar(
            weights_df.reset_index().rename(columns={"index": "Ticker"}),
            x="Ticker",
            y="Peso %",
            title="Pesi del portafoglio (%)",
        )
        st.plotly_chart(fig_weights, use_container_width=True)

        if not weights_history.empty:
            st.write("Storico pesi rolling")
            st.dataframe((weights_history * 100).round(2), use_container_width=True)

    with tab3:
        equity_df = equity_curve.to_frame("Equity Curve")
        st.dataframe(equity_df.tail(20), use_container_width=True)
        fig_equity = px.line(equity_df, title="Equity Curve")
        st.plotly_chart(fig_equity, use_container_width=True)

    with tab4:
        stats_df = pd.DataFrame.from_dict(stats, orient="index", columns=["Valore"])
        st.dataframe(stats_df, use_container_width=True)

        st.write("Rendimenti portafoglio")
        st.dataframe(portfolio_returns.tail(20).to_frame(), use_container_width=True)

else:
    st.info("Imposta i parametri e clicca **Esegui backtest**.")
