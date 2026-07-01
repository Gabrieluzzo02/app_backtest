import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import minimize
from pypfopt import expected_returns, risk_models, black_litterman
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.black_litterman import BlackLittermanModel
import warnings
import pandas_datareader.data as web
import datetime
import statsmodels.api as sm 
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURAZIONE DELLA PAGINA WEB
# ==============================================================================
st.set_page_config(page_title="Piattaforma Backtest Quantitativo", layout="wide")

st.title("📈 Piattaforma di Backtest: Multi-Strategia")
st.markdown("""
Questo strumento calcola e confronta le performance di 4 diversi modelli di allocazione del portafoglio 
(Markowitz, Black-Litterman, Momentum, Inverse Volatility) rispetto all'indice S&P 500.
""")
import streamlit as st
import pandas as pd

from core.asset_search import names_to_tickers


st.title("Backtesting App")

assets = pd.read_csv("data/assets_universe.csv")

user_input = st.sidebar.text_input(
    "Scrivi i nomi delle aziende",
    placeholder="Apple, Microsoft, Enel"
)

if user_input:
    company_names = [
        x.strip()
        for x in user_input.split(",")
        if x.strip()
    ]

    tickers, not_found = names_to_tickers(company_names, assets)

    st.write("Ticker trovati:")
    st.write(tickers)

    if not_found:
        st.warning(f"Non trovati: {not_found}")
# ==============================================================================
# PARAMETRI DEGLI ASSET E INDICATORI
# ==============================================================================
#tickers = ['MSFT', 'JPM', 'JNJ', 'AMZN', 'WMT', 'XOM', 'CAT', 'NEE', 'FCX', 'GOOGL']
tickers_perc = ["FEDFUNDS", "UNRATE"]
tickers_nonperc = ["UMCSENT"]
ticker_vix = ['^VIX']
commodity = ['HG=F', 'GC=F']
valute= ['CHF=X']
indice_benchmark = ['SPY']
indicatori_macro=["FEDFUNDS", "UNRATE", 'CHF=X', '^VIX']

# ==============================================================================
# FUNZIONI CORE (Con Cache per velocizzare il sito)
# ==============================================================================
@st.cache_data(show_spinner=False)
def download_prices(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data['Close']

@st.cache_data(show_spinner=False)
def download_macro_data(start_date, end_date):
    macro_grezzi_nonperc = web.DataReader(tickers_nonperc, 'fred', start=start_date, end=end_date).pct_change().dropna()
    macro_grezzi_perc = web.DataReader(tickers_perc, 'fred', start=start_date, end=end_date).pct_change().dropna()
    return macro_grezzi_perc, macro_grezzi_nonperc

def portfolio_stats(series):
    returns = series.pct_change().dropna()
    if len(returns) == 0:
        return {"Return": 0, "Volatility": 0, "Sharpe": 0, "Max Drawdown": 0}
    
    cumulative_return = (series.iloc[-1] / series.iloc[0]) - 1
    volatility = returns.std() * np.sqrt(252) # Annualizzazione approssimativa
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
    drawdown = series / series.cummax() - 1
    max_dd = drawdown.min()
    
    return {
        "Rendimento Totale": cumulative_return,
        "Volatilità": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }

def calculate_returns(prices):
    return prices.pct_change().dropna()

def markowitz(returns):
    mean_returns = returns.mean() * 12
    cov_matrix = returns.cov() * 12
    n_assets = len(mean_returns)
    
    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)
        
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.ones(n_assets) / n_assets
    result = minimize(portfolio_volatility, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=returns.columns)

def format_weights(weights):
    weights = weights.copy()
    weights = weights.clip(lower=0)
    weights = weights[weights > 1e-6]
    weights = weights / weights.sum()
    return weights.round(4)

def apply_weights_to_returns(returns_test, weights):
    returns_test = returns_test[weights.index]
    portfolio_returns = returns_test.dot(weights)
    return portfolio_returns

# ==============================================================================
# BARRA LATERALE (INPUT UTENTE)
# ==============================================================================
st.sidebar.header("⚙️ Impostazioni Backtest")
st.sidebar.markdown("Modifica i parametri del modello rolling.")

anno_inizio = st.sidebar.number_input("Anno di Inizio", min_value=2005, max_value=2030, value=2012)
anno_fine = st.sidebar.number_input("Anno di Fine", min_value=2006, max_value=2030, value=2024)
finestra_train = st.sidebar.number_input("Finestra Train (anni)", min_value=1, max_value=10, value=5)

avvia_simulazione = st.sidebar.button("🚀 Avvia Backtest")

if anno_inizio >= anno_fine:
    st.sidebar.error("L'anno di inizio deve essere precedente all'anno di fine.")

# ==============================================================================
# MOTORE PRINCIPALE DEL BACKTEST
# ==============================================================================
if avvia_simulazione and anno_inizio < anno_fine:
    # Definisco l'intervallo di download dati garantendo margine per i lag e la finestra di training
    data_download_start = f"{anno_inizio - finestra_train - 2}-01-01"
    data_download_end = f"{anno_fine + 1}-01-01"

    with st.spinner('Calcolo modelli e simulazione in corso... Attendere.'):
        
        # ----------------------------------------------------------------------
        # 1. SCARICAMENTO E PREPARAZIONE DATI
        # ----------------------------------------------------------------------
        prices_giornalieri = download_prices(tickers, start_date=data_download_start, end_date=data_download_end)
        prices = prices_giornalieri.resample('MS').last().ffill()
        prezzi_asset_mensili = prices
        rendimenti_asset_mensili = prezzi_asset_mensili.pct_change().dropna()
        
        benchmark = download_prices(indice_benchmark, start_date=data_download_start, end_date=data_download_end)
        
        macro_grezzi_perc, macro_grezzi_nonperc = download_macro_data(data_download_start, data_download_end)
        valute_grezze = download_prices(valute, start_date=data_download_start, end_date=data_download_end).resample('MS').last().dropna()
        rendimenti_valute_mensili = valute_grezze.pct_change().dropna()
        
        vix = download_prices(ticker_vix, start_date=data_download_start, end_date=data_download_end).resample('MS').last().dropna()
        rendimenti_vix = vix.pct_change().dropna()
        
        macro_finale = pd.concat([macro_grezzi_perc, macro_grezzi_nonperc, rendimenti_vix, rendimenti_valute_mensili], axis=1)
        finale = pd.concat([prices, macro_finale], axis=1).dropna()

        # ----------------------------------------------------------------------
        # 2. MARKOWITZ OUT-OF-SAMPLE
        # ----------------------------------------------------------------------
        storico_change = []
        for anno in range(anno_inizio, anno_fine + 1):
            fine = anno - 1
            inizio = anno - finestra_train
            try:
                returns_train = calculate_returns(prices[tickers].loc[str(inizio) : str(fine)])
                return_test= calculate_returns(prices[tickers].loc[str(anno)])
                weights = markowitz(returns_train)
                weights_clean = format_weights(weights)
                rendimento_portfolio = apply_weights_to_returns(return_test, weights_clean)
                storico_change.append(rendimento_portfolio)
            except:
                continue
        
        if storico_change:
            rendimenti_totali_mark = pd.concat(storico_change)
            equity_curve_mark = (1 + rendimenti_totali_mark).cumprod()
        else:
            equity_curve_mark = pd.Series(dtype=float)

        # ----------------------------------------------------------------------
        # 3. REGRESSIONE OLS E BLACK-LITTERMAN OUT-OF-SAMPLE
        # ----------------------------------------------------------------------
        lag_veloce = 1
        lag_lento = 2
        storico_previsioni = []
        
        # Estrazione Views tramite OLS
        for anno in range(anno_inizio, anno_fine + 1):
            fine = anno - 1
            inizio = anno - finestra_train
            try:
                price_train = prices[tickers].loc[str(inizio) : str(fine)]
                rendimenti_train = prices.loc[str(inizio) : str(fine)].pct_change().dropna()
                rendimenti_macro_train = finale[indicatori_macro].loc[str(inizio) : str(fine)]
                
                merged_data = pd.concat([rendimenti_train, rendimenti_macro_train], axis=1).dropna()
                previsioni_anno_corrente = {'Anno_Previsione': anno}
                
                for variabile_dipendente in tickers:
                    data = merged_data.copy()
                    if merged_data.empty: continue
                    
                    data['Stock_Return_Lag1'] = merged_data[variabile_dipendente].shift(lag_veloce)
                    data['FEDFUNDS_Diff_Lag1'] = merged_data['FEDFUNDS'].shift(lag_veloce)
                    data['UNRATE_Diff_Lag1'] = merged_data['UNRATE'].shift(lag_lento)
                    data['CHF=X_Diff_Lag1'] = merged_data['CHF=X'].shift(lag_veloce)
                    data['VIX_Diff_Lag1'] = merged_data['^VIX'].shift(lag_veloce)
                    data.replace([np.inf, -np.inf], np.nan, inplace=True)
                    data = data.dropna()
                    
                    if len(data) < 10: continue
                    
                    regressori = ['Stock_Return_Lag1', 'FEDFUNDS_Diff_Lag1', 'UNRATE_Diff_Lag1', 'CHF=X_Diff_Lag1','VIX_Diff_Lag1']
                    ultimo_macro = rendimenti_macro_train.iloc[-1]
                    ultimo_rendimento = merged_data[variabile_dipendente].iloc[-1]
                    
                    Y = data[variabile_dipendente]
                    X = sm.add_constant(data[regressori])
                    
                    X_new = pd.DataFrame({
                        "const": [1.0],
                        "Stock_Return_Lag1":[ultimo_rendimento],
                        "FEDFUNDS_Diff_Lag1":[ultimo_macro["FEDFUNDS"]],
                        "UNRATE_Diff_Lag1":[ultimo_macro["UNRATE"]],
                        "CHF=X_Diff_Lag1":[ultimo_macro["CHF=X"]],
                        "VIX_Diff_Lag1":[ultimo_macro["^VIX"]]
                    })
                    X_new = X_new[X.columns]
                    
                    model = sm.OLS(Y, X).fit()
                    prediction = model.predict(X_new).iloc[0]*12
                    previsioni_anno_corrente[variabile_dipendente] = prediction
                    
                storico_previsioni.append(previsioni_anno_corrente)
            except:
                continue

        df_views = pd.DataFrame(storico_previsioni)
        if not df_views.empty:
            df_views = df_views.set_index('Anno_Previsione')

        # Esecuzione Black-Litterman
        market_caps = {"MSFT": 3200, "GOOGL": 2200, "AMZN": 1900, "JPM": 570, "WMT": 530, "XOM": 460, "JNJ": 360, "CAT": 170, "NEE": 150, "FCX": 75}
        risultati_out_of_sample_bl = []
        delta = 2.5
        
        for anno in range(anno_inizio, anno_fine + 1):
            anno_inizio_train_bl = anno - finestra_train
            anno_fine_train_bl = anno - 1
            try:
                price_train_bl = prezzi_asset_mensili.loc[str(anno_inizio_train_bl) : str(anno_fine_train_bl)]
                rendimenti_test_bl = rendimenti_asset_mensili.loc[str(anno)]
                
                S_bl = risk_models.sample_cov(price_train_bl, frequency=12)
                rendimenti_impliciti_mercato = black_litterman.market_implied_prior_returns(market_caps, delta, S_bl)
                
                if anno in df_views.index:
                    views_pulite = df_views.loc[anno].dropna()
                    if not views_pulite.empty:
                        views_assolute = (views_pulite / 100).to_dict()
                        bl = BlackLittermanModel(S_bl, pi=rendimenti_impliciti_mercato, absolute_views=views_assolute)
                        rendimenti_bl = bl.bl_returns()
                        S_posteriore = bl.bl_cov()
                        
                        ef_bl = EfficientFrontier(rendimenti_bl, S_posteriore, weight_bounds=(0, 1))
                        ef_bl.max_quadratic_utility(risk_aversion=delta)
                        pesi_series = pd.Series(ef_bl.clean_weights())
                        
                        crescita_titoli = (1 + rendimenti_test_bl).cumprod()
                        valore_portafoglio = (crescita_titoli * pesi_series).sum(axis=1)
                        rendimenti_portafoglio_bl = valore_portafoglio.pct_change()
                        rendimenti_portafoglio_bl.iloc[0] = valore_portafoglio.iloc[0] - 1
                        
                        risultati_out_of_sample_bl.append(rendimenti_portafoglio_bl)
            except:
                continue
                
        if risultati_out_of_sample_bl:
            rendimenti_mark_bl = pd.concat(risultati_out_of_sample_bl)
            equity_curve_bl = (1 + rendimenti_mark_bl).cumprod()
        else:
            equity_curve_bl = pd.Series(dtype=float)

        # ----------------------------------------------------------------------
        # 4. MOMENTUM OUT-OF-SAMPLE
        # ----------------------------------------------------------------------
        risultati_out_of_sample_mom = []
        top_n = 3
        for anno in range(anno_inizio, anno_fine + 1):
            anno_fine_train = anno - 1
            anno_inizio_train = anno - 2
            try:
                # Usa 'YE' (o 'Y') compatibile con le versioni moderne di pandas
                prezzo_inizio = prezzi_asset_mensili.loc[str(anno_inizio_train)].resample('YE').last().iloc[-1]
                prezzo_fine = prezzi_asset_mensili.loc[str(anno_fine_train)].resample('YE').last().iloc[-1]
                rendimento_12m = (prezzo_fine / prezzo_inizio) - 1
                
                vincitori = rendimento_12m.sort_values(ascending=False).head(top_n).index
                pesi_mom = {ticker: (1.0 / top_n if ticker in vincitori else 0.0) for ticker in tickers}
                
                price_test = prices.loc[str(anno)]
                rendimenti_test = price_test.pct_change().dropna()
                rendimenti_portafoglio_mom = (rendimenti_test * pd.Series(pesi_mom)).sum(axis=1)
                risultati_out_of_sample_mom.append(rendimenti_portafoglio_mom)
            except:
                continue

        if risultati_out_of_sample_mom:
            rendimenti_totali_mom = pd.concat(risultati_out_of_sample_mom)
            equity_curve_mom = (1 + rendimenti_totali_mom).cumprod().resample('MS').last()
        else:
            equity_curve_mom = pd.Series(dtype=float)

        # ----------------------------------------------------------------------
        # 5. INVERSE VOLATILITY OUT-OF-SAMPLE
        # ----------------------------------------------------------------------
        risultati_out_of_sample_invvol = []
        for anno in range(anno_inizio, anno_fine + 1):
            anno_inizio_train = anno - finestra_train
            anno_fine_train = anno - 1
            try:
                rendimenti_train = rendimenti_asset_mensili.loc[str(anno_inizio_train) : str(anno_fine_train)]
                rendimenti_test = rendimenti_asset_mensili.loc[str(anno)]
                
                volatilita_titoli = rendimenti_train.std() * np.sqrt(12)
                inv_vol = 1.0 / volatilita_titoli
                pesi_invvol_puliti = inv_vol / inv_vol.sum()
                
                crescita_titoli = (1 + rendimenti_test).cumprod()
                matrice_crescita = crescita_titoli.values
                vettore_pesi = np.array([pesi_invvol_puliti[ticker] for ticker in crescita_titoli.columns])
                
                valore_portafoglio = np.dot(matrice_crescita, vettore_pesi)
                val_series = pd.Series(valore_portafoglio, index=crescita_titoli.index)
                rendimenti_portafoglio = val_series.pct_change()
                rendimenti_portafoglio.iloc[0] = val_series.iloc[0] - 1
                risultati_out_of_sample_invvol.append(rendimenti_portafoglio)
            except:
                continue

        if risultati_out_of_sample_invvol:
            rendimenti_invvol = pd.concat(risultati_out_of_sample_invvol)
            equity_curve_invvol = (1 + rendimenti_invvol).cumprod()
        else:
            equity_curve_invvol = pd.Series(dtype=float)


    # ==============================================================================
    # OUTPUT A SCHERMO (SOLO GRAFICO FINALE E TABELLA INDICATORI)
    # ==============================================================================
    st.success("✅ Backtest completato con successo!")

    # 1. Preparazione Dati Grafico e S&P500 Benchmark
    benchmark_return = benchmark.loc[str(anno_inizio):str(anno_fine)].resample('MS').last().ffill().pct_change().dropna()
    if not benchmark_return.empty:
        equity_curve_benchmark = (1 + benchmark_return).cumprod()
    else:
        equity_curve_benchmark = pd.Series(dtype=float)

    try:
        df_confronto= pd.concat([equity_curve_bl, equity_curve_mark, equity_curve_mom, equity_curve_invvol, equity_curve_benchmark], axis=1)
        df_confronto.columns = ["Black-Litterman", "Markowitz", "Momentum", "Inverse Volatility", "S&P 500"]
        df_confronto = df_confronto.dropna(how='all').ffill().fillna(1.0)

        # 2. Rendering del Grafico
        fig, ax = plt.subplots(figsize=(15, 8), dpi=120)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        
        colors = ["#FFD700", "#00FFFF", "#7CFC00", "#FF69B4", "#8A2BE2"]
        
        for i, col in enumerate(df_confronto.columns):
            if not df_confronto[col].isna().all():
                ax.plot(df_confronto.index, df_confronto[col], lw=2.6, color=colors[i], label=col)
                ax.scatter(df_confronto.index[-1], df_confronto[col].iloc[-1], s=50, color=colors[i], edgecolors="white", linewidth=0.8, zorder=5)

        ax.set_title("Confronto Equity Curves", fontsize=22, color="white", fontweight="bold", loc="left", pad=20)
        ax.set_ylabel("Portfolio Value (Base 1)", fontsize=14, color="white")
        ax.set_xlabel("Time", fontsize=14, color="white")
        
        ax.grid(True, which="major", color="#444444", linewidth=0.7)
        ax.minorticks_on()
        ax.grid(True, which="minor", color="#222222", linewidth=0.4)
        ax.tick_params(colors="white", labelsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("white")
        ax.spines["bottom"].set_color("white")
        
        leg = ax.legend(loc="upper left", frameon=False, fontsize=12)
        for text in leg.get_texts():
            text.set_color("white")
            
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.2f}'))
        plt.tight_layout()
        
        st.pyplot(fig)

        # 3. Creazione e Rendering della Tabella
        st.subheader("📉 Analisi Metriche Portafoglio e Scenari di Crisi")
        eventi = {
            "Intero Periodo Selezionato": (f"{anno_inizio}-01-01", f"{anno_fine}-12-31"),
            "Euro Debt Crisis aftermath": ("2012-01-01", "2012-12-31"),
            "Taper Tantrum": ("2013-05-01", "2013-09-30"),
            "China Devaluation shock": ("2015-08-01", "2015-10-01"),
            "COVID Crash": ("2020-02-01", "2020-04-30"),
            "Inflation + Rates shock": ("2022-01-01", "2023-03-31"),
            "AI / Tech rally phase": ("2023-01-01", "2024-12-31")
        }
        
        results = []
        for event, (start, end) in eventi.items():
            try:
                df_event = df_confronto.loc[start:end]
                if len(df_event) > 1:
                    for col in df_event.columns:
                        if not df_event[col].isna().all():
                            stats = portfolio_stats(df_event[col])
                            stats["Scenario Temporale"] = event
                            stats["Strategia"] = col
                            results.append(stats)
            except Exception:
                continue
                
        if results:
            df_event_analysis = pd.DataFrame(results)
            cols = ['Scenario Temporale', 'Strategia', 'Rendimento Totale', 'Volatilità', 'Sharpe Ratio', 'Max Drawdown']
            df_event_analysis = df_event_analysis[cols]
            
            st.dataframe(
                df_event_analysis.style.format({
                    'Rendimento Totale': '{:.2%}',
                    'Volatilità': '{:.2%}',
                    'Sharpe Ratio': '{:.2f}',
                    'Max Drawdown': '{:.2%}'
                }),
                use_container_width=True,
                height=600
            )

    except Exception as e:
        st.error("Si è verificato un errore durante la generazione del grafico. Assicurati che il range di date selezionato contenga dati di trading validi.")

else:
    if not avvia_simulazione:
        st.info("👈 Seleziona gli anni desiderati nella barra laterale a sinistra e clicca su 'Avvia Backtest' per visualizzare i risultati.")
