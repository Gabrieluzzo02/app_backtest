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

# ==============================================================================
# PARAMETRI DEGLI ASSET E INDICATORI
# ==============================================================================
tickers = ['MSFT', 'JPM', 'JNJ', 'AMZN', 'WMT', 'XOM', 'CAT', 'NEE', 'FCX', 'GOOGL']
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
        market_caps = {"MSFT": 3200, "GOOGL": 2200, "AMZN": 1900, "JPM": 570, "WMT":
