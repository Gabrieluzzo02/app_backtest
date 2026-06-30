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
import warnings
import pandas_datareader.data as web
import datetime
import statsmodels.api as sm 
import matplotlib.dates as mdates
from pypfopt.black_litterman import BlackLittermanModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')

# Configurazione della pagina Streamlit
st.set_page_config(page_title="Analisi Portafogli Multipli", layout="wide")

st.title("📈 Analisi Backtest: Markowitz vs Black-Litterman vs Momentum vs Inverse Volatility")
st.markdown("""
Questa applicazione web esegue un backtest su 10 titoli USA diversificati per settore, 
confrontando diverse strategie di ottimizzazione del portafoglio.
""")

# ==============================================================================
# PARAMETRI INIZIALI
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
# FUNZIONI (con Caching per velocizzare Streamlit)
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
    cumulative_return = (series.iloc[-1] / series.iloc[0]) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    drawdown = series / series.cummax() - 1
    max_dd = drawdown.min()
    return {
        "Return": cumulative_return,
        "Volatility": volatility,
        "Sharpe": sharpe,
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

def predict(df, anno_predizione, alpha, coeff):
    last_row = df.iloc[[-1]]
    prediction = (alpha + last_row * coeff).sum().round(2)
    return prediction

# ==============================================================================
# SIDEBAR PER I CONTROLLI
# ==============================================================================
st.sidebar.header("Impostazioni di Simulazione")
start_year = st.sidebar.slider("Anno inizio Test", 2012, 2023, 2012)
end_year = 2025 # Fissato come da tuo codice
finestra_train = st.sidebar.number_input("Anni di Lookback (Finestra Train)", min_value=1, max_value=10, value=5)

avvia_simulazione = st.sidebar.button("🚀 Avvia Backtest Completo")

# ==============================================================================
# ESECUZIONE PRINCIPALE
# ==============================================================================
if avvia_simulazione:
    with st.spinner('Scaricamento dati storici in corso... (potrebbe richiedere 1-2 minuti)'):
        
        # DOWNLOAD DATI
        prices = download_prices(tickers, start_date="2007-01-01", end_date="2024-12-31").resample('MS').last().ffill()
        prices_giornalieri = download_prices(tickers, start_date="2007-01-01", end_date="2024-12-31")
        prezzi_asset_mensili = prices
        rendimenti_asset_mensili = prezzi_asset_mensili.pct_change().dropna()
        
        materie_prime = download_prices(commodity, start_date="2007-01-01", end_date="2024-12-31")
        benchmark = download_prices(indice_benchmark, start_date="2007-01-01", end_date="2024-12-31")
        
        macro_grezzi_perc, macro_grezzi_nonperc = download_macro_data("2007-01-01", "2024-12-31")
        
        valute_grezze = download_prices(valute, start_date="2007-01-01", end_date="2025-01-01").resample('MS').last().dropna()
        rendimenti_valute_mensili = valute_grezze.pct_change().dropna()
        
        vix = download_prices(ticker_vix, start_date="2007-01-01", end_date="2024-12-31").resample('MS').last().dropna()
        rendimenti_vix = vix.pct_change().dropna()
        
        macro_finale = pd.concat([macro_grezzi_perc, macro_grezzi_nonperc, rendimenti_vix, rendimenti_valute_mensili], axis=1)
        finale = pd.concat([prices, macro_finale], axis=1).dropna()

    st.success("Dati scaricati con successo!")
    
    st.subheader("Dati Finali (Preview)")
    st.dataframe(finale.tail())

    # ==============================================================================
    # MARKOWITZ
    # ==============================================================================
    with st.spinner('Calcolo Portafoglio Markowitz...'):
        storico_change = []
        for anno in range(start_year, end_year):
            fine = anno - 1
            inizio = anno - finestra_train
            returns_train = calculate_returns(prices[tickers].loc[str(inizio) : str(fine)])
            return_test= calculate_returns(prices[tickers].loc[str(anno)])
            
            weights = markowitz(returns_train)
            weights_clean = format_weights(weights)
            rendimento_portfolio = apply_weights_to_returns(return_test, weights_clean)
            storico_change.append(rendimento_portfolio)
            
        rendimenti_totali = pd.concat(storico_change)
        equity_curve_mark = (1 + rendimenti_totali).cumprod()

    # ==============================================================================
    # CALCOLO VIEWS & OLS (Per Black-Litterman)
    # ==============================================================================
    with st.spinner('Calcolo Views Macro (Regressione OLS)...'):
        lag_veloce = 1
        lag_lento = 2
        storico_previsioni = []
        storico_rendimenti_effettivi = []
        storico_r_quadro = []
        
        # Test di Multicollinearità (VIF)
        inizio_vif = start_year - finestra_train
        fine_vif = end_year - 1
        try:
            rendimenti_train_vif = prices.loc[str(inizio_vif) : str(fine_vif)].pct_change().dropna()
            rendimenti_macro_train_vif = finale[indicatori_macro].loc[str(inizio_vif) : str(fine_vif)]
            merged_data_vif = pd.concat([rendimenti_train_vif, rendimenti_macro_train_vif], axis=1).dropna()
            data_vif = merged_data_vif.copy()
            data_vif['Stock_Return_Lag1'] = merged_data_vif[tickers[0]].shift(lag_veloce) # Usiamo il primo ticker per il VIF
            data_vif['FEDFUNDS_Diff_Lag1'] = merged_data_vif['FEDFUNDS'].shift(lag_veloce)
            data_vif['UNRATE_Diff_Lag1'] = merged_data_vif['UNRATE'].shift(lag_lento)
            data_vif['CHF=X_Diff_Lag1'] = merged_data_vif['CHF=X'].shift(lag_veloce)
            data_vif['VIX_Diff_Lag1'] = merged_data_vif['^VIX'].shift(lag_veloce)
            data_vif.replace([np.inf, -np.inf], np.nan, inplace=True)
            data_vif = data_vif.dropna()
            
            regressori_vif = ['Stock_Return_Lag1', 'FEDFUNDS_Diff_Lag1', 'UNRATE_Diff_Lag1', 'CHF=X_Diff_Lag1','VIX_Diff_Lag1']
            X_vif = data_vif[regressori_vif]
            X_vif = sm.add_constant(X_vif)
            
            vif_data = pd.DataFrame()
            vif_data["Fattore"] = X_vif.columns
            vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
            vif_data = vif_data[vif_data["Fattore"] != "const"].sort_values(by="VIF", ascending=False).reset_index(drop=True)
            
            def get_vif_status(vif):
                if vif >= 10: return "🚨 PERICOLO (Clone)"
                elif vif >= 5: return "⚠️ ATTENZIONE (Correlato)"
                else: return "✅ OK (Indipendente)"
            vif_data["Stato"] = vif_data["VIF"].apply(get_vif_status)
            
            st.subheader("🔍 TEST MULTICOLLINEARITÀ (VIF)")
            st.dataframe(vif_data)
        except Exception as e:
            st.warning(f"Errore nel calcolo VIF: {e}")

        for anno in range(start_year, end_year):
            fine = anno - 1
            inizio = anno - finestra_train
            
            price_train = prices[tickers].loc[str(inizio) : str(fine)]
            rendimenti_train = prices.loc[str(inizio) : str(fine)].pct_change().dropna()
            rendimenti_test = prices.loc[str(anno)]
            rendimenti_macro_train = finale[indicatori_macro].loc[str(inizio) : str(fine)]
            merged_data = pd.concat([rendimenti_train, rendimenti_macro_train], axis=1).dropna()
            
            previsioni_anno_corrente = {'Anno_Previsione': anno}
            r2_anno = {"Anno": anno}
            
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
                X = data[regressori]
                X = sm.add_constant(X)
                
                try:
                    X_new = pd.DataFrame({
                        "Stock_Return_Lag1":[ultimo_rendimento],
                        "FEDFUNDS_Diff_Lag1":[ultimo_macro["FEDFUNDS"]],
                        "UNRATE_Diff_Lag2":[ultimo_macro["UNRATE"]], # Mantenuto il lag2 come nel tuo codice
                        "CHF=X_Diff_Lag1":[ultimo_macro["CHF=X"]],
                        "VIX_Diff_Lag1":[ultimo_macro["^VIX"]]
                    })
                    X_new = sm.add_constant(X_new, has_constant="add")
                    # Gestione colonne mancanti
                    for col in X.columns:
                        if col not in X_new.columns:
                            X_new[col] = 1.0 if col == 'const' else 0.0
                    X_new = X_new[X.columns]
                    
                    model = sm.OLS(Y, X).fit()
                    prediction = model.predict(X_new).iloc[0]*12
                    r2_anno[variabile_dipendente] = model.rsquared
                    previsioni_anno_corrente[variabile_dipendente] = prediction
                except Exception:
                    previsioni_anno_corrente[variabile_dipendente] = 0.0

            storico_previsioni.append(previsioni_anno_corrente)
            storico_r_quadro.append(r2_anno)

        df_views = pd.DataFrame(storico_previsioni)
        if not df_views.empty:
            df_views = df_views.set_index('Anno_Previsione')

    # ==============================================================================
    # BLACK-LITTERMAN
    # ==============================================================================
    with st.spinner('Calcolo Portafoglio Black-Litterman...'):
        market_caps = {"MSFT": 3200, "GOOGL": 2200, "AMZN": 1900, "JPM": 570,
                       "WMT": 530, "XOM": 460, "JNJ": 360, "CAT": 170,
                       "NEE": 150, "FCX": 75}
        risultati_out_of_sample_bl = []
        delta = 2.5
        
        for anno_test_bl in range(start_year, end_year):
            anno_inizio_train_bl = anno_test_bl - finestra_train
            anno_fine_train_bl = anno_test_bl - 1
            
            price_train_bl = prezzi_asset_mensili.loc[str(anno_inizio_train_bl) : str(anno_fine_train_bl)]
            rendimenti_test_bl = rendimenti_asset_mensili.loc[str(anno_test_bl)]
            
            try:
                S_bl = risk_models.sample_cov(price_train_bl, frequency=12)
                rendimenti_impliciti_mercato = black_litterman.market_implied_prior_returns(market_caps, delta, S_bl)
                
                if anno_test_bl in df_views.index:
                    views_grezze = df_views.loc[anno_test_bl]
                    views_pulite = views_grezze.dropna()
                    views_assolute = (views_pulite / 100).to_dict()
                    
                    bl = BlackLittermanModel(S_bl, pi=rendimenti_impliciti_mercato, absolute_views=views_assolute)
                    rendimenti_bl = bl.bl_returns()
                    S_posteriore = bl.bl_cov()
                    ef_bl = EfficientFrontier(rendimenti_bl, S_posteriore, weight_bounds=(0, 1))
                    pesi_bl_grezzi = ef_bl.max_quadratic_utility(risk_aversion=delta)
                    pesi_bl_puliti = ef_bl.clean_weights()
                    pesi_series = pd.Series(pesi_bl_puliti)
                    
                    crescita_titoli = (1 + rendimenti_test_bl).cumprod()
                    valore_portafoglio = (crescita_titoli * pesi_series).sum(axis=1)
                    rendimenti_portafoglio_bl = valore_portafoglio.pct_change()
                    rendimenti_portafoglio_bl.iloc[0] = valore_portafoglio.iloc[0] - 1
                    risultati_out_of_sample_bl.append(rendimenti_portafoglio_bl)
            except Exception as e:
                pass # Ignora errori specifici per l'anno

        if risultati_out_of_sample_bl:
            rendimenti_mark_bl = pd.concat(risultati_out_of_sample_bl)
            equity_curve_bl = (1 + rendimenti_mark_bl).cumprod()

    # ==============================================================================
    # MOMENTUM
    # ==============================================================================
    with st.spinner('Calcolo Portafoglio Momentum...'):
        risultati_out_of_sample_mom = []
        storico_pesi_mom = []
        top_n = 3
        
        for anno in range(start_year, end_year):
            anno_fine_train = anno - 1
            anno_inizio_train = anno - 2
            try:
                prezzo_inizio = prezzi_asset_mensili.loc[str(anno_inizio_train)].resample('YE').last().iloc[-1]
                prezzo_fine = prezzi_asset_mensili.loc[str(anno_fine_train)].resample('YE').last().iloc[-1]
                rendimento_12m = (prezzo_fine / prezzo_inizio) - 1
                
                titoli_ordinati = rendimento_12m.sort_values(ascending=False)
                vincitori = titoli_ordinati.head(top_n).index
                pesi_mom = {ticker: (1.0 / top_n if ticker in vincitori else 0.0) for ticker in tickers}
                
                price_test = prices.loc[str(anno)]
                rendimenti_test = price_test.pct_change().dropna()
                pesi_series = pd.Series(pesi_mom)
                rendimenti_portafoglio_mom = (rendimenti_test * pesi_series).sum(axis=1)
                
                risultati_out_of_sample_mom.append(rendimenti_portafoglio_mom)
                diz_pesi = {t: p*100 for t, p in pesi_mom.items()}
                diz_pesi['Anno'] = anno
                storico_pesi_mom.append(diz_pesi)
            except Exception:
                pass

        if risultati_out_of_sample_mom:
            rendimenti_totali_mom = pd.concat(risultati_out_of_sample_mom)
            equity_curve_mom = (1 + rendimenti_totali_mom).cumprod().resample('MS').last()
            df_storico_pesi_mom = pd.DataFrame(storico_pesi_mom).set_index('Anno')
            
            st.subheader(f"🔥 Matrice Storica Pesi Momentum (Top {top_n}) (%)")
            st.dataframe(df_storico_pesi_mom.round(2).replace(0.0, '-'))

    # ==============================================================================
    # INVERSE VOLATILITY
    # ==============================================================================
    with st.spinner('Calcolo Portafoglio Inverse Volatility...'):
        risultati_out_of_sample_invvol = []
        for anno_test in range(start_year, end_year):
            anno_inizio_train = anno_test - finestra_train
            anno_fine_train = anno_test - 1
            try:
                rendimenti_train = rendimenti_asset_mensili.loc[str(anno_inizio_train) : str(anno_fine_train)]
                rendimenti_test = rendimenti_asset_mensili.loc[str(anno_test)]
                
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
            except Exception:
                pass
                
        if risultati_out_of_sample_invvol:
            rendimenti_invvol = pd.concat(risultati_out_of_sample_invvol)
            equity_curve_invvol = (1 + rendimenti_invvol).cumprod()

    # ==============================================================================
    # PLOT FINALE (GRAFICO)
    # ==============================================================================
    st.subheader("📊 Confronto Equity Curves")
    
    benchmark_return = benchmark.resample('MS').last().ffill().pct_change()
    equity_curve_benchmark = (1 + benchmark_return).cumprod()
    
    try:
        # Allineamento curve disponibili
        curve_list = [equity_curve_bl, equity_curve_mark, equity_curve_mom, equity_curve_invvol, equity_curve_benchmark]
        df_confronto = pd.concat(curve_list, axis=1)
        df_confronto.columns = ["Black-Litterman", "Markowitz", "Momentum", "Reverse Volatility", "S&P 500"]
        df_confronto = df_confronto.dropna().ffill()

        # Creazione grafico Matplotlib per Streamlit
        fig, ax = plt.subplots(figsize=(15, 8), dpi=120)
        fig.patch.set_facecolor("#1E1E1E")
        ax.set_facecolor("#1E1E1E")
        
        colors = ["#FFD700", "#00FFFF", "#7CFC00", "#FF69B4", "#8A2BE2"]
        
        for i, col in enumerate(df_confronto.columns):
            ax.plot(df_confronto.index, df_confronto[col], lw=2.6, color=colors[i], label=col)
            
        ax.set_title("Equity Curve Comparison", fontsize=22, color="white", fontweight="bold", loc="left", pad=20)
        ax.set_ylabel("Portfolio Value", fontsize=14, color="white")
        ax.set_xlabel("Time", fontsize=14, color="white")
        
        ax.grid(True, which="major", color="#444444", linewidth=0.7)
        ax.tick_params(colors="white", labelsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("white")
        ax.spines["bottom"].set_color("white")
        
        leg = ax.legend(loc="upper left", frameon=False, fontsize=12)
        for text in leg.get_texts():
            text.set_color("white")
            
        st.pyplot(fig) # <- MOSTRAMO IL GRAFICO SU STREAMLIT
        
        # ==============================================================================
        # ANALISI DEGLI EVENTI
        # ==============================================================================
        st.subheader("📉 Analisi Scenari di Crisi")
        eventi = {
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
                if not df_event.empty:
                    for col in df_event.columns:
                        stats = portfolio_stats(df_event[col])
                        stats["Event"] = event
                        stats["Portfolio"] = col
                        results.append(stats)
            except Exception:
                continue
                
        if results:
            df_event_analysis = pd.DataFrame(results)
            # Riordina colonne per una migliore leggibilità
            cols = ['Event', 'Portfolio', 'Return', 'Volatility', 'Sharpe', 'Max Drawdown']
            df_event_analysis = df_event_analysis[cols]
            st.dataframe(df_event_analysis)

    except Exception as e:
        st.error(f"Errore durante la generazione dei grafici: {e}")

else:
    st.info("👈 Seleziona i parametri nella barra laterale e clicca su 'Avvia Backtest Completo'")
