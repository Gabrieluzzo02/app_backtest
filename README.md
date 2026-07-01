# Backtesting App

Prototipo Streamlit per selezionare asset da un universo investibile, scaricare prezzi da Yahoo Finance e testare modelli di portafoglio.

## Funzionalità incluse

- Ricerca assistita degli asset per nome/ticker.
- Inserimento rapido di nomi separati da virgola, ad esempio `Apple, Microsoft, Enel`.
- Menu con filtri per settore, paese e area mercato.
- Lista finale `selected_tickers` usata dal motore di backtesting.
- Download prezzi via `yfinance`.
- Calcolo rendimenti.
- Markowitz minimum variance.
- Equal Weight.
- Inverse Volatility.
- Rolling Markowitz out-of-sample.
- Equity curve e metriche base.

## Struttura progetto

```text
backtesting-app/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── assets_universe.csv
│
├── core/
│   ├── asset_search.py
│   ├── backtest.py
│   ├── data_loader.py
│   ├── metrics.py
│   ├── optimization.py
│   ├── regression.py
│   └── returns.py
│
└── ui/
    └── asset_selector.py
```

## Avvio locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy su Streamlit Community Cloud

1. Crea un repository GitHub.
2. Carica tutti i file di questa cartella.
3. Vai su Streamlit Community Cloud.
4. Seleziona il repository.
5. Imposta come main file:

```text
app.py
```

## Come funziona la selezione ticker

La UI restituisce una lista Python:

```python
selected_tickers = ["AAPL", "MSFT", "ENEL.MI"]
```

Questa lista viene passata direttamente al download prezzi:

```python
prices = download_prices(selected_tickers, start_date, end_date)
```

Quindi il motore quantitativo non deve conoscere la logica dell'interfaccia: riceve solo una lista di ticker.

## Nota

Il file `data/assets_universe.csv` deriva dall'universo investibile che avevi già integrato nello script Colab. Alcuni ticker europei sono ADR o ticker Yahoo/USA: per una versione più professionale puoi sostituirli progressivamente con ticker locali Yahoo Finance.
