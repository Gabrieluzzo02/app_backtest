def names_to_tickers(company_names, assets):
    tickers = []
    not_found = []

    for company in company_names:
        query = company.lower().strip()

        matches = assets[
            assets["name"].str.lower().str.contains(query, na=False)
            | assets["ticker"].str.lower().str.contains(query, na=False)
            | assets["yahoo_ticker"].str.lower().str.contains(query, na=False)
        ]

        if len(matches) == 0:
            not_found.append(company)
        else:
            ticker = matches.iloc[0]["yahoo_ticker"]
            tickers.append(ticker)

    return tickers, not_found
