# date = "01-12-2023"
import numpy as np
import tools.spreadsheet as ts

date_str = "01-12-2023"

TAU = np.array([1/365, 1/52, 2/52, 3/52, 1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 9/12, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30])
TAU_DAYS = np.array([1, 7, 14, 21, 30, 60, 90, 120, 150, 180, 270, 365, 365+180, 365*2, 365*3, 365*4+1, 365*5+1, 365*7+1, 365*10+2, 365*15+3, 365*20+5, 365*25+6, 365*30+7])

days_to_tau = {days: tau for tau, days in zip(TAU, TAU_DAYS)}

spot = float(ts.get_fx_spot("data/EURUSD_SpotExchangeRate_2023.xlsx", date_str))

all_quotes = ts.get_vol_quotes("data/Dec_01.xlsx")
vol_quotes = all_quotes["mid"]
bid_quotes = all_quotes["bid"]
ask_quotes = all_quotes["ask"]
spread_quotes = all_quotes["spread"]
days = all_quotes["days"]

tau_days_to_ticks = {
    1: '1D',
    7: '1W',
    14: '2W',
    21: '3W',
    30: '1M',
    60: '2M',
    90: '3M',
    120: '4M',
    150: '5M',
    180: '6M',
    270: '9M',
    365: '1Y',
    365+180: '18M',
    365*2: '2Y',
    365*3: '3Y',
    365*4+1: '4Y',
    365*5+1: '5Y',
    365*7+1: '7Y',
    365*10+2: '10Y',
    365*15+3: '15Y',
    365*20+5: '20Y',
    365*25+6: '25Y',
    365*30+7: '30Y'
}
tau_ticks_to_days = {v: k for k, v in tau_days_to_ticks.items()}

#                    1D,     1W,     2W,     3W,     1M,     2M,     3M,     4M,     5M,     6M,     9M,     1Y,   18M,     2Y,    3Y,    4Y,      5Y,      7Y,    10Y,   15Y,     20Y,    25Y,    30Y
USD_OIS = ts.get_usd_interest_rates("data/US_OIS_2023.xlsx", date_str)
EUR_OIS = ts.get_euro_interest_rates("data/EUR_OIS_ESTR_2023.xlsx", date_str)

def get_volatility_quotes(tau_ticks):
    vols = []
    for tick in tau_ticks:
        vols.append(vol_quotes[tick])
    return np.array(vols, dtype=np.float64)

def get_bid_quotes(tau_ticks):
    bids = []
    for tick in tau_ticks:
        bids.append(bid_quotes[tick])
    return np.array(bids, dtype=np.float64)

def get_ask_quotes(tau_ticks):
    asks = []
    for tick in tau_ticks:
        asks.append(ask_quotes[tick])
    return np.array(asks, dtype=np.float64)

def get_volatility_spreads(tau_ticks):
    spreads = []
    for tick in tau_ticks:
        spreads.append(spread_quotes[tick])
    return np.array(spreads, dtype=np.float64)

def get_ois_rates(tau_ticks):
    usd_rates = []
    eur_rates = []
    for tick in tau_ticks:
        idx = list(tau_ticks_to_days.keys()).index(tick)
        usd_rates.append(USD_OIS[:,idx])
        eur_rates.append(EUR_OIS[:,idx])
    return {"USD_OIS": np.array(usd_rates, dtype=np.float64), "EUR_OIS": np.array(eur_rates, dtype=np.float64)}

def get_tau(tau_ticks):
    taus = []
    tau_days = []
    for tick in tau_ticks:
        days = tau_ticks_to_days[tick]
        tau_days.append(days)
        taus.append(days_to_tau[days])
    return {"tau": np.array(taus, dtype=np.float64), "tau_days": np.array(tau_days, dtype=np.int32)}