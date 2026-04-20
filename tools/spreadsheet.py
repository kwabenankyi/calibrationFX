import numpy as np
import pandas as pd

def read_spreadsheet(file_path):
    return pd.read_excel(file_path)

def get_vol_quotes(file_path):
    df = read_spreadsheet(file_path)
    col_order = np.array([21, 17, 13, 9, 5, 1, 3, 7, 11, 15, 19])
    # Reorder columns in mid_vol_quotes and spread_quotes according to col_order: OTM put to ATM put/call to OTM call
    mid_vol_quotes_df = df.iloc[:, col_order]
    spread_quotes_df = df.iloc[:, col_order+1]

    vol_days = df.iloc[2:28, 0]

    mid_vol_quotes = {}
    spread_quotes = {}
    bid_quotes = {}
    ask_quotes = {}

    count = 2
    for day in vol_days:
        # get quotes in same row as day and rearrange in col_order
        mid_vol_quotes[day] = np.array(mid_vol_quotes_df.loc[count].values.flatten()) / 100
        spread_quotes[day] = np.array(spread_quotes_df.loc[count].values.flatten()) / 100
        bid_quotes[day] = mid_vol_quotes[day] - spread_quotes[day]/2
        ask_quotes[day] = mid_vol_quotes[day] + spread_quotes[day]/2
        count += 1

    return {"mid": mid_vol_quotes, "bid": bid_quotes, "ask": ask_quotes, "spread": spread_quotes, "days": vol_days.values}

def get_usd_interest_rates(file_path, date_str):
    date = pd.to_datetime(date_str, format="%d-%m-%Y")
    df = read_spreadsheet(file_path)
    col_order = np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18, 19, 20, 21, 22, 25, 28, 29, 30, 31])
    rates = [df[df.iloc[:, 0] == date].iloc[0, col_order].values.flatten() / 100]
    return np.array(rates)

def get_euro_interest_rates(file_path, date_str):
    date = pd.to_datetime(date_str, format="%d-%m-%Y")
    df = read_spreadsheet(file_path)
    col_order = np.array([1] + [i for i in range (1, 24)])
    rates = [df[df.iloc[:, 0] == date].iloc[0, col_order].values.flatten() / 100]
    return np.array(rates)

def get_fx_spot(file_path, date_str):
    date = pd.to_datetime(date_str, format="%d-%m-%Y")
    df = read_spreadsheet(file_path)
    spot = df[df.iloc[:, 0] == date].iloc[0, 1]
    return spot

if __name__ == "__main__":
    file_path = "..data/Dec_01.xlsx"
    vol_quotes = get_vol_quotes(file_path)
    print(vol_quotes)