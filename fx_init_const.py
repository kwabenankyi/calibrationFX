import numpy as np
import tools.fx_data as fx_data

TAU_STR = ['1D', '1W', '2W', '3W', '1M', '2M', '3M', '4M', '5M', '6M','9M', '1Y', '18M', '2Y', '3Y', '4Y', '5Y']
TAU_TICKS = ['','','','','','','','','','','', '1Y', '18M', '2Y', '3Y', '4Y', '5Y']

# 17 expiries
ROWS = 3
COLS = 6

DELTA = np.array([-0.05, -0.1, -0.15, -0.25, -0.35, -0.5, 0.35, 0.25, 0.15, 0.1, 0.05])

spot = fx_data.spot

t = fx_data.getTau(TAU_STR)
TAU = t["tau"].ravel()
TAU_DAYS = t["tau_days"].ravel()

ois = fx_data.getOISRates(TAU_STR)
USD_OIS = ois["USD_OIS"].ravel()
EUR_OIS = ois["EUR_OIS"].ravel()

# continuous compounding of rates rather than simple interest
USD_OIS = np.log(1 + USD_OIS * TAU) / TAU
EUR_OIS = np.log(1 + EUR_OIS * TAU) / TAU

COMBINED_OIS = (USD_OIS - EUR_OIS)

VOL_QUOTES = fx_data.getVolQuotes(TAU_STR)

FWD = spot * np.exp(COMBINED_OIS * TAU)

PATHS = 10000
STEPS = 365 * 5 + 1