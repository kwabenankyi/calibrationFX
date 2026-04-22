import numpy as np
from tools.BlackScholes import price_option_mc_log_fwd, price_option_mc, price_fx_option, option_vega, option_vega_log_fwd
from tools.DeltaFx import delta_to_strike
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from tools.gatheral_black import black_otm_impvol_mc, black_impvol_brentq
import pandas as pd

"""
Grid construction and related functions for financial option pricing and implied volatility calculations.
Setup function generates a grid of strikes and prices from given volatility quotes and market data and 
initialises the necessary data structures for further analysis.

Anthony Nkyi, March 2026.
"""

DELTA = np.array([-0.05, -0.1, -0.15, -0.25, -0.35, -0.5, 0.35, 0.25, 0.15, 0.1, 0.05])

# Pricing grid functions
def get_mc_prices_grid(mc_paths_matrix, spot, K_grid, T_arr, r_b_arr, r_t_arr, cp_flags):
    n_expiries, _ = K_grid.shape
    prices_list = [] 

    for i in range(n_expiries):
        p = price_option_mc(cp_flags[i], spot, mc_paths_matrix[i], K_grid[i], T_arr[i], r_t_arr[i])
        p_atm = price_option_mc(np.array([1]), spot, mc_paths_matrix[i], K_grid[i][1], T_arr[i], r_t_arr[i])
        # atm calc
        p[5] = (p[5] + p_atm) / 2
        prices_list.append(p)
    
    return np.array(prices_list)

def get_mc_prices_grid_log_fwd(mc_paths_matrix, spot, log_fwd_moneyness_grid, T_arr, r_b_arr, r_t_arr, cp_flags, fwd_list):
    n_expiries, _ = log_fwd_moneyness_grid.shape
    prices_list = [] 

    # get central index of array of prices
    central_idx = log_fwd_moneyness_grid.shape[1] // 2

    for i in range(n_expiries):
        p = price_option_mc_log_fwd(cp_flags[i], spot, mc_paths_matrix[i], log_fwd_moneyness_grid[i], fwd_list[i], T_arr[i], r_t_arr[i])
        p_atm = price_option_mc_log_fwd(-1 * np.array([cp_flags[i, central_idx]]), spot, mc_paths_matrix[i], log_fwd_moneyness_grid[i, central_idx], fwd_list[i], T_arr[i], r_t_arr[i])
        p[central_idx] = (p[central_idx] + p_atm) / 2
        prices_list.append(p)
    
    return np.array(prices_list)

# Vega functions
def build_vega_grid(spot, K_grid, T_arr, r_b_arr, r_t_arr, vol_grid):
    n_expiries, n_strikes = K_grid.shape
    vega_mat = np.zeros((n_expiries, n_strikes))

    for i in range(n_expiries):
        vega_mat[i, :] = option_vega(spot, K_grid[i, :], T_arr[i], r_t_arr[i], r_b_arr[i], vol_grid[i, :])
    
    return vega_mat

def build_vega_grid_log_fwd(spot, k_grid, T_arr, r_b_arr, r_t_arr, vol_grid):
    n_expiries, n_strikes = k_grid.shape
    vega_mat = np.zeros((n_expiries, n_strikes))

    for i in range(n_expiries):
        vega_mat[i, :] = option_vega_log_fwd(spot, k_grid[i, :], T_arr[i], r_t_arr[i], r_b_arr[i], vol_grid[i, :])
    
    return vega_mat

# Implied vol functions
def get_iv_from_prices_grid_jaeckel(price_grid, spot, K_grid, T_arr, r_b_arr, r_t_arr, cp_flags, tol=1e-6):
    n_expiries, n_strikes = K_grid.shape
    imp_vols = np.zeros((n_expiries, n_strikes))
    flags = np.where(cp_flags > 0, 'c', 'p')
    for i in range(n_expiries):
        for j in range(n_strikes):
            imp_vols[i, j] = implied_volatility(price_grid[i,j], spot, K_grid[i,j], T_arr[i], r_t_arr[i], r_b_arr[i], flags[i,j])
        imp_vols[i, 5] = (imp_vols[i, 5] + implied_volatility(price_grid[i,5], spot, K_grid[i,5], T_arr[i], r_t_arr[i], r_b_arr[i], 'c')) / 2
    return imp_vols

def get_iv_from_paths_grid_gatheral(mc_paths_matrix, spot, lfw_grid, T_arr, r_b_arr, r_t_arr, cp_flags):
    n_expiries, n_strikes = lfw_grid.shape
    imp_vols = np.zeros((n_expiries, n_strikes))

    for i in range(n_expiries):
        sim_prices = spot * np.exp(mc_paths_matrix[i])
        imp_vols[i, :] = black_otm_impvol_mc(sim_prices, lfw_grid[i, :], T_arr[i], risk_free_rate=r_t_arr[i], opttype=cp_flags[i, :], q=r_b_arr[i])
    
    return imp_vols

def setup(volatility_grid, expiries, domestic_rate, foreign_rate, forward_prices, spot, data_expiries_path, verbose=False):
    fx_df = pd.DataFrame(columns=['Texp', 'Delta', 'Mid', 'BaseYield', 'TermYield', 'Strike', 'Fwd', 'LogFwdMoneyness', 'Price'])

    vol_size = volatility_grid.shape

    log_fwd_moneyness_grid = []
    strike_grid = []
    price_grid = []

    rates_base_list = []
    rates_term_list = []
    cp_flags_grid = [] # 1.0 for Call, -1.0 for Put

    DELTA_CALC = np.array([-0.05, -0.1, -0.15, -0.25, -0.35, -0.5, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05])
    call_put = np.where(DELTA > 0, 1, -1)
    call_put_calc = np.where(DELTA_CALC > 0, 1, -1)
    call_put_calc_str = np.where(DELTA_CALC > 0, 'c', 'p')

    atm_strikes = []
    atm_prices = []

    VOL_QUOTES_CALC = np.ones((vol_size[0], len(DELTA_CALC)))
    VOL_QUOTES_CALC[:,:5] = volatility_grid[:,:5]
    VOL_QUOTES_CALC[:,5] = volatility_grid[:,5]
    VOL_QUOTES_CALC[:,6] = volatility_grid[:,5]
    VOL_QUOTES_CALC[:,7:] = volatility_grid[:,6:]

    inverse_vol_grid = []
    inverse_vol_grid_gatheral = []
    # generating strike grid from deltas and vol quotes
    for i in range(vol_size[0]):
        current_tau = expiries[i]
        current_base = domestic_rate[i]
        current_term = foreign_rate[i]
        fwd = forward_prices[i]

        vol_quotes_calc = np.ones_like(DELTA_CALC)
        vol_quotes_calc[:5] = volatility_grid[i, :5]
        vol_quotes_calc[5] = volatility_grid[i, 5]
        vol_quotes_calc[6] = volatility_grid[i, 5]
        vol_quotes_calc[7:] = volatility_grid[i, 6:]

        strike_list = delta_to_strike(call_put_calc, DELTA_CALC, vol_quotes_calc, current_tau, spot, current_base, current_term)

        lfm = []
        inverse_vol_list = []
        inverse_vol_list_gatheral = []

        price_list = price_fx_option(call_put_calc, spot, strike_list, current_tau, current_base, current_term, vol_quotes_calc)
        for z in range (len(price_list)):
            inverse_vol_list.append(implied_volatility(price_list[z], spot, strike_list[z], 
                                                       current_tau, current_term, current_base, call_put_calc_str[z]))
            inverse_vol_list_gatheral.append(black_impvol_brentq(strike_list[z], current_tau, fwd, price_list[z], 
                                                                 r=current_term-current_base, opttype=call_put_calc[z]))

        inverse_vol_grid.append(inverse_vol_list)
        inverse_vol_grid_gatheral.append(inverse_vol_list_gatheral)
        max_deviation = np.max(vol_quotes_calc - inverse_vol_list)
        max_deviation_gatheral = np.max(vol_quotes_calc - inverse_vol_list_gatheral)
        rms_deviation = np.sqrt(np.mean((vol_quotes_calc - inverse_vol_list)**2))
        rms_deviation_gatheral = np.sqrt(np.mean((vol_quotes_calc - inverse_vol_list_gatheral)**2))
        if verbose:
            print(f"Py_vollib Max Deviation: {max_deviation:.6e}, RMS Deviation: {rms_deviation:.6e}")
            print(f"Gatheral Max Deviation : {max_deviation_gatheral:.6e}, RMS Deviation: {rms_deviation_gatheral:.6e}")

        # ATM value is average of middle two deltas
        atm_price = (price_list[6] + price_list[5]) / 2
        # new list of strikes / prices with the ATM at the centre
        strikes = np.ones_like(DELTA)
        strikes[5] = (strike_list[6] + strike_list[5]) / 2
        strikes[:5] = strike_list[:5]
        strikes[6:] = strike_list[7:]
        strike_grid.append(strikes)

        atm_prices.append(atm_price)
        atm_strikes.append([strike_list[5], strike_list[6]])
        prices = np.zeros_like(DELTA)
        prices[5] = atm_price
        prices[:5] = price_list[:5]
        prices[6:] = price_list[7:]
        price_grid.append(prices)

        for j in range(len(strikes)):
            log_forward_moneyness = np.log(strikes[j] / fwd)
            lfm.append(log_forward_moneyness)
            fx_df.loc[len(fx_df)] = [current_tau, DELTA[j], volatility_grid[i, j], current_base, current_term, 
                                     strikes[j], fwd, log_forward_moneyness, prices[j]]
        
        log_fwd_moneyness_grid.append(lfm)
        rates_base_list.append(current_base)
        rates_term_list.append(current_term)
        cp_flags_grid.append(call_put)
        
    strike_grid = np.array(strike_grid)
    price_grid = np.array(price_grid)
    log_fwd_moneyness_grid = np.array(log_fwd_moneyness_grid)
    inverse_vol_grid = np.array(inverse_vol_grid)

    np.savetxt(f'{data_expiries_path}strike_grid.csv', strike_grid, delimiter=',')
    np.savetxt(f'{data_expiries_path}price_grid.csv', price_grid, delimiter=',')
    np.savetxt(f'{data_expiries_path}log_fwd_moneyness_grid.csv', log_fwd_moneyness_grid, delimiter=',')
    np.savetxt(f'{data_expiries_path}inverse_vol_grid.csv', inverse_vol_grid, delimiter=',')

    base_rates_arr = np.array(rates_base_list)
    term_rates_arr = np.array(rates_term_list)
    cp_flags_grid = np.array(cp_flags_grid)

    return {
        "fx_df": fx_df,
        "strike_grid": strike_grid,
        "price_grid": price_grid,
        "log_fwd_moneyness_grid": log_fwd_moneyness_grid,
        "inverse_vol_grid": inverse_vol_grid,
        "base_rates_arr": base_rates_arr,
        "term_rates_arr": term_rates_arr,
        "cp_flags_grid": cp_flags_grid,
    }