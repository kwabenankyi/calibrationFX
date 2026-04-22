import numpy as np
from scipy.stats import norm

"""
Black-Scholes pricing functions for European call and put options, and Greeks.

Anthony Nkyi, April 2026.
"""

def value_check(S_0, K, tau, sigma):
    if np.any(S_0) < 0.0:
        return ValueError("S_0 cannot be less than zero.")
    if np.any(K) <= 0.0:
        return ValueError("K is strictly positive.")
    if np.any(tau) <= 0.0:
        return ValueError("Tau is strictly positive.")
    if np.any(sigma) <= 0.0:
        return ValueError("Sigma is strictly positive.")
    return True

def d1_value(S_0: float, K: np.ndarray, sigma: float, tau: float, r: float, q: float = 0.0):
    return (np.log(S_0 / K) + (r - q + 0.5*sigma**2)*tau) / (sigma * np.sqrt(tau))

def d1_value_log_fwd(K: np.ndarray, sigma: float, tau: float):
    return (- K + 0.5*sigma**2*tau) / (sigma * np.sqrt(tau))

def price_option(CP, S_0: float, K, tau: float, r: float, vol):
    """
    Pricing European call/put options using the Black-Scholes equation with no dividends.
    Vectorised: CP, K, and vol can be arrays.

    Parameters
    ----------
    CP : float / np.ndarray
        Call (+1) / Put (-1), scalar or array.
    S_0 : float
        Spot price of the underlying asset.
    K : float / np.ndarray
        Strike price(s) of the options.
    tau : float
        Time to maturity of the options.
    r : float
        Risk-free rate.
    vol : float / np.ndarray
        Implied volatility for the options.
    """
    CP  = np.asarray(CP, dtype=float)
    K   = np.asarray(K,  dtype=float)
    vol = np.asarray(vol, dtype=float)

    assert value_check(S_0, K, tau, vol)

    d1 = d1_value(S_0, K, vol, tau, r)
    d2 = d1 - vol * np.sqrt(tau)

    disc = np.exp(-r * tau)

    price = CP * (
        S_0 * norm.cdf(CP * d1)
        - K * disc * norm.cdf(CP * d2)
    )

    return price if price.size > 1 else float(price)

def price_fx_option(CP, S_0: float, K, tau: float, r_base: float, r_term: float, vol):
    """
    Pricing a European call/put FX option using the Black-Scholes equation. 
    
    Parameters
    ----------
    CP : int
        Call (+1) / Put (-1)
    S_0 : float
        Spot price of the underlying asset.
    K : float / array
        Strike price(s) of the options.
    tau : float
        Time to maturity of the options.
    r_base : float
        Base risk-free rate.
    r_term : float
        Term risk-free rate.
    vol : float / array
        Implied volatility for the options.

    Returns
    -------
    value : float / array
    """

    assert value_check(S_0, K, tau, vol)

    d1 = d1_value(S_0, K, vol, tau, r_term - r_base)
    d2 = d1 - vol * np.sqrt(tau)

    price = CP * (
        S_0 * np.exp(-r_base * tau) * norm.cdf(CP * d1)
        - K * np.exp(-r_term * tau) * norm.cdf(CP * d2)
    )
    return price

def price_option_mc(CP: np.ndarray, S_0: float, X_paths: np.ndarray, K, tau: float, r: float):
    """
    Pricing a European call/put option using Monte Carlo simulation.
    
    Parameters
    ----------
    CP : float / np.ndarray
        Call (+1) / Put (-1)
    S_0 : float
        Spot price of the underlying asset.
    X_paths : np.ndarray
        Simulated log-asset price paths. Should be one dimensional.
    K : float
        Strike price of the option.
    tau : float
        Time to maturity of the option.
    r : float
        Risk-free rate.
    
    Returns
    -------
    float
        Estimated option price.
    """
    K = np.asarray(K).reshape(-1, 1)       # (n_strikes, 1)
    CP = np.asarray(CP).reshape(-1, 1)     # (n_strikes, 1)
    # Price calc: spot * exp (monte carlo paths)
    S_T = S_0 * np.exp(X_paths)
    prices_grid = np.broadcast_to(S_T, (K.size, *S_T.shape))
    CP = np.asarray(CP).reshape(-1, 1)

    # Payoff logic: call payoff = max (Fwd price - K, 0) // put payoff = max (K - Fwd price, 0)
    payoffs = np.maximum(CP * (prices_grid - K), 0)
    option_prices = np.mean(payoffs, axis=1) * np.exp(-r * tau)

    # Return scalar if single strike, otherwise array
    return option_prices if option_prices.size > 1 else option_prices.item()

def price_option_mc_log_fwd(CP: np.ndarray, S_0: float, X_paths: np.ndarray, k_log_fwd, F: float, tau: float, r: float):
    """
    Pricing a European call/put option using Monte Carlo simulation
    with log forward moneyness.
    
    Parameters
    ----------
    CP : np.ndarray
        Call (+1) / Put (-1) flag array.
    S_0 : float
        Spot price of the underlying asset.
    X_paths : np.ndarray
        Simulated log-asset price paths (log returns). Should be 1D.
    k_log_fwd : np.ndarray or float
        Log forward moneyness: ln(K / F).
    F : float
        Forward price of the asset for the given maturity.
    tau : float
        Time to maturity of the option.
    r : float
        Risk-free (domestic) discount rate.
    
    Returns
    -------
    float or np.ndarray
        Estimated option price(s).
    """
    k = np.asarray(k_log_fwd).reshape(-1, 1)  # (n_strikes, 1)
    CP = np.asarray(CP).reshape(-1, 1)        # (n_strikes, 1)
    
    # 1. Recover raw absolute strikes from log forward moneyness
    K = F * np.exp(k)
    
    # 2. Calculate terminal spot prices from the simulated log-returns
    # S_T = S_0 * exp(X)
    S_T = S_0 * np.exp(X_paths)
    
    S_T_grid = np.broadcast_to(S_T, (K.size, *S_T.shape))
    payoffs = np.maximum(CP * (S_T_grid - K), 0)
    option_prices = np.mean(payoffs, axis=1) * np.exp(-r * tau)

    # Return scalar if single strike, otherwise array
    return option_prices if option_prices.size > 1 else option_prices.item()

def option_vega(S_0: float, K: np.ndarray, tau: float, r: float, q: float, sigma: float):
    """
    Calculating vega: rate of change of option price to volatility change (DV/Dsigma) for a European call/put option using the Black-Scholes equation. K should be in vector form: shape (n,1)
    """
    return S_0 * norm.pdf(d1_value(S_0, K, sigma, tau, r-q)) * np.sqrt(tau)

def option_vega_log_fwd(S_0: float, k_grid: np.ndarray, tau: float, r: float, q: float, sigma: float):
    """
    Vega calculation using log forward moneyness.
    """
    return S_0 * norm.pdf(d1_value_log_fwd(k_grid, sigma, tau)) * np.sqrt(tau)