import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar

"""
Abridged from the original code by Florian Bourgey and Jim Gatheral (2025).

Authors: Florian Bourgey, Jim Gatheral
Additional contributions: Anthony Nkyi
Latest edits: April 2026.

Sources: 
https://github.com/jgatheral/QuadraticRoughHeston/black.py

"""

def black_price(K, T, F, vol, r=0.0, opttype=1):
    """
    Calculate the Black option price.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.
    r : float, optional
        Risk-free interest rate. Default is 0.0.
    opttype : int, optional
        Option type: 1 for call options, -1 for put options. Default is 1.

    Returns
    -------
    float
        The Black price of the option.
    """
    w = T * vol**2
    d1 = np.log(F / K) / w**0.5 + 0.5 * w**0.5
    d2 = d1 - w**0.5
    undiscounted_price = opttype * (
        F * norm.cdf(opttype * d1) - K * norm.cdf(opttype * d2)
    )
    return np.exp(-r * T) * undiscounted_price


@np.vectorize
def black_impvol_brentq(K, T, F, value, r=0.0, opttype=1):
    """
    Calculate the Black implied volatility using the Brent's method.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    value : float
        Observed market price of the option.
    r : float, optional
        Risk-free interest rate. Default is 0.0.
    opttype : int, optional
        Option type: 1 for call options, -1 for put options. Default is 1.

    Returns
    -------
    float
        Implied volatility corresponding to the input option price. Returns NaN
        if the implied volatility does not converge or if invalid inputs are provided.
    """
    if (K <= 0) or (T <= 0) or (F <= 0) or (value <= 0):
        return np.nan

    # Check intrinsic value bounds
    intrinsic = np.exp(-r * T) * np.maximum(opttype * (F - K), 0)
    if value < intrinsic * 0.999:  # Small tolerance for numerical errors
        return np.nan

    try:
        result = root_scalar(
            f=lambda vol: black_price(K, T, F, vol, r, opttype) - value,
            bracket=[1e-10, 5.0],
            method="brentq",
            xtol=1e-9,
        )
        return result.root if result.converged else np.nan
    except ValueError:
        return np.nan


def black_impvol(K, T, F, value, r=0.0, opttype=1, TOL=1e-6, MAX_ITER=10000):
    """
    Calculate the Black implied volatility using a bisection method.

    Parameters
    ----------
    K : ndarray or float
        Strike price(s) of the option(s).
    T : float
        Time to maturity of the option(s).
    F : float
        Forward price of the underlying asset.
    value : ndarray or float
        Observed market price(s) of the option(s).
    r : float, optional
        Risk-free interest rate. Default is 0.0.
    opttype : int or ndarray, optional
        Option type: 1 for call options, -1 for put options. Default is 1.
    TOL : float, optional
        Tolerance for convergence of the implied volatility. Default is 1e-6.
    MAX_ITER : int, optional
        Maximum number of iterations for the bisection method. Default is 1000.

    Returns
    -------
    ndarray or float
        Implied volatility(ies) corresponding to the input option prices. If the
        input arrays are multidimensional, the output will have the same shape.
        Returns NaN if the implied volatility does not converge or if invalid
        inputs are provided.

    Raises
    ------
    ValueError
        If `K` and `value` do not have the same shape.
        If `opttype` is not 1 or -1.
        If the implied volatility does not converge within `MAX_ITER` iterations.
    """
    K = np.atleast_1d(K)
    value = np.atleast_1d(value)
    opttype = np.full_like(K, opttype)

    if K.shape != value.shape:
        raise ValueError(f"K ({K.shape}) and value ({value.shape}) must have the same shape.")

    # Fixed: Correct validation of opttype
    if not np.all(np.abs(opttype) == 1):
        raise ValueError("opttype must be either 1 or -1.")

    F = float(F)
    T = float(T)
    r = float(r)

    if T <= 0 or F <= 0:
        return np.full_like(K, np.nan)

    # Check for invalid strikes or values
    invalid_mask = (K <= 0) | (value <= 0)
    
    # Check intrinsic value bounds
    intrinsic = np.exp(-r * T) * np.maximum(opttype * (F - K), 0)
    invalid_mask |= (value < intrinsic * 0.999)  # Small tolerance

    low = 1e-10 * np.ones_like(K, dtype=float)
    high = 5.0 * np.ones_like(K, dtype=float)
    mid = 0.5 * (low + high)
    
    for _ in range(MAX_ITER):
        price = black_price(K, T, F, mid, r, opttype)
        diff = price - value
        
        # Use combined absolute and relative error for better convergence
        abs_error = np.abs(diff)
        rel_error = np.abs(diff / np.maximum(value, 1e-10))
        converged = (abs_error < TOL * 0.01) | (rel_error < TOL)
        
        if np.all(converged | invalid_mask):
            result = mid.copy()
            result[invalid_mask] = np.nan
            return result

        mask = diff > 0
        high[mask] = mid[mask]
        low[~mask] = mid[~mask]
        mid = 0.5 * (low + high)

    # Return NaN for non-converged values instead of raising
    result = mid.copy()
    result[~converged] = np.nan
    result[invalid_mask] = np.nan
    return result


def black_otm_impvol_mc(S, k, T, risk_free_rate=0.0, q=0.0, mc_error=False, opttype=None):
    """
    Calculate Black implied volatility using Monte Carlo simulated stock prices and
    out-of-the-money (OTM) prices.

    Parameters
    ----------
    S : ndarray
        Array of Monte Carlo simulated terminal stock prices (at time T).
        Must already be multiplied by spot price S_0.
    k : float or ndarray
        Log-Forward Moneyness `k=log(K/F)` for which the implied volatility is
        calculated.
    T : float
        Time to maturity of the option.
    risk_free_rate : float, optional
        Risk-free interest rate. Default is 0.0.
    q : float, optional
        Dividend yield. Default is 0.0.
    mc_error : bool, optional
        If True, computes the 95% confidence interval for the implied volatility.
    opttype : int or ndarray, optional
        Option type: 1 for call, -1 for put. If None, automatically determines
        OTM options (calls for K > F, puts for K < F).

    Returns
    -------
    dict or ndarray
        If `mc_error` is False, returns an ndarray of OTM implied volatilities.
        If `mc_error` is True, returns a dictionary with the following keys:
        - 'otm_impvol': ndarray of OTM implied volatilities.
        - 'otm_impvol_high': ndarray of upper bounds of the 95% confidence interval.
        - 'otm_impvol_low': ndarray of lower bounds of the 95% confidence interval.
        - 'error_95': ndarray of the 95% confidence interval errors for the option
                      prices.
        - 'otm_price': ndarray of the calculated OTM option prices.
    """
    k = np.atleast_1d(np.asarray(k))
    F = np.mean(S)
    K = F * np.exp(k)
    rate = risk_free_rate - q
    
    # Determine OTM option types: calls (+1) for K > F, puts (-1) for K < F
    if opttype is None:
        opttype = np.where(K >= F, 1, -1)
    else:
        opttype = np.atleast_1d(opttype)
    
    # Calculate payoffs and option prices
    payoff = np.maximum(opttype[None, :] * (S[:, None] - K[None, :]), 0.0)
    otm_price = np.mean(payoff, axis=0) * np.exp(-rate * T)
    
    # Calculate implied volatility
    otm_impvol = black_impvol(K=K, T=T, F=F, value=otm_price, opttype=opttype, r=rate)

    if mc_error:
        error_95 = 1.96 * np.std(payoff, axis=0) / S.shape[0] ** 0.5 * np.exp(-rate * T)
        otm_impvol_high = black_impvol(
            K=K, T=T, F=F, value=otm_price + error_95, opttype=opttype, r=rate
        )
        otm_impvol_low = black_impvol(
            K=K, T=T, F=F, value=otm_price - error_95, opttype=opttype, r=rate
        )
        return {
            "otm_impvol": otm_impvol,
            "otm_impvol_high": otm_impvol_high,
            "otm_impvol_low": otm_impvol_low,
            "error_95": error_95,
            "otm_price": otm_price,
        }

    return otm_impvol