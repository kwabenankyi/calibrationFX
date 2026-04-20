import numpy as np
from scipy.stats import norm

"""
Converting FX spot delta IV quotes to strike equivalents under the Black-Scholes pricing model.

Anthony Nkyi, March 2026.
"""

def delta_to_strike(call_put, delta, vol, tau, spot, r_base, r_term):
    """
    Converting a FX spot delta IV quote to its strike K equivalent under the Black-Scholes pricing model.

    """
    arg = np.where(
        tau < 1.0, 
        np.exp(r_base * tau) * call_put * delta, 
        call_put * delta
    )

    arg = np.clip(arg, 1e-10, 1.0 - 1e-10)

    K = spot * np.exp(
        -call_put * norm.ppf(arg) * vol * np.sqrt(tau)
        + (r_term - r_base + 0.5 * vol**2) * tau
    )
    return K

def strike_to_delta(call_put, K, vol, tau, spot, r_base, r_term):
    """
    Converting a strike K to its FX spot/forward delta equivalent under the Black-Scholes pricing model.
    Inverse of delta_to_strike.
    """
    d1 = (
        (r_term - r_base + 0.5 * vol**2) * tau - np.log(K / spot)
    ) / (call_put * vol * np.sqrt(tau))

    arg = norm.cdf(d1)

    if tau < 1.0:
        # Spot Delta
        delta = call_put * np.exp(-r_base * tau) * arg
    else:
        # Forward Delta convention
        delta = call_put * np.exp(-r_term * tau) * arg
    return delta
