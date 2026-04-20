import numpy as np
from scipy.optimize import minimize
from scipy.linalg import inv

from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm
from scipy.special import gamma, gammainc

"""
Abridged from the original code by Florian Bourgey and Jim Gatheral (2025).

Original code contributions: solve_for_xi

Authors: Florian Bourgey, Jim Gatheral
Additional contributions: Anthony Nkyi
Latest edits: April 2026.

Sources: 
https://github.com/jgatheral/QuadraticRoughHeston/blob/main/fwd_var_curve.py
https://github.com/jgatheral/QuadraticRoughHeston/blob/main/utils.py
"""

def var_swap_robust(ivol_data, exp_dates, verbose=True):
    """Robust estimation of variance swap quotes."""
    ivol_data = ivol_data.dropna()
    n_slices = len(exp_dates)

    vs_mid = np.zeros(n_slices)

    def varswap(k_in, vol_series, slice_idx):
        t = exp_dates[slice_idx]
        # Convert to numpy array to avoid indexing issues
        sig_in = vol_series.values * np.sqrt(t)
        k_in = np.asarray(k_in).flatten()
        
        zm_in = -k_in / sig_in - sig_in / 2
        y_in = norm.cdf(zm_in)
        ord_y_in = np.argsort(y_in)
        sig_in_y = sig_in[ord_y_in]
        y_min = np.min(y_in)
        y_max = np.max(y_in)
        sig_in_0 = sig_in_y[0]
        sig_in_1 = sig_in_y[-1]

        wbar_flat = quad(PchipInterpolator(np.sort(y_in), sig_in_y**2), y_min, y_max)[0]
        res_mid = wbar_flat
        zm_slice = zm_in[ord_y_in]
        z_minus = zm_slice[0]
        z_plus  = zm_slice[-1]
        res_lh = sig_in_0**2 * norm.cdf(z_minus)
        res_rh = sig_in_1**2 * norm.cdf(-z_plus)

        res_vs = res_mid + res_lh + res_rh
        return res_vs

    for slice_idx in range(len(exp_dates)):
        if verbose:
            print(F"Processing slice: {slice_idx} (TAU={exp_dates[slice_idx]})")
        t = exp_dates[slice_idx]
        texp = ivol_data["Texp"]
        mid_vol = ivol_data["Mid"][texp == t]
        F = ivol_data["Fwd"][texp == t].iloc[0] # forward price
        k = np.log(ivol_data["Strike"][texp == t].values / F) # log-fwd moneyness, use .values
        vs_mid[slice_idx] = varswap(k, mid_vol, slice_idx) / t

    return {
        "expiries": exp_dates,
        "vs_mid": vs_mid,
    }

def obj_w(expiries, w_in):
    def objective(err_vec):
        w_in_1 = w_in + 2 * np.sqrt(w_in) * err_vec * np.sqrt(expiries)
        xi_vec = np.concatenate(
            ([w_in_1[0] / expiries[0]], np.diff(w_in_1) / np.diff(expiries))
        )
        dxi_dt = np.diff(xi_vec) / np.diff(expiries)
        w_out = (
            np.concatenate(([0], np.cumsum(xi_vec[1:] * np.diff(expiries))))
            + xi_vec[0] * expiries[0]
        )
        res = np.sum((w_in - w_out) ** 2) + np.sum(dxi_dt**2)
        return res * 1e3

    return objective

def xi_curve(expiries, w_in, eps=0):
    n = len(w_in)
    if eps > 0:
        res_optim = minimize(
            obj_w(expiries, w_in),
            np.zeros(n),
            method="L-BFGS-B",
            bounds=[(-eps, eps)] * n,
        )
        err_vec = res_optim.x
        w_in_1 = w_in + 2 * np.sqrt(w_in) * err_vec * np.sqrt(expiries)
    else:
        w_in_1 = w_in
    xi_vec_out = np.concatenate(
        ([w_in_1[0] / expiries[0]], np.diff(w_in_1) / np.diff(expiries))
    )

    def xi_curve_raw(t):
        if t <= expiries[-1]:
            return xi_vec_out[np.sum(expiries < t)]
        else:
            return xi_vec_out[-1]

    xi_curve_out = np.vectorize(xi_curve_raw)
    fit_errs = np.sqrt(w_in_1 / expiries) - np.sqrt(w_in / expiries)

    return {
        "xi_vec": xi_vec_out,
        "xi_curve": xi_curve_out,
        "fit_errs": fit_errs,
        "w_out": w_in_1,
    }

def xi_curve_smooth(expiries, w_in, xi=True, eps=0.0):
    def phi(tau):
        def func(x):
            min_val = np.minimum(x, tau)
            return 1 - min_val**3 / 6 + x * tau * (2 + min_val) / 2

        return func

    def phi_deri(tau):
        def func(x):
            min_val = np.minimum(x, tau)
            return tau - min_val**2 / 2 + tau * min_val

        return func

    n = len(expiries)
    A = np.array([[phi(expiries[i])(expiries[j]) for j in range(n)] for i in range(n)])
    A_inv = inv(A)

    def obj_1(err_vec):
        v = w_in + 2 * np.sqrt(w_in) * err_vec * np.sqrt(expiries)
        return v.T @ A_inv @ v

    res_optim = minimize(
        obj_1, np.zeros(n), method="L-BFGS-B", bounds=[(-eps, eps)] * n
    )
    err_vec = res_optim.x
    w_in_1 = w_in + 2 * np.sqrt(w_in) * err_vec * np.sqrt(expiries)
    Z = A_inv @ w_in_1

    def curve_raw(x):
        sum_curve = sum(Z[i] * phi(expiries[i])(x) for i in range(n))
        sum_curve_deri = sum(Z[i] * phi_deri(expiries[i])(x) for i in range(n))
        return sum_curve_deri if xi else sum_curve

    xi_curve_out = np.vectorize(curve_raw)
    fit_errs = np.sqrt(w_in_1 / expiries) - np.sqrt(w_in / expiries)

    return {"xi_curve": xi_curve_out, "fit_errs": fit_errs, "w_out": w_in_1}

# Optimisation tools

def solve_for_xi(alpha, lam, nu, y_bar, v_min, a, b, expiry_days_grid, dt=1/365):
    """
    Solves for the xi curve using a stable implicit method.
    Parameters:
    - alpha, lam, nu, y_bar, v_min, a, b: model parameters
    - expiry_days_grid: array of expiry days
    - dt: time step
    Returns:
    - xi_disc_by_expiry: xi values at the specified expiry days
    - xi_disc: full discretized xi curve"""
    # Calculate strictly to the maximum day required
    N = int(np.max(expiry_days_grid))
    
    # Use N+1 to cleanly map day index to time index (xi[n] is time n*dt)
    xi_disc = np.zeros(N + 1)
    
    C = (a * (y_bar - b)) ** 2 + v_min # this is the initial value xi[0]
    coeff = (nu / gamma(alpha)) ** 2
    gamma_exp = 2 * alpha - 2
    beta = 2 * lam
    
    # Grid of delays
    x = np.arange(N + 1) * dt
    
    # 1. Evaluate exact integrals using the Lower Incomplete Gamma function
    I0 = (beta ** -(gamma_exp + 1)) * gamma(gamma_exp + 1) * gammainc(gamma_exp + 1, beta * x)
    I1 = (beta ** -(gamma_exp + 2)) * gamma(gamma_exp + 2) * gammainc(gamma_exp + 2, beta * x)
        
    dI0 = np.diff(I0)
    dI1 = np.diff(I1)
    x_k = x[:-1]
    x_kp1 = x[1:]
    
    # 2. Compute interpolation weights A_k and B_k for piecewise linear assumption
    A = (coeff / dt) * (x_kp1 * dI0 - dI1)
    B = (coeff / dt) * (dI1 - x_k * dI0)
    
    # 3. Construct convolution weights
    w = np.zeros(N)
    w[0] = A[0]
    if N > 1:
        w[1:] = B[:-1] + A[1:]
        
    xi_disc[0] = C
    
    # 4. Solve implicitly step-by-step
    for n in range(1, N + 1):
        if n == 1:
            sum_val = B[0] * xi_disc[0]
        else:
            # np.dot handles the causal convolution
            sum_val = np.dot(w[1:n], xi_disc[1:n][::-1]) + B[n-1] * xi_disc[0]
            
        # The implicit formulation stabilizes the long-term curve
        xi_disc[n] = (C + sum_val) / (1 - w[0])
        
    # 5. Extract values at requested expiries
    xi_disc_by_expiry = np.zeros(len(expiry_days_grid))
    for i, expiry in enumerate(expiry_days_grid):
        # Note: Corrected off-by-one mapping. expiry=1 matches xi[1] exactly
        xi_disc_by_expiry[i] = xi_disc[int(expiry)]

    # Returning xi[1:] to keep output dimensions consistent with your original code
    return xi_disc_by_expiry, xi_disc

