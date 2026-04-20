import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import warnings
import traceback
import argparse
import cma
import multiprocess as mp

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocess import shared_memory
from scipy.stats import qmc
from tools.Asset import brownianPaths
from tools.grid import *
from tools.qrh_converge_test import *
from tools.qrh_params import *
from fwd_var_curve import xi_curve_smooth, var_swap_robust, solve_for_xi
from fx_init_const import *
from QuadraticRoughHeston import *
from optimiser import *
from tqdm import tqdm
from datetime import datetime
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint, Bounds
from plotWindow.plotWindow import plotWindow

"""
A full calibration of the Quadratic Rough Heston model to EURUSD FX options market data, 
using a forward variance curve fitting approach for the initial guess parameters, and a 
subsequent random search optimisation over the parameter space. The code is structured 
to be modular, with separate files for different components of the calibration process 
(e.g., Black-Scholes pricing, parameter handling, convergence testing). 

The main script orchestrates the entire calibration workflow, from data loading and 
preprocessing to model simulation, pricing, and plotting of results.

Submitted as part fulfilment of COMP0029, 
Individual Project for Year 3 BSc Computer Science, University College London, 2026.

Anthony Nkyi, April 2026.
"""

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

warnings.filterwarnings("ignore")
np.seterr(all='ignore')

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Quadratic Rough Heston FX calibration")
    parser.add_argument(
        "date",
        nargs="?",
        default="01-12-2023",
        help="Data date folder in DD-MM-YYYY format (default: 01-12-2023)",
    )
    parser.add_argument(
        "--ui-update-every",
        type=int,
        default=8,
        dest="ui_update_every",
        help="Process UI events every N simulation timesteps (smaller is smoother, default: 8)",
    )
    args = parser.parse_args()
    if args.ui_update_every <= 0:
        parser.error("--ui-update-every must be a positive integer")
    return args


cli_args = parse_cli_args()
date = cli_args.date
ui_update_every = cli_args.ui_update_every

FIGSIZE = (15,7.5)
FOUR_THREE = (16,12)
SQUARE = (14,14)
fig_count = 1

delta_ticks = [0.05,0.1,0.15,0.25,0.35,0.5,0.65,0.75,0.85,0.9,0.95]
delta_tick_labels = ['-0.05','','-0.15','-0.25','-0.35','ATM','0.35','0.25','0.15','','0.05']

img_date_path = f"{date}/img/"
data_date_path = f"{date}/data/"
img_expiries_path = f"{img_date_path}{TAU_STR[0]}-to-{TAU_STR[-1]}/"
data_expiries_path = f"{data_date_path}{TAU_STR[0]}-to-{TAU_STR[-1]}/"
ig_path = f"{img_expiries_path}initial_guess/"
mkt_path = f"{img_expiries_path}market/"
opt_path = f"{img_expiries_path}optimised/"

if not os.path.exists(img_expiries_path) or not os.path.exists(data_expiries_path):
    os.makedirs(img_expiries_path)
    os.makedirs(data_expiries_path)
    for path in [ig_path, opt_path]:
        os.makedirs(f"{path}model_fits/")
        os.makedirs(f"{path}model_smiles/")
        os.makedirs(f"{path}market_smiles/")

if not os.path.exists(data_date_path):
    os.makedirs(data_date_path)
    os.makedirs(data_expiries_path)

delta_put = np.where(DELTA > 0, DELTA - 1, DELTA)

setup_dict = setup(
    volatility_grid=VOL_QUOTES,
    expiries=TAU,
    foreign_rate=USD_OIS,
    domestic_rate=EUR_OIS,
    forward_prices=FWD,
    spot=spot,
    data_expiries_path=data_expiries_path,
)

fx_df = setup_dict["fx_df"]
strike_grid = setup_dict["strike_grid"]
PRICE_FROM_VOL_QUOTES = setup_dict["price_grid"]
log_fwd_moneyness_grid = setup_dict["log_fwd_moneyness_grid"]
inverse_vol_grid = setup_dict["inverse_vol_grid"]
base_rates_arr = setup_dict["base_rates_arr"]
term_rates_arr = setup_dict["term_rates_arr"]
cp_flags_grid = setup_dict["cp_flags_grid"]


# ------------- Main program starts here --------------
if __name__ == "__main__":
    fx_var_swap = var_swap_robust(fx_df, TAU)
    fx_expiries_arr = fx_var_swap["expiries"]
    w_in = fx_var_swap["vs_mid"] * fx_expiries_arr
    xi_smooth = xi_curve_smooth(fx_expiries_arr, w_in, eps=0.03)["xi_curve"]

    u = np.linspace(0, max(fx_expiries_arr), 1000)
    fig1, ax = plt.subplots()
    ax.plot(u, xi_smooth(u), color="red", linewidth=2)
    ax.set_xlabel("Maturity u")
    ax.set_ylabel(r"$\xi(u)$")
    ax.set_title(f"Smoothed xi curve from market variance swap data ({date})")
    pw = plotWindow()
    ui_last_update = [0.0]

    def pump_ui(force=False):
        now = time.perf_counter()
        if not force and now - ui_last_update[0] < 0.1:
            return
        if pw.MainWindow.isVisible():
            pw.update()
        ui_last_update[0] = now

    def add_plot_window_figure(figure, title=None):
        global fig_count
        pw.addPlot(title or f"Fig. {fig_count}", figure)
        fig_count += 1
        pump_ui(force=True)

    add_plot_window_figure(fig1)

    COMBINED_OIS_DICT = {expiry: rate for (expiry, rate) in zip(fx_expiries_arr, COMBINED_OIS)}

    initial = {"al": 0.568, "lam": 9.68, "nu": 0.572, "c": 1.1e-4, "a": 1.0, "b": 0.0}
    initial_arr = const_dict_to_param_arr(initial)

    max_exp = np.max(TAU_DAYS)

    xi_curve_over_tau_days = np.array([xi_smooth(i/365) for i in range (0, max_exp + 1)])
    days = np.arange(1, max_exp + 1, 1)

    mc_path_X = brownianPaths(PATHS, STEPS)
    mc_path_Variance = brownianPaths(PATHS, STEPS)

    AL_RANGE = (0.5001, 0.6150)
    LAM_RANGE = (0.1, 15.0) 
    NU_RANGE = (0.001, 1.5)
    C_RANGE = (xi_smooth(0.0) * 0.11, xi_smooth(0.0) * 0.9)
    A_RANGE = (0.001, 2.0)
    B_RANGE = (-10.0, 10.0)

    param_bounds = np.array([C_RANGE, NU_RANGE, LAM_RANGE, AL_RANGE, A_RANGE, B_RANGE])

    lower_bounds = [C_RANGE[0], NU_RANGE[0], LAM_RANGE[0], AL_RANGE[0], A_RANGE[0], B_RANGE[0]]
    upper_bounds = [C_RANGE[1], NU_RANGE[1], LAM_RANGE[1], AL_RANGE[1], A_RANGE[1], B_RANGE[1]]
    bounds = [lower_bounds, upper_bounds]

    range_widths = np.array([
        C_RANGE[1] - C_RANGE[0],
        NU_RANGE[1] - NU_RANGE[0],
        LAM_RANGE[1] - LAM_RANGE[0],
        AL_RANGE[1] - AL_RANGE[0],
        A_RANGE[1] - A_RANGE[0],
        B_RANGE[1] - B_RANGE[0]
    ])

    obj_calls = []

    converge_test_val_lambda = lambda x: (x[1] / gamma(x[3])) ** 2 * gamma(2 * (x[3] - 0.5)) / (2 * x[2]) ** (2 * (x[3] - 0.5))
    converge_test_constraint_lambda =  NonlinearConstraint(converge_test_val_lambda, 0, 0.9999)

    # Forward variance curve fitting objective function for initial guess parameters

    def xi_objective_with_grid(params):
        pump_ui()
        start_time = time.perf_counter()
        c, nu, lam, al, a, b = params[0], params[1], params[2], params[3], params[4], params[5]

        y_bar = (np.sqrt(xi_smooth(0.0) - c) / a) + b
        _, full_curve = solve_for_xi(al, lam, nu, y_bar, c, a, b, days)
        
        curve_loss = np.sum((full_curve - xi_curve_over_tau_days) ** 2) / max_exp

        if np.isnan(curve_loss):
            return 1e6

        diff = np.mean(np.abs(full_curve - xi_curve_over_tau_days) / xi_curve_over_tau_days)
        obj_calls.append(time.perf_counter() - start_time)
        return diff

    def de_progress_callback(_xk, _convergence):
        pump_ui(force=True)
        return False

    print("------------------------------------------------------------------------------")
    print("Starting Forward Variance Curve Fitting...")
    diff_ev_wall_time_start = time.perf_counter()
    initial_guess_result = differential_evolution(
        xi_objective_with_grid,
        param_bounds,
        constraints=(converge_test_constraint_lambda,),
        seed=79,
        workers=1,  # Single-threaded for debugging
        maxiter=228,
        popsize=72,
        atol=1e-8,
        disp=True,
        polish=False,
        callback=de_progress_callback,
    )
    diff_ev_time = time.perf_counter() - diff_ev_wall_time_start
    print(f"Optimisation completed in {diff_ev_time} seconds")
    print(f"Average time per objective call: {np.mean(obj_calls)} seconds")
    print(f"Average time per generation: {np.mean(obj_calls)*72} seconds")

    initial_guess_params = initial_guess_result.x
    c_ig = float(initial_guess_params[0])
    nu_ig = float(initial_guess_params[1])
    lam_ig = float(initial_guess_params[2])
    al_ig = float(initial_guess_params[3])
    a_ig = float(initial_guess_params[4])
    b_ig = float(initial_guess_params[5])

    # {'al': 0.8282318750477802, 'lam': 11.57363468461794, 'nu': 2.309373246487681, 'c': 1.340e-04} 
    params_ig = {"al": al_ig, "lam": lam_ig, "nu": nu_ig, "c": c_ig, "a": a_ig, "b": b_ig}

    print("\nInitial parameters:")
    print(f"c: {initial_arr[0]:.6e}")
    print(f"nu: {initial_arr[1]:.6f}")
    print(f"lam: {initial_arr[2]:.6f}")
    print(f"al: {initial_arr[3]:.6f}")
    print(f"a: {initial_arr[4]:.6f}")
    print(f"b: {initial_arr[5]:.6f}")

    print("\nOptimized Parameters for xi curve (initial guess):")
    print(f"xi(0): {xi_smooth(0.0):.6e}")
    print(f"c: {c_ig:.6e}")
    print(f"nu: {nu_ig:.6f}")
    print(f"lam: {lam_ig:.6f}")
    print(f"al: {al_ig:.6f}")
    print(f"a: {a_ig:.6f}")
    print(f"b: {b_ig:.6f}")

    #print(f"Final loss: {initial_guess_result.fun:.6e}")
    print(f"Passes convergence test: {converge_test_obj(nu_ig, lam_ig, al_ig)} ({converge_test(params_ig)})")

    y_bar_ig = (np.sqrt(xi_smooth(0.0) - c_ig) / a_ig) + b_ig
    e_c, fit_curve = solve_for_xi(al_ig, lam_ig, nu_ig, y_bar_ig, c_ig, a_ig, b_ig, TAU_DAYS)
    print(f"RMSE between fitted xi curve and smoothed xi curve: {np.sqrt(np.mean((fit_curve - xi_curve_over_tau_days) ** 2)):.6e}")
    print(f"Percentage difference between fitted xi curve and smoothed xi curve: {100*np.mean(np.abs((fit_curve - xi_curve_over_tau_days) / xi_curve_over_tau_days)):.6f}%")

    fig = plt.figure(figsize=(8,8))
    title = "Forward variance curve, {} ({} to {})".format(date, TAU_STR[0], TAU_STR[-1])
    u = np.linspace(0, max(fx_var_swap["expiries"]), 1000)
    plt.plot(u, xi_smooth(u), color="red", linewidth=3, label="Market")
    plt.plot([(x)/365 for x in range(0, fit_curve.size)], fit_curve, label="Fitted parameters", color="blue", linestyle='dashed', linewidth=3)
    plt.title(title, fontsize=16)
    plt.xlabel("Maturity u (years)", fontsize=14, labelpad=10)
    plt.ylabel(r"$\xi(u)$", fontsize=14, labelpad=10)

    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f'{ig_path}xi_curve.eps', format='eps')
    plt.savefig(f'{ig_path}xi_curve.png', dpi=600, format='png')
    add_plot_window_figure(fig)

    # Run simulation with initial guess parameters

    print(params_ig)

    qrh_ig = QuadraticRoughHeston(**params_ig, xi0=xi_smooth)

    start_time = time.perf_counter()
    res = qrh_ig.simulate_filtered(
        mc_path_Variance,
        mc_path_X,
        fx_expiries_arr,
        markovian_lift=False,
        interest_rates=COMBINED_OIS_DICT,
        ui_callback=pump_ui,
        ui_update_every=ui_update_every,
    )
    end_time = time.perf_counter() - start_time

    print("Number of paths", PATHS)
    print("Number of steps", STEPS)
    print(f"Simulation time : {end_time}s")

    # Pricing and plotting for initial guess parameters

    mc_path_matrix_ig = np.array([res[expiry]["X"] for expiry in fx_expiries_arr])
    initial_guess_price_grid = get_mc_prices_grid_log_fwd(
        mc_path_matrix_ig, spot, log_fwd_moneyness_grid, fx_expiries_arr, base_rates_arr, term_rates_arr, cp_flags_grid, FWD
    )

    initial_guess_iv_grid_jaeckel = get_iv_from_prices_grid_jaeckel(
        np.copy(initial_guess_price_grid), spot, strike_grid, fx_expiries_arr, base_rates_arr, term_rates_arr, cp_flags_grid, tol=1e-6
    )

    initial_guess_iv_grid_gatheral = get_iv_from_paths_grid_gatheral(
        np.copy(mc_path_matrix_ig), spot, log_fwd_moneyness_grid, fx_expiries_arr, base_rates_arr, term_rates_arr, cp_flags_grid
    )

    np.savetxt(f'{data_expiries_path}initial_guess_price_grid.csv', initial_guess_price_grid, delimiter=',')
    np.savetxt(f'{data_expiries_path}initial_guess_iv_grid_jaeckel.csv', initial_guess_iv_grid_jaeckel, delimiter=',')
    np.savetxt(f'{data_expiries_path}initial_guess_iv_grid_gatheral.csv', initial_guess_iv_grid_gatheral, delimiter=',')

    grid_size = VOL_QUOTES.size
    market_vega_grid = build_vega_grid_log_fwd(spot, log_fwd_moneyness_grid, fx_expiries_arr, term_rates_arr, base_rates_arr, VOL_QUOTES)
    ig_price_loss = np.sqrt(np.sum(((PRICE_FROM_VOL_QUOTES - initial_guess_price_grid) ** 2) / (market_vega_grid * grid_size)))
    ig_vol_loss = np.sqrt(np.sum((VOL_QUOTES - initial_guess_iv_grid_jaeckel ** 2) / grid_size))
    plt.rcParams.update({'font.size': 20})

    # PLOTTING THE 3D PRICE SURFACE (Market vs Model initial guess)
    fig = plt.figure(figsize=FOUR_THREE)
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(-delta_put, TAU)

    ax.plot_wireframe(X, Y, PRICE_FROM_VOL_QUOTES, color='red', label='Market')
    ax.plot_wireframe(X, Y, initial_guess_price_grid, color='blue', linestyle='--', label='Initial guess')
    ax.set_title(f'EURUSD option price initial guess')
    ax.tick_params(axis='both', pad=10)

    ax.set_xlabel('$\\Delta$', labelpad=20)
    ax.set_ylabel('Time to Expiry $\\tau$', labelpad=20)
    ax.set_zlabel(r'BS Market price $V_{mkt}$ (\$)', labelpad=20)
    ax.set_xticks([0.05,0.1,0.15,0.25,0.35,0.5,0.65,0.75,0.85,0.9,0.95])
    ax.set_xticklabels(['-0.05','','-0.15','-0.25','-0.35','ATM','0.35','0.25','0.15','','0.05'])
    ax.set_yticks(TAU)
    ax.set_yticklabels(TAU_TICKS)
    ax.legend()

    plt.savefig(f'{ig_path}model_fits/price_ig.eps', format='eps')
    plt.savefig(f'{ig_path}model_fits/price_ig.png', dpi=600, format='png')
    add_plot_window_figure(fig)

    # PLOTTING THE 3D VOLATILITY SURFACE (Market vs Model initial guess - Jaeckel)
    fig = plt.figure(figsize=FOUR_THREE)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_wireframe(X, Y, VOL_QUOTES, color='red', label='Market')
    ax.plot_wireframe(X, Y, initial_guess_iv_grid_jaeckel, color='blue', linestyle='--', label="Initial guess [Jaeckel]")
    ax.set_title("EURUSD implied volatility initial guess")
    ax.tick_params(axis='both', pad=10)

    ax.set_xlabel('$\\Delta$', labelpad=20)
    ax.set_ylabel('Time to Expiry $\\tau$', labelpad=20)
    ax.set_zlabel('BS Implied Vol $\\sigma_{imp}$', labelpad=20)
    ax.set_xticks([0.05,0.1,0.15,0.25,0.35,0.5,0.65,0.75,0.85,0.9,0.95])
    ax.set_xticklabels(['-0.05','','-0.15','-0.25','-0.35','ATM','0.35','0.25','0.15','','0.05'])
    ax.set_yticks(TAU)
    ax.set_yticklabels(TAU_TICKS)
    ax.legend(loc="upper right")

    plt.savefig(f'{ig_path}model_fits/vol_ig_jaeckel.eps', format='eps')
    plt.savefig(f'{ig_path}model_fits/vol_ig_jaeckel.png', dpi=600, format='png')
    add_plot_window_figure(fig)

    # PLOTTING THE 3D VOLATILITY SURFACE (Market vs Model initial guess - Gatheral)
    fig = plt.figure(figsize=FOUR_THREE)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_wireframe(X, Y, VOL_QUOTES, color='red', label='Market')
    ax.plot_wireframe(X, Y, initial_guess_iv_grid_gatheral, color='blue', linestyle='--', label="Initial guess [Gatheral]")
    ax.set_title("EURUSD implied volatility initial guess")

    ax.tick_params(axis='both', pad=10)
    ax.set_xlabel('$\\Delta$', labelpad=20)
    ax.set_ylabel('Time to Expiry $\\tau$', labelpad=20)
    ax.set_zlabel('BS Implied Vol $\\sigma_{imp}$', labelpad=20)
    ax.set_xticks([0.05,0.1,0.15,0.25,0.35,0.5,0.65,0.75,0.85,0.9,0.95])
    ax.set_xticklabels(['-0.05','','-0.15','-0.25','-0.35','ATM','0.35','0.25','0.15','','0.05'])
    ax.set_yticks(TAU)
    ax.set_yticklabels(TAU_TICKS)
    ax.legend(loc = "upper right")

    plt.savefig(f'{ig_path}model_fits/vol_ig_gatheral.eps', format='eps')
    plt.savefig(f'{ig_path}model_fits/vol_ig_gatheral.png', dpi=600, format='png')
    add_plot_window_figure(fig)

    print("Jaeckel Median/ Mean Diff:", np.median(np.abs(VOL_QUOTES - initial_guess_iv_grid_jaeckel)), np.mean(np.abs(VOL_QUOTES - initial_guess_iv_grid_jaeckel)))
    print("Gatheral Median/ Mean Diff:", np.median(np.abs(VOL_QUOTES - initial_guess_iv_grid_gatheral)), np.mean(np.abs(VOL_QUOTES - initial_guess_iv_grid_gatheral)))

    # Random search for params

    m = 10
    N = 2**m 

    sampler = qmc.Sobol(d=6, scramble=True, seed=79)
    sample_unit = sampler.random_base2(m=m)
    random_params = qmc.scale(sample_unit, lower_bounds, upper_bounds)

    valid_points = []

    count = 0

    best_loss = [100, 100]
    best_theta = None

    simulation_time = []
    valid_point_runtime = []
    total_time = time.perf_counter()

    for theta in random_params:
        pump_ui()
        count += 1
        if count % np.round(N / 20) == 0:
            print(f"------------------------------------------- RUN {count}/{N} -------------------------------------------")
        start_time = time.perf_counter()
        c, nu, lam, al, a, b = theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]
        param_dict = const_param_arr_to_dict([c, nu, lam, al, a, b])

        integral = (nu / gamma(al)) ** 2 * gamma(2 * (al - 0.5)) / (2 * lam) ** (2 * (al - 0.5))
        if integral >= 0.999999:
            continue
        
        try:
            qrh = QuadraticRoughHeston(**param_dict, xi0=xi_smooth)

            sim_time = time.perf_counter()
            paths = qrh.simulate_filtered(
                mc_path_Variance,
                mc_path_X,
                fx_expiries_arr,
                interest_rates=COMBINED_OIS_DICT,
                markovian_lift=True,
                ui_callback=pump_ui,
                ui_update_every=ui_update_every,
            )
            simulation_time.append(time.perf_counter() - sim_time)
            path_matrix = np.array([paths[expiry]["X"] for expiry in fx_expiries_arr])
            output_prices = get_mc_prices_grid_log_fwd(
                path_matrix, spot, log_fwd_moneyness_grid, fx_expiries_arr, base_rates_arr, term_rates_arr, cp_flags_grid, FWD
            )
            output_vol = get_iv_from_prices_grid_jaeckel(
                output_prices, spot, strike_grid, fx_expiries_arr, base_rates_arr, term_rates_arr, cp_flags_grid, tol=1e-6
            )
        except:
            continue
        
        if np.any(np.isnan(output_prices)):
            continue
        
        if np.any(np.isnan(output_vol)):
            continue

        vol_diff = (VOL_QUOTES - output_vol)
        price_diff = (PRICE_FROM_VOL_QUOTES - output_prices) 

        # RMSE: vega weighted price, volatility

        iv_diff_approx = price_diff / market_vega_grid
        p_loss = np.sqrt(np.mean((iv_diff_approx ** 2)))
        v_loss = np.sqrt(np.mean((vol_diff ** 2)))

        if (v_loss < best_loss[1]) and ((p_loss < best_loss[0]) or (v_loss * p_loss < best_loss[0] * best_loss[1])):
            best_loss = [p_loss, v_loss]
            best_theta = theta
            print(f"RUN {count}/{N} | NEW BEST LOSS: (Price={p_loss}, Vol={v_loss}) | (al={al}, lam={lam}, nu={nu}, c={c}, a={a}, b={b})")
        else:
            print(f"RUN {count}/{N} | Current Loss : (Price={p_loss}, Vol={v_loss}) | (al={al}, lam={lam}, nu={nu}, c={c}, a={a}, b={b})")
        valid_points.append([theta, (p_loss, v_loss)])
        valid_point_runtime.append(time.perf_counter() - start_time)

    print(f"FINAL BEST LOSS: (Price={best_loss[0]}, Vol={best_loss[1]}) | (al={best_theta[3]:.6f}, lam={best_theta[2]:.6f}, nu={best_theta[1]:.6f}, c={best_theta[0]}, a={best_theta[4]}, b={best_theta[5]})")
    print(f"Average simulation time: {np.mean(simulation_time):.6f} seconds")
    print(f"Average valid point runtime: {np.mean(valid_point_runtime):.6f} seconds from {len(valid_point_runtime)}/{N} samples")
    print(f"Total time: {time.perf_counter() - total_time:.6f} seconds")

    print("\n------------------------------------------------------------------------------\n")

    # sort list by price loss
    valid_points.sort(key=lambda x: x[1][0])

    # Print and keep top 8 parameter sets with lowest price loss
    print("Top 10 parameter sets with lowest price loss:")
    for i in range(min(8, len(valid_points))):
        theta, losses = valid_points[i]
        p_loss, v_loss = losses
        print(f"Rank {i+1}: Price Loss={p_loss:.6e}, Vol Loss={v_loss:.6e}, Kernel Val={converge_test_val_lambda(theta):.6f} | (al={theta[3]:.6f}, lam={theta[2]:.6f}, nu={theta[1]:.6f}, c={theta[0]:.6e}, a={theta[4]}, b={theta[5]})")

    # save top 20 to csv
    top_20 = []
    for i in range(min(20, len(valid_points))):
        theta, losses = valid_points[i]
        p_loss, v_loss = losses
        top_20.append([theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], p_loss, v_loss])

    top_20 = np.array(top_20)
    np.savetxt(f'{data_expiries_path}const_best_params_{TAU_STR[0]}_{TAU_STR[-1]}_2.csv', top_20, delimiter=',', fmt='%s', header='c,nu,lam,al,a,b,price_loss,vol_loss')

    # Plot random points 

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE)

    x_price_loss = []
    y_vol_loss = []
    c_converge = []

    for point in reversed(valid_points):
        theta = point[0]
        nu, lam, al = theta[1], theta[2], theta[3]
        x_price_loss.append(point[1][0])
        y_vol_loss.append(point[1][1])
        c_converge.append((nu / gamma(al)) ** 2 * gamma(2 * (al - 0.5)) / (2 * lam) ** (2 * (al - 0.5)))

    x_price_loss = np.array(x_price_loss)
    y_vol_loss = np.array(y_vol_loss)
    c_converge = np.array(c_converge)

    # Plot on first axis
    scatter0 = ax[0].scatter(x_price_loss, y_vol_loss,
                        c=c_converge,
                        marker='s',      
                        cmap='jet',
                        #edgecolors='black',
                        #linewidths=0.01,
                        vmin=0,
                        vmax=1)

    ax[0].tick_params(labelsize=13)

    ax[0].set_xlabel("Vega-weighted Price RMSE", fontsize=13)
    ax[0].set_ylabel("Volatility RMSE", fontsize=13)
    ax[0].legend()

    # Plot on second axis
    scatter1 = ax[1].scatter(ig_price_loss, ig_vol_loss,
            marker='*',            
            s=200,                 
            c=converge_test(params_ig), 
            cmap='jet',
            vmin=0,
            vmax=1,
            edgecolors='black',    
            linewidths=0.5,          
            zorder=10,             
            label='Initial Guess') 

    ax[1].scatter(x_price_loss, y_vol_loss,
                        c=c_converge, 
                        cmap='jet',
                        #edgecolors='black',
                        marker='s',
                        #linewidths=0.01,
                        vmin=0,
                        vmax=1)

    ax[1].tick_params(labelsize=13)


    ax[1].set_xlabel("Vega-weighted Price RMSE", fontsize=13)
    ax[1].set_ylabel("Volatility RMSE", fontsize=13)
    ax[1].legend(fontsize=13, loc='upper right')

    # https://stackoverflow.com/questions/23270445/adding-a-colorbar-to-two-subplots-with-equal-aspect-ratios 
    # Create dividers for BOTH axes
    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)

    # Create colorbar for first subplot, then immediately delete
    cbar1 = fig.colorbar(scatter0, cax=cax1)
    fig.delaxes(cax1)  

    cbar2 = fig.colorbar(scatter1, cax=cax2)
    cbar2.set_label('Resolvent Kernel Value', rotation=270, labelpad=20, fontsize=14)
    cbar2.ax.tick_params(labelsize=13)

    plt.tight_layout()
    plt.savefig(f'{ig_path}sobol_search_{len(valid_points)}_{N}.png', dpi=600, format='png')
    plt.savefig(f'{ig_path}sobol_search_{len(valid_points)}_{N}.eps', format='eps')
    add_plot_window_figure(fig)

    # Global optimisation: Warm started IPOP-CMA-ES
    converge_constraint = lambda x: (x[1] / gamma(x[3])) ** 2 * gamma(2 * (x[3] - 0.5)) / (2 * x[2]) ** (2 * (x[3] - 0.5))
    nlc = NonlinearConstraint(converge_constraint, 0, 0.9999999)
    cons = [nlc]

    def vol_price_obj(params):
        c, nu, lam, al, a, b = params
        param_dict = const_param_arr_to_dict([c, nu, lam, al, a, b])

        if converge_test_val(nu, lam, al) >= 0.9999999:
            return 1e10 * 5

        qrh = QuadraticRoughHeston(**param_dict, xi0=xi_smooth)

        paths = qrh.simulate_filtered(
            mc_path_Variance,
            mc_path_X,
            fx_expiries_arr,
            interest_rates=COMBINED_OIS_DICT,
            markovian_lift=True,
            ui_callback=pump_ui,
            ui_update_every=ui_update_every,
        )
        path_matrix = np.array([paths[expiry]["X"] for expiry in fx_expiries_arr])
        
        output_prices = get_mc_prices_grid_log_fwd(
            path_matrix, spot, log_fwd_moneyness_grid, fx_expiries_arr, base_rates_arr, term_rates_arr, cp_flags_grid, FWD
        )
        if np.any(np.isnan(output_prices)):
            return 1e10 * 2.5
        
        output_vol = get_iv_from_prices_grid_jaeckel(
            output_prices, spot, strike_grid, fx_expiries_arr, base_rates_arr, term_rates_arr, cp_flags_grid, tol=1e-6
        )
        
        vol_diff = (VOL_QUOTES - output_vol)
        
        return np.sqrt(np.mean(vol_diff ** 2))

    def vega_price_obj(params):
        c, nu, lam, al, a, b = params
        param_dict = const_param_arr_to_dict([c, nu, lam, al, a, b])

        if converge_test_val(nu, lam, al) >= 0.999999:
            return 1e10 * 5

        qrh = QuadraticRoughHeston(**param_dict, xi0=xi_smooth)

        paths = qrh.simulate_filtered(
            mc_path_Variance,
            mc_path_X,
            fx_expiries_arr,
            interest_rates=COMBINED_OIS_DICT,
            markovian_lift=True,
            ui_callback=pump_ui,
            ui_update_every=ui_update_every,
        )
        path_matrix = np.array([paths[expiry]["X"] for expiry in fx_expiries_arr])
        
        output_prices = get_mc_prices_grid_log_fwd(
            path_matrix, spot, log_fwd_moneyness_grid, fx_expiries_arr, base_rates_arr, term_rates_arr, cp_flags_grid, FWD
        )
        if np.any(np.isnan(output_prices)):
            return 1e10 * 2.5

        # RMSE: vega weighted price, volatility
        price_diff = (PRICE_FROM_VOL_QUOTES - output_prices) 
        iv_diff_approx = price_diff / market_vega_grid
        p_loss = np.sqrt(np.mean(iv_diff_approx ** 2))

        return p_loss

    def price_obj(params):
        c, nu, lam, al, a, b = params
        param_dict = {"c": c, "nu": nu, "lam": lam, "al": al, "a": a, "b": b}
        qrh = QuadraticRoughHeston(**param_dict, xi0=xi_smooth)
        paths = qrh.simulate_filtered(
            mc_path_Variance,
            mc_path_X,
            fx_expiries_arr,
            interest_rates=COMBINED_OIS_DICT,
            markovian_lift=True,
            ui_callback=pump_ui,
            ui_update_every=ui_update_every,
        )
        path_matrix = np.array([paths[expiry]["X"] for expiry in fx_expiries_arr])

        output_prices = get_mc_prices_grid_log_fwd(
            path_matrix, spot, log_fwd_moneyness_grid, fx_expiries_arr, base_rates_arr, term_rates_arr, cp_flags_grid, FWD
        )
        if np.any(np.isnan(output_prices)):
            return 1e10
        
        return np.sqrt(np.mean((PRICE_FROM_VOL_QUOTES - output_prices) ** 2))

    # Worker-safe objective: local import avoids NameError under spawn multiprocessing
    def worker_vega_price_obj(params):
        from tools.qrh_converge_test import converge_test_val
        from tools.qrh_params import const_param_arr_to_dict
        from tools.grid import get_mc_prices_grid_log_fwd
        from QuadraticRoughHeston import QuadraticRoughHeston
        from scipy.integrate import IntegrationWarning
        import numpy as np
        import warnings

        # Ensure regularization scales are always available in workers and out-of-order execution.
        c, nu, lam, al, a, b = params
        param_dict = const_param_arr_to_dict([c, nu, lam, al, a, b])

        if converge_test_val(nu, lam, al) >= 0.999999:
            return 9e9 * (converge_test_val(nu, lam, al))**3
        with np.errstate(invalid='ignore', over='ignore'):
            qrh = QuadraticRoughHeston(**param_dict, xi0=xi_smooth)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=IntegrationWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                paths = qrh.simulate_filtered(
                    mc_path_Variance, mc_path_X, fx_expiries_arr,
                    interest_rates=COMBINED_OIS_DICT, markovian_lift=True
                )
            path_matrix = np.array([paths[expiry]["X"] for expiry in fx_expiries_arr])

            output_prices = get_mc_prices_grid_log_fwd(
                path_matrix, spot, log_fwd_moneyness_grid, fx_expiries_arr,
                base_rates_arr, term_rates_arr, cp_flags_grid, FWD
            )

        if np.any(np.isnan(output_prices)):
            return 2e9 * (np.count_nonzero(np.isnan(output_prices))) * (1 + converge_test_val(nu, lam, al))

        # RMSE
        price_diff = (PRICE_FROM_VOL_QUOTES - output_prices)
        iv_diff_approx = (price_diff / market_vega_grid)

        # tikhonov regularisation of kernel parameters
        kernel_params = np.array([nu, lam, al])
        reg_term = 1e-5 * np.sum(np.square(kernel_params / range_widths[1:4]))
        return np.sqrt(np.mean((iv_diff_approx ** 2))) + reg_term

    cma_options = {
        'bounds': [lower_bounds, upper_bounds],
        'CMA_stds': range_widths,
        'popsize': 16,
        'maxfevals': 750 * 6,
        'tolfun': 1e-6,
        'verbose': -9,
        'verb_disp': 0,
        'seed': 79,
        'tolflatfitness': 10,
        'tolstagnation': 10,
        'CMA_active': True, # pushes the strategy away from bad regions
    }

    x0 = best_theta
    x1 = initial_guess_params
    x2 = valid_points[1][0]
    num_choose = 9

    ipop_x0 = np.array([vp[0] for vp in valid_points[:num_choose]] + [x1])

    # check at least 7 points are linearly independent for the optimization to work
    rank = np.linalg.matrix_rank(ipop_x0 - np.mean(ipop_x0, axis=0), tol=1e-5)
    print("\n------------------------------------------------------------------------------\n")
    print(f"Rank of initial points: {rank} out of {ipop_x0.shape[0]}")  

    while rank < 6:
        num_choose += 1
        ipop_x0 = np.array([vp[0] for vp in valid_points[:num_choose]] + [x1])
        rank = np.linalg.matrix_rank(ipop_x0 - np.mean(ipop_x0, axis=0), tol=1e-5)
        print(f"Rank of initial points: {rank} out of {ipop_x0.shape[0]}")

    ipop_loss = np.array([valid_points[i][1][0] + 1e-5 * np.sum(np.square(ipop_x0[i] / range_widths)) for i in range(num_choose)] + [worker_vega_price_obj(x1)])

    mp.set_start_method('spawn', force=True)

    # Fix 1: Exception class that can be pickled correctly
    class AboveMaximumException(Exception):
        pass

    # keep track of the shared memory blocks in the workers
    worker_shm_X = None
    worker_shm_Var = None

    def _worker_init(shm_name_X, shape_X, dtype_X, 
                    shm_name_Var, shape_Var, dtype_Var, 
                    xi_, fx_expiries_, COMBINED_OIS_DICT_, 
                    spot_, log_fwd_moneyness_grid_, fx_expiries_arr_, base_rates_arr_, 
                    term_rates_arr_, cp_flags_arr_, FWD_,
                    PRICE_FROM_VOL_QUOTES_, market_vega_grid_, range_widths_):
        
        import numpy as np
        from multiprocess import shared_memory
        
        global xi_smooth, mc_path_Variance, mc_path_X, fx_expiries, COMBINED_OIS_DICT
        global spot, log_fwd_moneyness_grid, fx_expiries_arr, base_rates_arr, term_rates_arr, cp_flags_arr, FWD
        global PRICE_FROM_VOL_QUOTES, market_vega_grid, worker_shm_X, worker_shm_Var, range_widths

        # attach to shared memory blocks created by the main process
        worker_shm_X = shared_memory.SharedMemory(name=shm_name_X)
        worker_shm_Var = shared_memory.SharedMemory(name=shm_name_Var)
        
        # reconstruct arrays backed by the shared memory buffer
        mc_path_X = np.ndarray(shape_X, dtype=dtype_X, buffer=worker_shm_X.buf)
        mc_path_Variance = np.ndarray(shape_Var, dtype=dtype_Var, buffer=worker_shm_Var.buf)

        xi_smooth = xi_
        fx_expiries = fx_expiries_
        COMBINED_OIS_DICT = COMBINED_OIS_DICT_
        spot = spot_
        log_fwd_moneyness_grid = log_fwd_moneyness_grid_
        fx_expiries_arr = fx_expiries_arr_
        base_rates_arr = base_rates_arr_
        term_rates_arr = term_rates_arr_
        cp_flags_arr = cp_flags_arr_
        FWD = FWD_
        PRICE_FROM_VOL_QUOTES = PRICE_FROM_VOL_QUOTES_
        market_vega_grid = market_vega_grid_
        range_widths = range_widths_

    def run_ws_ipop_cma_multiprocessed(
        obj,
        options=None,
        max_restarts=2, 
        incpop_factor=2, 
        prior_X=None,        # Historical parameters evaluated on a source task
        prior_loss=None,     # Historical losses corresponding to prior_X
        gamma=1,           # WS parameter - fraction of top solutions to keep
        alpha=0.25,           # WS parameter - regularization/uncertainty
        fallback_x0=ipop_x0,
        fallback_sigma0=0.25
    ):
        if options is None:
            options = {
                'bounds': bounds,
                'CMA_stds': range_widths,
                'popsize': 16,
                'maxfevals': 10000,
                'tolfun': 1e-8,
                'verbose': -9,
                'verb_disp': 0,
                'seed': 79
            }

        # 1. WARM START CALCULATION (WS-IPOP-CMA-ES)
        use_warm_start = prior_X is not None and prior_loss is not None
        
        if use_warm_start:
            print("\n--- Initializing WS-IPOP-CMA-ES with Prior Knowledge ---")
            sigma0 = 0.75
            x0_list, Sigma_star = warm_start(prior_X, prior_loss, gamma, alpha, bounds)
        else:
            print("\n--- No prior data provided. Using fallback initial guesses. ---")
            x0_list = fallback_x0
            sigma0 = fallback_sigma0

        num_cores = mp.cpu_count() // 2 + 3
        print(f"Starting parallel IPOP-CMA-ES optimization across {num_cores} cores using Shared Memory...")

        global_best_params = None
        global_best_loss = np.inf
        es_strategies = []
        times_per_generation = []

        # ALLOCATE SHARED MEMORY
        shm_X = shared_memory.SharedMemory(create=True, size=mc_path_X.nbytes)
        shm_Var = shared_memory.SharedMemory(create=True, size=mc_path_Variance.nbytes)

        lower_bounds_arr = np.array(options['bounds'][0])
        upper_bounds_arr = np.array(options['bounds'][1])

        try:
            shared_X = np.ndarray(mc_path_X.shape, dtype=mc_path_X.dtype, buffer=shm_X.buf)
            shared_Var = np.ndarray(mc_path_Variance.shape, dtype=mc_path_Variance.dtype, buffer=shm_Var.buf)
            np.copyto(shared_X, mc_path_X)
            np.copyto(shared_Var, mc_path_Variance)

            init_args = (
                shm_X.name, shared_X.shape, shared_X.dtype,
                shm_Var.name, shared_Var.shape, shared_Var.dtype,
                xi_smooth, fx_expiries_arr, COMBINED_OIS_DICT,
                spot, log_fwd_moneyness_grid, fx_expiries_arr, base_rates_arr, term_rates_arr, cp_flags_grid, FWD,
                PRICE_FROM_VOL_QUOTES, market_vega_grid, range_widths
            )

            base_popsize = options['popsize']

            with mp.Pool(processes=num_cores, initializer=_worker_init, initargs=init_args) as pool:
                for guess_idx, current_x0 in enumerate(x0_list):
                    print(f"\n=======================================================")
                    if use_warm_start:
                        print(f"   Evaluating Warm-Started Mean (m_star)")
                    else:
                        print(f"   Evaluating Initial Guess {guess_idx + 1}/{len(x0_list)}")
                    print(f"   Starting params: {current_x0}")
                    print(f"   Initial Loss: {obj(current_x0):.8e}")
                    print(f"=======================================================")
                    
                    # Reset popsize for each new guess sequence
                    current_popsize = base_popsize
                    t_per_gen = []
                    
                    for run in range(max_restarts + 1):
                        print(f"\n--- Starting IPOP Run {run + 1}/{max_restarts + 1} ---")
                        print(f"Population Size: {current_popsize}")
                        
                        run_options = options.copy()
                        run_options['popsize'] = current_popsize

                        es = cma.CMAEvolutionStrategy(current_x0, sigma0, inopts=run_options)
                        
                        # WARM START OVERRIDE: Manually inject the custom covariance matrix 
                        if use_warm_start and run == 0:
                            # calculate scaling matrix based on range_widths and your sigma0
                            # outer product scales the full covariance matrix (including cross-correlations)
                            scale_matrix = (sigma0**2) * np.outer(range_widths, range_widths)
                            
                            # 2. Convert raw Sigma_star into PyCMA's internal scaled space
                            es.C = Sigma_star / scale_matrix
                            
                            # 3. Perform standard eigen-decomposition on the normalized matrix
                            D2, B = np.linalg.eigh(es.C)
                            es.D = np.sqrt(np.maximum(D2, 1e-18)) 
                            es.B = B
                            es.invsqrtC = np.dot(es.B, np.diag(1.0 / es.D)).dot(es.B.T)
                            
                            print("  -> Injecting Normalized Warm-Start Covariance Matrix...")
                            print(f"  [Debug] True Initial Step Sizes: {es.sigma * np.sqrt(np.diag(es.C)) * range_widths}")

                        es.logger = cma.CMADataLogger(f'outcmaes/{TAU_STR[0]}-to-{TAU_STR[-1]}/guess{guess_idx}_run{run}/').register(es)

                        res = None
                        try:
                            wall_clock_start = time.perf_counter()
                            while not es.stop():
                                tick = time.perf_counter()
                                raw_solutions = es.ask()
                                clipped_solutions = [np.clip(sol, lower_bounds_arr, upper_bounds_arr) for sol in raw_solutions]
                                fitnesses = []
                                for fit in tqdm(
                                    pool.imap(obj, clipped_solutions),
                                    total=len(clipped_solutions),
                                    desc=f"Gen {es.result.iterations} | Best Loss: {es.result.fbest:.8e}",
                                    leave=False,
                                ):
                                    fitnesses.append(fit)
                                    pump_ui()
                                es.tell(
                                    raw_solutions,
                                    fitnesses,
                                    constraints_values=[[converge_constraint(np.array(sol))] for sol in clipped_solutions]
                                )
                                es.logger.add()
                                es.disp()
                                pump_ui(force=True)
                                tock = time.perf_counter()
                                t_per_gen.append(tock - tick)

                            res = es.result

                        except Exception as e:
                            print(f"Error during optimization run {run + 1}: {e}")
                            traceback.print_exc()
                            try:
                                res = es.result
                            except Exception:
                                res = None

                        es_strategies.append(es)

                        if res is None or res.xbest is None:
                            print(f"Run {run + 1} produced no valid result — skipping.")
                            current_popsize *= incpop_factor
                            continue
                        
                        time_gen = {"popsize": current_popsize, "mean_time_per_gen": np.mean(t_per_gen), 
                                    "wall_clock": time.perf_counter() - wall_clock_start, "loss": res.fbest, 
                                    "params": res.xbest, 'generations': res.iterations}
                        times_per_generation.append(time_gen)
                        print(f"Run {run + 1} Stopped: {es.stop()} | Iterations: {res.iterations}")
                        print(f"Run {run + 1} Best Loss: {res.fbest:.8e}")
                        print(f"Run {run + 1} Best Params: {res.xbest}")
                        print(f"Run {run + 1} Time per Gen: {time_gen['mean_time_per_gen']:.8f}s | Total Wall Clock: {time_gen['wall_clock']:.8f}s")

                        if res.fbest < global_best_loss:
                            print(f"  -> New Global Best Found! Updating global best loss and parameters.")
                            print(f"  -> New Best Parameters: {res.xbest} | New Best Loss: {res.fbest:.8e}")
                            global_best_loss = res.fbest
                            global_best_params = res.xbest

                        current_popsize *= incpop_factor
                        
                        # If restarting, start from the best point found in the previous run
                        current_x0 = global_best_params if global_best_params is not None else current_x0

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Initiating safe shutdown...")
        finally:
            try:
                shm_X.close()
                shm_X.unlink()
            except FileNotFoundError: pass
            
            try:
                shm_Var.close()
                shm_Var.unlink()
            except FileNotFoundError: pass

        return {"params": global_best_params, "loss": global_best_loss, "es": es_strategies, "times_per_generation": times_per_generation}

    # https://github.com/CMA-ES/pycma
    run_res = run_ws_ipop_cma_multiprocessed(obj=worker_vega_price_obj, options=cma_options, prior_X=ipop_x0, prior_loss=ipop_loss)

    count = 0
    plt.rcParams.update({'font.size': 11})
    for i, es in enumerate(run_res['es']):
        if es.result.iterations < 2:
            print(f"Skipping plot for run {i + 1} due to insufficient iterations.")
            continue
        es.plot()
        fig = plt.gcf()
        fig.set_size_inches(18, 10)
        cma.s.figsave(f'{opt_path}run_{i}_plot.png', dpi=600, format='png')
        cma.s.figsave(f'{opt_path}run_{i}_plot.eps', format='eps')
        add_plot_window_figure(fig)

    # Local optimization: Nelder-Mead
    opt_method = "Nelder-Mead"

    final_opt_sim = Simulator(worker_vega_price_obj)
    print("\n------------------------------------------------------------------------------\n")
    print(f"Starting final {opt_method} optimization with vega price objective...")

    def final_opt_callback(xk):
        pump_ui(force=True)
        return final_opt_sim.callback(xk)

    total_optimization_time = time.perf_counter()
    final_opt = minimize(
        method=opt_method,
        fun=final_opt_sim.simulate,
        callback=final_opt_callback,
        x0=run_res["params"],
        bounds=Bounds(lb=lower_bounds, ub=upper_bounds, keep_feasible=True),
        constraints=cons,
        tol=1e-8,
        options={"maxiter": 1500, "disp": True}, #"seed": 79,"adaptive": True,"workers": 8,"eps": 1e-6}
    )
    total_optimization_time = time.perf_counter() - total_optimization_time
    print(f"Final optimization completed in {total_optimization_time:.2f} seconds.")
    print(f"Average sim time per iteration: {np.sum(final_opt_sim.sim_times) / final_opt.nit}s over {final_opt.nit} iterations")

    # Display final optimization results
    print("\n------------------------------------------------------------------------------\n")

    # Initial guess loss evaluation
    loss1 = vega_price_obj(initial_guess_params)
    loss2 = vol_price_obj(initial_guess_params)
    loss3 = price_obj(initial_guess_params)
    print(f"Initial curve RMSE Loss: \nVega={loss1:.9e}, Vol={loss2:.9e}, Price={loss3:.9e}")
    print(f"Params: {initial_guess_params}")
    print(f"Resolvent kernel: {converge_test_val(initial_guess_params[1], initial_guess_params[2], initial_guess_params[3])}\n")

    # random search
    loss1 = vega_price_obj(best_theta)
    loss2 = vol_price_obj(best_theta)
    loss3 = price_obj(best_theta)
    print(f"Initial curve RMSE Loss: \nVega={loss1:.9e}, Vol={loss2:.9e}, Price={loss3:.9e}")
    print(f"Params: {best_theta}")
    print(f"Resolvent kernel: {converge_test_val(best_theta[1], best_theta[2], best_theta[3])}\n")

    # run 1 cma_es
    params = run_res['times_per_generation'][0]['params']
    loss1 = vega_price_obj(params)
    loss2 = vol_price_obj(params)
    loss3 = price_obj(params)
    kernel_val = converge_test_val(params[1], params[2], params[3])
    print(f"CMA-ES Run 1 RMSE Loss: \nVega={loss1:.9e}, Vol={loss2:.9e}, Price={loss3:.9e} | Kernel={kernel_val:.9e} | Params: {params}\n")

    # run 2 cma_es
    params =  run_res['times_per_generation'][1]['params']
    loss1 = vega_price_obj(params)
    loss2 = vol_price_obj(params)
    loss3 = price_obj(params)
    kernel_val = converge_test_val(params[1], params[2], params[3])
    print(f"CMA-ES Run 2 RMSE Loss: \nVega={loss1:.9e}, Vol={loss2:.9e}, Price={loss3:.9e} | Kernel={kernel_val:.9e} | Params: {params}\n")

    # run 3 cma_es
    params = run_res['times_per_generation'][2]['params']
    loss1 = vega_price_obj(params)
    loss2 = vol_price_obj(params)
    loss3 = price_obj(params)
    kernel_val = converge_test_val(params[1], params[2], params[3])
    print(f"Final CMA-ES RMSE Loss: \nVega={loss1:.9e}, Vol={loss2:.9e}, Price={loss3:.9e} | Kernel={kernel_val:.9e} | Params: {params}\n")

    loss1 = vega_price_obj(final_opt.x)
    loss2 = vol_price_obj(final_opt.x)
    loss3 = price_obj(final_opt.x)
    kernel_val = converge_test_val(final_opt.x[1], final_opt.x[2], final_opt.x[3])
    print(f"Final CMA-ES + {opt_method} RMSE Loss: \nVega={loss1:.9e}, Vol={loss2:.9e}, Price={loss3:.9e} | Kernel={kernel_val:.9e} | Params: {final_opt.x}\n")

    c_qrh, nu_qrh, lam_qrh, al_qrh, a_qrh, b_qrh = final_opt.x

    print("\n------------------------------------------------------------------------------\n")

    qrh_final_params = run_res["params"] #[c, result_sol.x[0], result_sol.x[1], result_sol.x[2]]

    c_qrh = float(qrh_final_params[0])
    nu_qrh = float(qrh_final_params[1])
    lam_qrh = float(qrh_final_params[2])
    al_qrh = float(qrh_final_params[3])
    a_qrh = float(qrh_final_params[4])
    b_qrh = float(qrh_final_params[5])

    print("\nInitial parameters:")
    print(f"c: {c_ig:.6e}")
    print(f"nu: {nu_ig:.6f}")
    print(f"lam: {lam_ig:.6f}")
    print(f"al: {al_ig:.6f}")
    print(f"a: {a_ig:.6f}")
    print(f"b: {b_ig:.6f}")

    print("\nSecond Calibration Parameters:")
    print(f"c: {c_qrh:.6e}")
    print(f"nu: {nu_qrh:.6f}")
    print(f"lam: {lam_qrh:.6f}")
    print(f"al: {al_qrh:.6f}")
    print(f"a: {a_qrh:.6f}")
    print(f"b: {b_qrh:.6f}")

    qrh_final_params = [final_opt.x[0], final_opt.x[1], final_opt.x[2], final_opt.x[3], final_opt.x[4], final_opt.x[5]]

    c_qrh = float(qrh_final_params[0])
    nu_qrh = float(qrh_final_params[1])
    lam_qrh = float(qrh_final_params[2])
    al_qrh = float(qrh_final_params[3])
    a_qrh = float(qrh_final_params[4])
    b_qrh = float(qrh_final_params[5])
    print("\nFinal Calibration Parameters:")
    print(f"c: {c_qrh:.6e}")
    print(f"nu: {nu_qrh:.6f}")
    print(f"lam: {lam_qrh:.6f}")
    print(f"al: {al_qrh:.6f}")
    print(f"a: {a_qrh:.6f}")
    print(f"b: {b_qrh:.6f}")
    print(f"Passes convergence test: {converge_test_obj(nu_qrh, lam_qrh, al_qrh)}")

    qrh_final_saved = np.array([str(datetime.now())] + qrh_final_params).reshape(1, -1)

    if os.path.exists(f"{data_expiries_path}final_calibrated_params.csv"):
        with open(f"{data_expiries_path}final_calibrated_params.csv", "ab") as f:
            np.savetxt(f, qrh_final_saved, delimiter=",", fmt="%s", comments="")
    else:
        np.savetxt(f"{data_expiries_path}final_calibrated_params.csv", qrh_final_saved, delimiter=",", header="time,c,nu,lam,al,a,b", fmt="%s", comments="")

    qrh_params = {"al": al_qrh, "lam": lam_qrh, "nu": nu_qrh, "c": c_qrh, "a": a_qrh , "b": b_qrh }

    optimized_qrh = QuadraticRoughHeston(**qrh_params, xi0=xi_smooth)
    res_optimized = optimized_qrh.simulate_filtered(
        mc_path_X,
        mc_path_Variance,
        fx_var_swap["expiries"],
        interest_rates=COMBINED_OIS_DICT,
        ui_callback=pump_ui,
        ui_update_every=ui_update_every,
    )

    mc_path_matrix_new = np.array([res_optimized[expiry]["X"] for expiry in fx_expiries])
    mc_var_matrix_new = np.array([res_optimized[expiry]["V"] for expiry in fx_expiries])

    optimised_model_price_grid = get_mc_prices_grid_log_fwd(
        mc_path_matrix_new, spot, log_fwd_moneyness_grid, fx_expiries_arr, base_rates_arr, term_rates_arr, cp_flags_arr, FWD
    )
    optimised_iv_grid_jaeckel = get_iv_from_prices_grid_jaeckel(
        optimised_model_price_grid, spot, strike_grid, fx_expiries_arr, base_rates_arr, term_rates_arr, cp_flags_arr,
    )
    optimised_iv_grid_gatheral = get_iv_from_paths_grid_gatheral(
        mc_path_matrix_new, spot, log_fwd_moneyness_grid, fx_expiries_arr, base_rates_arr, term_rates_arr, cp_flags_arr
    )

    np.savetxt(f"{data_expiries_path}optimized_model_price_grid.csv", optimised_model_price_grid, delimiter=",")
    np.savetxt(f"{data_expiries_path}optimized_iv_grid_jaeckel.csv", optimised_iv_grid_jaeckel, delimiter=",")
    np.savetxt(f"{data_expiries_path}optimized_iv_grid_gatheral.csv", optimised_iv_grid_gatheral, delimiter=",")

    # Plot results

    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'axes.labelpad': 20})

    # PLOTTING THE 3D PRICE SURFACE (Market vs Model)
    fig = plt.figure(figsize=FOUR_THREE)
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(-delta_put, TAU)

    ax.plot_wireframe(X, Y, PRICE_FROM_VOL_QUOTES, color='red', label='Market', linewidth=2,)
    ax.plot_wireframe(X, Y, optimised_model_price_grid, color='green', linestyle='--', label='Model', linewidth=2)
    ax.set_title("EURUSD option price surface")
    ax.set_xlabel('$\\Delta$', labelpad=20)
    ax.set_ylabel('Time to Expiry $\\tau$', labelpad=20)
    ax.set_zlabel(r'BS Market price $V_{mkt}$ (\$)', labelpad=20)
    ax.tick_params(axis='both', pad=10)
    ax.set_xticks(delta_ticks)
    ax.set_xticklabels(delta_tick_labels)
    ax.set_yticks(TAU)
    ax.set_yticklabels(TAU_TICKS)
    ax.legend(loc = "upper right")

    plt.savefig(f'{opt_path}model_fits/price_opt.eps', format='eps', bbox_inches='tight', pad_inches=1)
    plt.savefig(f'{opt_path}model_fits/price_opt.png', dpi=600, format='png')
    add_plot_window_figure(fig)

    # PLOTTING THE 3D IMP VOL SURFACE (Market vs Model) - JAECKEL
    fig = plt.figure(figsize=FOUR_THREE)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_wireframe(X, Y, VOL_QUOTES, color='red', label='Market', linewidth=2)
    ax.plot_wireframe(X, Y, optimised_iv_grid_jaeckel, color='green', linestyle='--', label="Model [Jaeckel]", linewidth=2)
    ax.set_title("EURUSD implied volatility")
    ax.tick_params(axis='both', pad=10)
    ax.set_xlabel('$\\Delta$')
    ax.set_ylabel('Time to Expiry $\\tau$')
    ax.set_zlabel('BS Implied Vol $\\sigma_{imp}$')
    ax.set_xticks(delta_ticks)
    ax.set_xticklabels(delta_tick_labels)
    ax.set_yticks(TAU)
    ax.set_yticklabels(TAU_TICKS)
    ax.legend(loc = "upper right")

    plt.savefig(f'{opt_path}model_fits/vol_opt_jaeckel.eps', format='eps', bbox_inches='tight', pad_inches=1)
    plt.savefig(f'{opt_path}model_fits/vol_opt_jaeckel.png', dpi=600, format='png')
    add_plot_window_figure(fig)

    # PLOTTING THE 3D IMP VOL SURFACE (Market vs Model) - GATHERAL
    fig = plt.figure(figsize=FOUR_THREE)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_wireframe(X, Y, VOL_QUOTES, color='red', label='Market', linewidth=2)
    ax.plot_wireframe(X, Y, optimised_iv_grid_gatheral, color='green', linestyle='--', label="Model [Gatheral]", linewidth=2)
    ax.set_title("EURUSD implied volatility")

    ax.set_xlabel('$\\Delta$')
    ax.set_ylabel('Time to Expiry $\\tau$')
    ax.set_zlabel('BS Implied Vol $\\sigma_{imp}$')
    ax.set_xticks(delta_ticks)
    ax.set_xticklabels(delta_tick_labels)
    ax.set_yticks(TAU)
    ax.set_yticklabels(TAU_TICKS)
    ax.legend(loc = "upper right")

    plt.savefig(f'{opt_path}model_fits/vol_opt_gatheral.eps', format='eps', bbox_inches='tight', pad_inches=1)
    plt.savefig(f'{opt_path}model_fits/vol_opt_gatheral.png', dpi=600, format='png')
    add_plot_window_figure(fig)

    plt.rcParams.update({'font.size': 9})
    plt.rcParams.update({'axes.labelpad': 9})

    # Jaeckel smiles
    fig, axes = plt.subplots(nrows=ROWS, ncols=COLS, figsize=FIGSIZE)
    axes = axes.flatten()

    Z = VOL_QUOTES

    for i in range (len(TAU)):
        ax = axes[i]
        ax.set_xlabel('Delta $\\Delta$')
        ax.set_xticks([0.05,0.15,0.25,0.35,0.5,0.65,0.75,0.85,0.95])
        ax.set_xticklabels(['','','-0.25','','ATM','','0.25','',''])
        ax.set_ylabel('$\\sigma_{mkt}$')
        ax.set_title(f'$\\tau = {TAU[i]:.4f}$')
        ax.plot(X[i], Z[i], color="red", linewidth=2)
        ax.plot(X[i], optimised_iv_grid_jaeckel[i], color="green", linestyle="--", linewidth=2)

    fig.legend(["Market", "Model [Jaeckel]"], loc="lower right")
    fig.tight_layout()
    plt.savefig(f'{opt_path}model_smiles/jaeckel_strike_smile_{TAU_STR[0]}_{TAU_STR[-1]}.eps', format='eps')
    plt.savefig(f'{opt_path}model_smiles/jaeckel_strike_smile_{TAU_STR[0]}_{TAU_STR[-1]}.png', dpi=600, format='png')
    add_plot_window_figure(fig)

    # Gatheral smiles
    fig, axes = plt.subplots(nrows=ROWS, ncols=COLS, figsize=FIGSIZE)
    axes = axes.flatten()

    for i in range (len(TAU)):
        ax = axes[i]
        ax.set_xlabel('Delta $\\Delta$')
        ax.set_xticks([0.05,0.15,0.25,0.35,0.5,0.65,0.75,0.85,0.95])
        ax.set_xticklabels(['','','-0.25','','ATM','','0.25','',''])
        ax.set_ylabel('$\\sigma_{mkt}$')
        ax.set_title(f'$\\tau = {TAU[i]:.4f}$')
        ax.plot(X[i], Z[i], color="red", linewidth=2)
        ax.plot(X[i], optimised_iv_grid_gatheral[i], color="green", linestyle="--", linewidth=2)

    fig.legend(["Market", "Model [Gatheral]"], loc="lower right")
    fig.tight_layout()
    plt.savefig(f'{opt_path}model_smiles/gatheral_strike_smile_{TAU_STR[0]}_{TAU_STR[-1]}.eps', format='eps')
    plt.savefig(f'{opt_path}model_smiles/gatheral_strike_smile_{TAU_STR[0]}_{TAU_STR[-1]}.png', dpi=600, format='png')
    add_plot_window_figure(fig)

    # Keep the plotting UI responsive until closed by the user.
    while pw.MainWindow.isVisible():
        pw.update()
        time.sleep(0.05)