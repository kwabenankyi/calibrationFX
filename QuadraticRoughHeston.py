import numpy as np
import time

from numfracpy import Mittag_Leffler_two
from scipy import integrate
from scipy.special import gamma, gammainc, roots_jacobi
from scipy.optimize import nnls
from typing import Callable

"""
Quadratic Rough Heston model. Updated by Anthony Nkyi.

Authors: Florian Bourgey, Jim Gatheral, Anthony Nkyi

Latest edits: April 2026.
Original for reference is at the URL below.
Original: https://github.com/jgatheral/QuadraticRoughHeston. [Retrieved 2025-11-19]
"""

class QuadraticRoughHeston:
    def __init__(
        self,
        xi0: Callable[[float], float],
        c: float,
        nu: float,
        lam: float,
        al: float,
        n_quad: int = 20,
        a: float = 1.0,
        b: float = 0.0
    ):
        if not (0.5 < al < 1):
            raise ValueError(f"'al' ({al}) must be between 0.5 and 1.")

        if not c > 0:
            raise ValueError(f"'c' ({c}) must be positive.")

        if not nu > 0:
            raise ValueError(f"'nu' ({nu}) must be positive.")

        if not lam > 0:
            raise ValueError(f"'lam' ({lam}) must be positive.")
        
        if xi0(0.0) < c:
            raise ValueError(f"'c' ({c}) must be not be greater than xi0(0.0) {xi0(0.0)}.")

        self.xi0 = xi0
        self.c = c
        self.nu = nu
        self.lam = lam
        self.al = al
        self.H = self.al - 0.5
        self.n_quad = n_quad
        self.nu_hat = self.nu * gamma(2.0 * self.H) ** 0.5 / gamma(self.al)
        # Skew constant
        self.b = b
        # Scale factor on quadratic term in variance function
        self.a = a
        # Initial forward variance
        self.v0 = self.xi0(0.0)
        # Initial y0: weighted average of historical returns - equivalent to y0_field at t=0 (y_0 (0))
        self.y0_0 = self.b + np.sqrt((self.xi0(0.0) - self.c) / self.a)
        # Conditional variance theta constant
        self.alp = 1.0 / (2.0 * self.H + 1.0)

    def kernel(self, x: np.ndarray) -> np.ndarray:
        """Gamma kernel."""
        return (self.nu / gamma(self.al)) * x ** (self.al - 1) * np.exp(-self.lam * x)

    def y0(self, u):
        """Compute y0(u) from xi0(u)."""
        u = np.atleast_1d(u)
        integral = np.zeros_like(u)
        mask = u > 0
        x_jac, w_jac = roots_jacobi(n=self.n_quad, alpha=2.0 * self.H - 1.0, beta=0.0)
        integral[mask] = (
            w_jac[:, None]
            * self.xi0(0.5 * u[mask][None, :] * (1 + x_jac[:, None]))
            * np.exp(-self.lam * u[mask][None, :] * (1 - x_jac[:, None]))
        ).sum(axis=0)
        integral[mask] *= (self.nu / gamma(self.al)) ** 2 * (0.5 * u[mask]) ** (
            2.0 * self.H
        )
        return self.b + np.sqrt((self.xi0(u) - self.c) / self.a  - integral)

    def y0_shifted(self, u: np.ndarray, h: float) -> np.ndarray:
        """Compute shifted or blipped y0(u)."""
        return self.y0(u) - h * self.kernel(u)

    def resolvent_kernel(self, x):
        """Compute the resolvent kernel at x."""
        return (
            self.nu_hat**2
            * np.exp(-2.0 * self.lam * x)
            * x ** (2.0 * self.H - 1.0)
            * Mittag_Leffler_two(
                self.nu_hat**2 * x ** (2.0 * self.H),
                2.0 * self.H,
                2.0 * self.H,
            )
        )

    def integral_bigK0(self, x):
        r"""
        Compute the integral \int_0^x resolvent_kernel(s) ds.
        """
        if x > 0:
            return integrate.quad(lambda x: self.resolvent_kernel(x), 0.0, x)[0]
        elif x == 0:
            return 0.0
        else:
            raise ValueError("'x' must be non-negative.")
        
    def integral_K00(self, x, quad_scipy=False):
        r"""
        Compute the integral K00 at x. It corresponds to the integral
        \int_0^x k(s)^2 ds where k(s) is the gamma kernel function at s.
        Note: in SciPy, gammainc is the regularized lower incomplete gamma function.
        """
        if quad_scipy:
            return integrate.quad(lambda s: self.kernel(s) ** 2, 0.0, x)[0]
        else:
            x = np.atleast_1d(np.asarray(x))
            mask = x > 0
            res = np.empty_like(x)
            res[mask] = (
                gamma(2.0 * self.H)
                * gammainc(2.0 * self.H, 2 * self.lam * x[mask])
                / (2.0 * self.lam) ** (2.0 * self.H)
            )
            res[~mask] = x[~mask] ** (2.0 * self.H) / (2.0 * self.H)
            res *= (self.nu / gamma(self.al)) ** 2
            return res

    def integral_K0(self, x, quad_scipy=False):
        r"""
        Compute the integral K0 at x. It corresponds to the integral \int_0^x k(s) ds
        where k(s) is the gamma kernel function at s.
        """
        if quad_scipy:
            return integrate.quad(lambda s: self.kernel(s), 0.0, x)[0]
        else:
            x = np.atleast_1d(np.asarray(x))
            mask = x > 0
            res = np.empty_like(x)
            res[mask] = (
                gamma(self.H + 0.5)
                * gammainc(self.H + 0.5, self.lam * x[mask])
                / self.lam ** (self.H + 0.5)
            )
            res[~mask] = x[~mask] ** self.al / self.al
            res *= self.nu / gamma(self.al)
            return res
    
    def calibrate_markovian_lift(self, bstar, dt, M, T_max):
        """
        Fitting M exponentials directly to bstar in the simulation.
        """
        num_steps = len(bstar)
        
        if num_steps > 400:
            # Create a logarithmically spaced index array up to num_steps
            idx = np.unique(np.geomspace(1, num_steps, 400).astype(int)) - 1
            # Ensure the very first critical point is included
            if idx[0] != 0:
                idx = np.insert(idx, 0, 0)
        else:
            idx = np.arange(num_steps)

        time_grid = idx * dt
        bstar_subset = bstar[idx]
        
        # geometric grid for decay rates: add 'lam' to decay rates as the original kernel already has an e^(-lam * t) factor.
        decay_min = 0.1 / T_max
        decay_max = np.sqrt(T_max) / dt
        decay_rates = np.geomspace(decay_min, decay_max, M) + self.lam
        
        # shape (num_points, M)
        # each col is one exponential e^(-decay_rate_i * t)
        A = np.asfortranarray(np.exp(-time_grid[:, None] * decay_rates[None, :]))  # outer product
        # Solve for weights - must be non-negative to ensure stability in Monte Carlo paths.
        final_weights, _ = nnls(A, bstar_subset)
        
        # Approximated target: A * w
        self.bstar_approx = A @ final_weights
        self.markov_calibration_rmse = np.sqrt(np.mean((bstar_subset - self.bstar_approx) ** 2))
        return final_weights, decay_rates, idx
    
    def plot_markovian_lift_fit(self, max_expiry, num_steps, M, in_plot=False, export_path=None, show_plot=True):
        """
        Plots the true bstar kernel against the M-exponential Markovian lift approximation.
        
        Parameters:
        - max_expiry: Maximum time horizon (T_max)
        - num_steps: Number of simulation steps (used to calculate dt)
        - M: Number of exponentials for the Markovian lift (self.n_quad)
        """
        from matplotlib import pyplot as plt
        import time
        dt = max_expiry / num_steps
    
        time_field = np.arange(1, num_steps + 1) * dt
        
        K00_field = np.zeros(num_steps + 1)
        K00_field[1:] = self.integral_K00(time_field)
        
        bstar = np.sqrt(np.diff(K00_field) / dt)
        
        time_taken = time.perf_counter()
        try:
            _, _, time_idx = self.calibrate_markovian_lift(bstar, dt, M, max_expiry)
        except RuntimeError as e:
            print(f"Markovian lift calibration failed: {e}")
            return
        time_taken = time.perf_counter() - time_taken
        print(f"Markovian lift calibration completed in {time_taken:.8f} seconds.")
        
        # Reconstruct the approximation: matrix A is shape (num_steps, M)
        time_grid = np.arange(0, num_steps) * dt
        
        print(f"Calibration RMSE: {self.markov_calibration_rmse:.8e}")
        
        # Plot the comparison
        if show_plot:
            plt.figure(figsize=(8, 8))
            plt.plot(time_grid, bstar, label='Original $b\\star$', color='red', lw=3)
            plt.plot(time_grid[time_idx], self.bstar_approx, label=f'Markovian Approx. $(M={M})$', 
                        color='blue', linestyle='--', lw=3)
            
            plt.title(f"Markovian Lift Kernel Calibration $(M={M})$", fontsize=16)
            plt.xlabel("Time $\\tau$ (years)", fontsize=14)
            plt.ylabel("Kernel increment RMS $(b\\star)$", fontsize=14)
            plt.legend(fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Optional: inset axis to zoom in on the singularity at t=0 (Where rough kernels usually struggle to fit)
            if in_plot:
                axins = plt.gca().inset_axes([0.4, 0.4, 0.4, 0.4])
                axins.plot(time_grid[:int(num_steps*0.1)], bstar[:int(num_steps*0.1)], color='blue', lw=3)
                axins.plot(time_grid[time_idx[:int(num_steps*0.1)]], self.bstar_approx[:int(num_steps*0.1)], color='red', linestyle='--', lw=3)
                axins.set_title("Zoom near t=0", fontsize=14)
                axins.grid(True, alpha=0.3)
            # Output the Mean Squared Error for numerical tracking
        
            if export_path:
                plt.savefig(export_path + ".png", dpi=600, bbox_inches='tight')
                plt.savefig(export_path + ".eps", bbox_inches='tight')
            plt.show()

    def simulate(
        self,
        paths,
        steps,
        expiries,
        output="all",
        delvix=1.0 / 12.0,
        nvix=10,
        h_ssr=None,
    ):
        if output not in ["all", "spx", "vix"]:
            raise ValueError("'output' must be one of ['all', 'spx', 'vix'].")
        if not paths > 0:
            raise ValueError("'paths' must be postive.")
        if not steps > 0:
            raise ValueError("'steps' must be positive.")
        if not delvix > 0:
            raise ValueError("'delvix' must be positive.")
        if not nvix > 0:
            raise ValueError("'nvix' must be positive.")
        if not isinstance(expiries, (list, np.ndarray)):
            raise ValueError("'expiries' must be a list or numpy array.")
        # if h_ssr is not None:
        #     h_ssr = np.atleast_1d(np.asarray(h_ssr))
        #     if h_ssr.shape[0] == 1:
        #         h_ssr = np.full_like(expiries, h_ssr[0])
        #     if h_ssr.shape[0] != len(expiries):
        #         raise ValueError(
        #             "'h_ssr' must be a scalar or have the same length as 'expiries'."
        #         )
        # print(h_ssr)

        Z_eps = np.random.normal(size=(steps, paths))
        Z_chi = np.random.normal(size=(steps, paths))
        v0 = self.xi0(0.0)
        y0_0 = (self.xi0(0.0) - self.c) ** 0.5
        alp = 1.0 / (2.0 * self.H + 1.0)

        def sim(expiry):
            dt = expiry / steps
            K0del = float(self.integral_K0(dt))
            K00del = float(self.integral_K00(dt))
            bigK0del = self.integral_bigK0(dt)
            tj = np.arange(1, steps + 1) * dt
            yj = self.y0(tj)
            K00j = np.zeros(steps + 1)
            K00j[1:] = self.integral_K00(tj)
            bstar = np.sqrt(np.diff(K00j) / dt)
            chi = np.zeros((steps, paths))
            v = np.full(paths, v0)
            Y = np.full(paths, y0_0)
            yhat = np.full(paths, yj[0])
            rho_uchi = K0del / (K00del * dt) ** 0.5
            beta_uchi = K0del / dt
            X = np.zeros(paths)
            w = np.zeros(paths)

            if h_ssr is not None:
                chi_h = np.zeros((steps, paths))
                v_h = np.full(paths, v0)
                Y_h = np.full(paths, y0_0)
                yj_h = self.y0_shifted(tj, h_ssr(expiry))
                yhat_h = np.full(paths, yj_h[0])
                X_h = np.zeros(paths)
                w_h = np.zeros(paths)

            for j in range(steps):
                vbar = bigK0del * (alp * yhat**2 + (1 - alp) * Y**2 + self.c) / K00del
                sig_chi = np.sqrt(vbar * dt)
                sig_eps = np.sqrt(vbar * K00del * (1.0 - rho_uchi**2))
                chi[j, :] = sig_chi * Z_chi[j, :]
                eps = sig_eps * Z_eps[j, :]
                u = beta_uchi * chi[j, :] + eps
                Y = yhat + u
                vf = Y**2 + self.c
                dw = (v + vf) / 2 * dt
                w += dw
                X -= 0.5 * dw + chi[j, :]
                if j < steps - 1:
                    btilde = bstar[1 : j + 2][::-1]
                    yhat = yj[j + 1] + np.tensordot(btilde, chi[: j + 1, :], axes=1)
                v = vf

                if h_ssr is not None:
                    vbar_h = (
                        bigK0del
                        * (alp * yhat_h**2 + (1 - alp) * Y_h**2 + self.c)
                        / K00del
                    )
                    sig_chi_h = np.sqrt(vbar_h * dt)
                    sig_eps_h = np.sqrt(vbar_h * K00del * (1.0 - rho_uchi**2))
                    chi_h[j, :] = sig_chi_h * Z_chi[j, :]
                    eps_h = sig_eps_h * Z_eps[j, :]
                    u_h = beta_uchi * chi_h[j, :] + eps_h
                    Y_h = yhat_h + u_h
                    vf_h = Y_h**2 + self.c
                    dw_h = (v_h + vf_h) / 2 * dt
                    w_h += dw_h
                    X_h -= 0.5 * dw_h + chi_h[j, :]
                    if j < steps - 1:
                        btilde = bstar[1 : j + 2][::-1]
                        yhat_h = yj_h[j + 1] + np.tensordot(
                            btilde, chi_h[: j + 1, :], axes=1
                        )
                    v_h = vf_h

            if output in ["vix", "all"]:
                vix2 = 0.0
                ds = delvix / nvix
                for k in range(nvix):
                    tk = expiry + (k + 1.0) * ds
                    Ku = np.concatenate(
                        (self.integral_K00(tk), self.integral_K00(tk - tj))
                    )
                    ck_vec = np.sqrt(-np.diff(Ku) / dt)
                    dyTu = np.dot(ck_vec, chi)
                    yTu = self.y0(tk) + dyTu
                    vix2 += (
                        (yTu**2 + self.c)
                        * (1.0 + self.integral_bigK0((nvix - k - 1) * ds))
                        / nvix
                    )
                vix2 += v * (1.0 + self.integral_bigK0(delvix)) / (2 * nvix) - (
                    yTu**2 + self.c
                ) / (2.0 * nvix)
                vix = np.sqrt(vix2)

            res_sim = {}

            if output in ["all", "spx"]:
                res_sim["X"] = X
            if output in ["all", "vix"]:
                res_sim["vix"] = vix
            if output == "all":
                res_sim["v"] = v
                res_sim["w"] = w

            if h_ssr is not None:
                res_sim.update({"v_h": v_h, "X_h": X_h, "w_h": w_h})

            return res_sim

        sim_out = {expiry: sim(expiry) for expiry in expiries}
        return sim_out

    def simulate_filtered(
        self,
        mc_path_V: np.ndarray,
        mc_path_X: np.ndarray,
        expiries,
        output="all",
        delvix=1.0 / 12.0,
        nvix=10,
        interest_rates: dict = None,
        markovian_lift: bool = False
    ):
        """
        QRH simulation which processes all expiries on one path, rather than separate computation.
        Respects the non-Markovian nature of memory function (Y), while maintaining the Markovian 
        structure for the lifted variables. Includes interest rate (drift) adjustments.

        Must have enough steps to protect against error.

        Must input pre-generated independent paths for the Brownian motions (mc_path_V and mc_path_X)
        to run the simulation (for optimisation purposes). If not wanted, use simulate_filtered_random 
        instead.
        """
        if output not in ["all", "spx", "vix"]:
            raise ValueError("'output' must be one of ['all', 'spx', 'vix'].")
        if not delvix > 0:
            raise ValueError("'delvix' must be positive.")
        if not nvix > 0:
            raise ValueError("'nvix' must be positive.")
        if not isinstance(expiries, (list, np.ndarray)):
            raise ValueError("'expiries' must be a list or numpy array.")
        
        # Sort expiries and create time grid
        expiries = np.array(sorted(expiries))
        max_expiry = expiries[-1]
        if interest_rates is not None:
            interest_rates_array = np.array([interest_rates[exp] for exp in expiries])
        
        # Independent paths
        # For variance, log-asset paths
        Z_eps = mc_path_V
        Z_chi = mc_path_X
        
        num_paths = Z_eps.shape[0]
        num_steps = Z_eps.shape[1]
        
        # Assume num_steps corresponds to max_expiry
        dt = max_expiry / num_steps
        
        # Find which step indices correspond to each expiry
        expiry_steps = np.round(expiries / dt).astype(int)
        expiry_steps = np.clip(expiry_steps, 1, num_steps)
        expiry_steps_set = set(expiry_steps)
        
        # Precompute kernel integrals
        K0_delta = np.float64(self.integral_K0(dt))
        K00_delta = np.float64(self.integral_K00(dt))
        bigK0_delta = np.float64(self.integral_bigK0(dt))
        
        time_field = np.arange(1, num_steps + 1) * dt
        y0_field = self.y0(time_field)
        
        K00_field = np.zeros(num_steps + 1)
        K00_field[1:] = self.integral_K00(time_field)
        
        # Independent vectors for each path
        V = np.full(num_paths, self.v0)
        Y = np.full(num_paths, self.y0_0)
        yhat = np.full(num_paths, y0_field[0])
        
        beta_uchi = K0_delta / dt
        rho_uchi = K0_delta / (K00_delta * dt) ** 0.5
        
        # RMS average of kernel increments
        bstar = np.sqrt(np.diff(K00_field) / dt)
        bstar_rev = bstar[::-1]
        
        X = np.zeros(num_paths)
        w = np.zeros(num_paths)

        # For conditional variance simulation: u = Beta * chi + eps
        if markovian_lift:
            chi_j = np.empty(num_paths)
        else:
            chi = np.zeros((num_steps, num_paths))
        
        # Pre-computations of scalars to speed up ops
        sqrt_sig_eps_constant = np.sqrt(K00_delta * (1.0 - rho_uchi**2))
        alp_complement = (1 - self.alp)
        K_ratio = bigK0_delta / K00_delta
        sqrt_dt = np.sqrt(dt)

        # Results stored at each expiry
        results = {}
        exp_completed = 0

        # Markovian lift - approximation for the approximation of yhat
        if markovian_lift:
            calibration_time_start = time.perf_counter()
            try:
                self.lift_mus, self.lift_gammas, _ = self.calibrate_markovian_lift(bstar, dt, self.n_quad, max_expiry)
                self.markov_calibration_time = time.perf_counter() - calibration_time_start
                lift_gammas = np.exp(-self.lift_gammas * dt)
                U = np.zeros((self.n_quad, num_paths))
            except:
                raise RuntimeError("Markovian lift calibration failed.")
        
        for j in range(num_steps):
            """
            X_n = X_n-1 - 0.25 (v_n + v_n-1) dt + chi_n
            V_n = Y_n^2 + c
            Y_n = yhat_n-1 + u_n
            """
            # Conditional variance expectation
            sqrt_vbar = np.sqrt(
                K_ratio * (self.a * (self.alp * (yhat - self.b)**2 + alp_complement * (Y - self.b)**2) + self.c)
            )
            
            # Calculating u_n
            sig_chi = sqrt_vbar * sqrt_dt
            sig_eps = sqrt_vbar * sqrt_sig_eps_constant

            if markovian_lift:
                chi_j[:] = sig_chi * Z_chi[:, j]
                u = beta_uchi * chi_j + sig_eps * Z_eps[:, j]
            else:
                chi[j, :] = sig_chi * Z_chi[:, j]
                u = beta_uchi * chi[j, :] + sig_eps * Z_eps[:, j]
            
            # Updating Y_n
            Y = yhat + u
            
            # Updating V_n
            vf = self.a * (Y - self.b)**2 + self.c
            dw = (V + vf) / 2 * dt
            w += dw
            
            # Updating X_n
            if markovian_lift:
                X -= 0.5 * dw + chi_j
            else:
                X -= 0.5 * dw + chi[j, :]
            
            # Update yhat for next iteration
            if j < num_steps - 1:
                # equiv to bstar[1 : j + 2][::-1]
                if markovian_lift:
                    # Markovian lift update
                    U = (U + chi_j[:]) * lift_gammas[:, None]
                    yhat = y0_field[j+1] + np.sum(self.lift_mus[:, None] * U, axis=0)
                else:
                    yhat = y0_field[j+1] + np.tensordot(bstar_rev[num_steps-j-2 : num_steps-1], chi[: j + 1], axes=1)
            V = vf

            # Check if current step matches any expiry
            current_step = j + 1
            if current_step in expiry_steps_set:
                # Find which expiry(ies) match this step
                expiry_idx = np.where(expiry_steps == current_step)[0]

                for idx in expiry_idx:
                    expiry = expiries[idx]
                    res_sim = {}
                    
                    if output in ["all", "spx"]:
                        res_sim["X"] = X.copy()
                        if interest_rates is not None:
                            res_sim["X"] += expiry * interest_rates_array[idx]
                    if output == "all":
                        res_sim["V"] = vf.copy()
                        res_sim["w"] = w.copy()
                    
                    results[expiry] = res_sim
                    exp_completed += 1
        
        return results
    
    def simulate_filtered_random(
        self,
        num_paths: int,
        num_steps: int,
        expiries,
        output="all",
        delvix=1.0 / 12.0,
        nvix=10,
        interest_rates: dict = None,
        markovian_lift: bool = False
    ):
        """
        Wrapper for simulate_filtered which generates random paths on demand.
        """
        mc_path_X = np.random.normal(size=(num_paths, num_steps))
        mc_path_V = np.random.normal(size=(num_paths, num_steps))
        return self.simulate_filtered(mc_path_V, mc_path_X, expiries, output, delvix, nvix, interest_rates, markovian_lift)
        