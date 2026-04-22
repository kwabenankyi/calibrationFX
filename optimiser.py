import time
import traceback
import cma
import multiprocess as mp
import numpy as np
from multiprocess import shared_memory
from tqdm import tqdm

"""
Warm start function to generate initial parameters for CMA-ES, based on Nomura et al.(2021).

Simulator class for optimisation routines, inspired by the solution to a 
StackOverflow question on how to track function evaluations and their results during optimisation.

Anthony Nkyi, March 2026.

Sources:
1. Masahiro Nomura, Shuhei Watanabe, Youhei Akimoto, Yoshihiko Ozaki, and Masaki Onishi. 
Warm Starting CMA-ES for hyperparameter optimization. Proceedings of the AAAI Conference on 
Artificial Intelligence, 35(10):9188–9196, 2021. doi:10.1609/aaai.v35i10.17109.

2. https://stackoverflow.com/a/59330005, Posted by Henri. Retrieved 2026-04-04, License - CC BY-SA 4.0
"""

def warm_start(prior_X, prior_loss, gamma, alpha, bounds):
    N = len(prior_loss)
    N_gamma = max(1, int(gamma * N))

    # Filter for top solutions
    top_indices = np.argsort(prior_loss)[:N_gamma]
    X_top = prior_X[top_indices]

    # Calculate optimal initial mean and covariance
    m_star = np.mean(X_top, axis=0)
    cov_top = np.cov(X_top, rowvar=False, ddof=0)

    bounds_arr = np.array(bounds)
    if bounds_arr.shape[0] == 2:
        range_widths = bounds_arr[1] - bounds_arr[0]
    else:
        range_widths = bounds_arr[:, 1] - bounds_arr[:, 0]
    
    m_star = np.clip(m_star, bounds_arr[0], bounds_arr[1])
    # The minimum variance is scaled proportionally to the bounds width
    min_variance_vec = (alpha * range_widths) ** 2
    Sigma_star = np.diag(min_variance_vec) + cov_top
    # Override guesses with the single optimal starting point
    return [m_star], Sigma_star

# Solution inspired by: https://stackoverflow.com/a/59330005
# Posted by Henri, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-04, License - CC BY-SA 4.0
class OptimiserSimulator:
    def __init__(self, function):
        """
        Initialize the simulator with the objective function.
        Parameters:
        function (callable): The objective function to be optimized. It should take an input (array-like) x
        """
        self.f = function # actual objective function
        self.num_calls = 0 # how many times f has been called
        self.callback_count = 0 # number of times callback has been called, also measures iteration count
        self.list_calls_inp = [] # input of all calls
        self.list_calls_res = [] # result of all calls
        self.decreasing_list_calls_inp = [] # input of calls that resulted in decrease
        self.decreasing_list_calls_res = [] # result of calls that resulted in decrease
        self.list_callback_inp = [] # only appends inputs on callback, as such they correspond to the iterations
        self.list_callback_res = [] # only appends results on callback, as such they correspond to the iterations
        self.best_res = None # best result so far
        self.sim_times = [] # length of each simulation, to be used for plotting

    def simulate(self, x, *args):
        """
        Executes the actual simulation and returns the result, while
        updating the lists too. Pass to optimizer without arguments or
        parentheses.
        Parameters:
        x (array-like): The input to the objective function.
        *args: Additional arguments to be passed to the objective function.
        Returns:
        The result of evaluating the objective function at x.
        """
        start_time = time.perf_counter()
        result = self.f(x, *args) # the actual evaluation of the function
        self.sim_times.append(time.perf_counter() - start_time)
        if not self.num_calls: # first call is stored in all lists
            self.decreasing_list_calls_inp.append(x)
            self.decreasing_list_calls_res.append(result)
            self.list_callback_inp.append(x)
            self.list_callback_res.append(result)
        elif result < self.decreasing_list_calls_res[-1]:
            self.decreasing_list_calls_inp.append(x)
            self.decreasing_list_calls_res.append(result)
        self.list_calls_inp.append(x)
        self.list_calls_res.append(result)
        self.num_calls += 1
        return result

    def callback(self, xk, *_):
        """
        Callback function that can be used by optimizers of scipy.optimize.
        The third argument "*_" makes sure that it still works when the
        optimizer calls the callback function with more than one argument. Pass
        to optimizer without arguments or parentheses.
        """
        best = False
        s1 = "Run {:d}: ".format(self.callback_count + 1)
        len_s1 = len(s1)
        xk = np.atleast_1d(xk)
        # search backwards in input list for input corresponding to xk
        for i, x in reversed(list(enumerate(self.list_calls_inp))):
            x = np.atleast_1d(x)
            if np.allclose(x, xk):
                break
        
        for comp in xk:
            s1 += f"{comp:10.5e}\t"
        
        val = self.list_calls_res[i]
        s1 += f"{val:10.5e}"

        if self.best_res is None or val < self.best_res:
            best = True
            self.best_res = val
            s1 += " *" # mark with a star if it's the best result so far

        self.list_callback_inp.append(xk)
        self.list_callback_res.append(val)

        if not self.callback_count:
            s0 = " " * len_s1
            for j, _ in enumerate(xk):
                tmp = f"Comp-{j+1}"
                s0 += f"{tmp:10s}\t"
            s0 += "Objective"
            print(s0)
        if best or (self.callback_count+1) % 100 == 0:
            print(s1)
        self.callback_count += 1

class GlobalOptimiser:
    def __init__(self, obj, converge_constraint, worker_init, ui_callback=None, num_cores=None):
        """
        Stateful runner for WS-IPOP-CMA-ES with shared-memory multiprocessing.

        Parameters:
        obj (callable): Worker-safe objective function evaluated in the process pool.
        converge_constraint (callable): Constraint function used in CMA ``constraints_values``.
        worker_init (callable): Pool initialiser used to hydrate worker globals from shared memory.
        ui_callback (callable|None): Optional callback used to pump GUI events.
        num_cores (int|None): Optional fixed worker count; if None it auto-selects.
        """
        self.obj = obj
        self.converge_constraint = converge_constraint
        self.worker_init = worker_init
        self.ui_callback = ui_callback
        if num_cores is None:
            num_cores = mp.cpu_count() // 2
        self.num_cores = num_cores
        self.reset_state()

    def reset_state(self):
        self.options_used = None
        self.use_warm_start = False
        self.sigma0 = None
        self.x0_list = None
        self.global_best_params = None
        self.global_best_loss = np.inf
        self.es_strategies = []
        self.times_per_generation = []
        self.run_records = []
        self.total_wall_clock = 0.0
        self.run_result = None

    def _pump_ui(self, force=False):
        if self.ui_callback is None:
            return
        try:
            self.ui_callback(force=force)
        except TypeError:
            self.ui_callback()

    def run_ws_ipop_cma_multiprocessed(
        self,
        options,
        bounds,
        range_widths,
        mc_path_X,
        mc_path_Variance,
        worker_init_args_tail,
        prior_X=None,
        prior_loss=None,
        max_restarts=2,
        incpop_factor=2,
        gamma=1,
        alpha=0.25,
        fallback_x0=None,
        fallback_sigma0=0.25,
        logger_path_fn=None,
    ):
        self.reset_state()

        if fallback_x0 is None:
            raise ValueError("fallback_x0 must be provided")

        options = dict(options) if options is not None else {
            "bounds": bounds,
            "CMA_stds": range_widths,
            "popsize": 16,
            "maxfevals": 10000,
            "tolfun": 1e-8,
            "verbose": -9,
            "verb_disp": 0,
            "seed": 79,
        }
        options.setdefault("bounds", bounds)
        options.setdefault("CMA_stds", range_widths)
        self.options_used = options.copy()

        # Warm start calculation (WS-IPOP-CMA-ES)
        self.use_warm_start = prior_X is not None and prior_loss is not None
        bounds_arr = np.array(bounds)

        if self.use_warm_start:
            print("\n--- Initialising WS-IPOP-CMA-ES with Prior Knowledge ---")
            self.sigma0 = 0.75
            self.x0_list, Sigma_star = warm_start(prior_X, prior_loss, gamma, alpha, bounds_arr)
        else:
            print("\n--- No prior data provided. Using fallback initial guesses. ---")
            self.x0_list = np.array(fallback_x0)
            self.sigma0 = fallback_sigma0
            Sigma_star = None

        num_cores = self.num_cores
        print(f"Starting parallel IPOP-CMA-ES optimisation across {num_cores} cores using Shared Memory...")

        lower_bounds_arr = np.array(options["bounds"][0])
        upper_bounds_arr = np.array(options["bounds"][1])

        start_total = time.perf_counter()
        shm_X = shared_memory.SharedMemory(create=True, size=mc_path_X.nbytes)
        shm_Var = shared_memory.SharedMemory(create=True, size=mc_path_Variance.nbytes)

        try:
            shared_X = np.ndarray(mc_path_X.shape, dtype=mc_path_X.dtype, buffer=shm_X.buf)
            shared_Var = np.ndarray(mc_path_Variance.shape, dtype=mc_path_Variance.dtype, buffer=shm_Var.buf)
            np.copyto(shared_X, mc_path_X)
            np.copyto(shared_Var, mc_path_Variance)

            init_args = (
                shm_X.name,
                shared_X.shape,
                shared_X.dtype,
                shm_Var.name,
                shared_Var.shape,
                shared_Var.dtype,
                *worker_init_args_tail,
            )

            base_popsize = options["popsize"]

            with mp.Pool(processes=num_cores, initializer=self.worker_init, initargs=init_args) as pool:
                for guess_idx, x0 in enumerate(self.x0_list):
                    current_x0 = np.array(x0, dtype=float)
                    print("\n=======================================================")
                    if self.use_warm_start:
                        print("> Evaluating Warm-Started Mean (m_star)")
                    else:
                        print(f"> Evaluating Initial Guess {guess_idx + 1}/{len(self.x0_list)}")
                    print(f"> Starting params: {current_x0}")
                    print(f"> Initial Loss: {self.obj(current_x0):.8e}")
                    print("=======================================================")

                    current_popsize = base_popsize

                    for run in range(max_restarts + 1):
                        print(f"\n--- Starting IPOP Run {run + 1}/{max_restarts + 1} ---")
                        print(f"Population Size: {current_popsize}")

                        run_options = options.copy()
                        run_options["popsize"] = current_popsize

                        es = cma.CMAEvolutionStrategy(current_x0, self.sigma0, inopts=run_options)

                        # Warm-start override: inject custom covariance matrix on the first run only.
                        if self.use_warm_start and run == 0 and Sigma_star is not None:
                            scale_matrix = (self.sigma0**2) * np.outer(range_widths, range_widths)
                            es.C = Sigma_star / scale_matrix

                            D2, B = np.linalg.eigh(es.C)
                            es.D = np.sqrt(np.maximum(D2, 1e-18))
                            es.B = B
                            es.invsqrtC = np.dot(es.B, np.diag(1.0 / es.D)).dot(es.B.T)

                            print("--> Injecting Normalised Warm-Start Covariance Matrix...")
                            print(
                                f" [Debug] True Initial Step Sizes: "
                                f"{es.sigma * np.sqrt(np.diag(es.C)) * range_widths}"
                            )

                        if logger_path_fn is not None:
                            logger_path = logger_path_fn(guess_idx, run)
                        else:
                            logger_path = f"outcmaes/guess{guess_idx}_run{run}/"
                        es.logger = cma.CMADataLogger(logger_path).register(es)

                        res = None
                        t_per_gen = []
                        wall_clock_start = time.perf_counter()

                        try:
                            while not es.stop():
                                tick = time.perf_counter()
                                raw_solutions = es.ask()
                                clipped_solutions = [
                                    np.clip(sol, lower_bounds_arr, upper_bounds_arr) for sol in raw_solutions
                                ]

                                fitnesses = []
                                for fit in tqdm(
                                    pool.imap(self.obj, clipped_solutions),
                                    total=len(clipped_solutions),
                                    desc=f"Gen {es.result.iterations} | Best Loss: {es.result.fbest:.8e}",
                                    leave=False,
                                ):
                                    fitnesses.append(fit)
                                    self._pump_ui()

                                if self.converge_constraint is not None:
                                    constraints_values = [
                                        [self.converge_constraint(np.array(sol))]
                                        for sol in clipped_solutions
                                    ]
                                    es.tell(
                                        raw_solutions,
                                        fitnesses,
                                        constraints_values=constraints_values,
                                    )
                                else:
                                    es.tell(raw_solutions, fitnesses)

                                es.logger.add()
                                es.disp()
                                self._pump_ui(force=True)
                                t_per_gen.append(time.perf_counter() - tick)

                            res = es.result

                        except Exception as err:
                            print(f"Error during optimisation run {run + 1}: {err}")
                            traceback.print_exc()
                            try:
                                res = es.result
                            except Exception:
                                res = None

                        self.es_strategies.append(es)

                        if res is None or res.xbest is None:
                            print(f"Run {run + 1} produced no valid result — skipping.")
                            self.run_records.append(
                                {
                                    "guess_idx": guess_idx,
                                    "run": run,
                                    "popsize": current_popsize,
                                    "valid": False,
                                    "stop": dict(es.stop()),
                                }
                            )
                            current_popsize *= incpop_factor
                            continue

                        run_wall_clock = time.perf_counter() - wall_clock_start
                        mean_time_per_gen = float(np.mean(t_per_gen)) if t_per_gen else np.nan

                        run_record = {
                            "guess_idx": guess_idx,
                            "run": run,
                            "popsize": current_popsize,
                            "mean_time_per_gen": mean_time_per_gen,
                            "wall_clock": run_wall_clock,
                            "loss": float(res.fbest),
                            "params": np.array(res.xbest, copy=True),
                            "generations": int(res.iterations),
                            "valid": True,
                            "stop": dict(es.stop()),
                        }
                        self.run_records.append(run_record)
                        self.times_per_generation.append(run_record)

                        print(f"Run {run + 1} Stopped: {es.stop()} | Iterations: {res.iterations}")
                        print(f"Run {run + 1} Best Loss: {res.fbest:.8e}")
                        print(f"Run {run + 1} Best Params: {res.xbest}")
                        print(
                            f"Run {run + 1} Time per Gen: {mean_time_per_gen:.8f}s "
                            f"| Total Wall Clock: {run_wall_clock:.8f}s"
                        )

                        if res.fbest < self.global_best_loss:
                            print("  -> New Global Best Found! Updating global best loss and parameters.")
                            print(
                                f"  -> New Best Parameters: {res.xbest} "
                                f"| New Best Loss: {res.fbest:.8e}"
                            )
                            self.global_best_loss = float(res.fbest)
                            self.global_best_params = np.array(res.xbest, copy=True)

                        current_popsize *= incpop_factor
                        current_x0 = self.global_best_params if self.global_best_params is not None else current_x0

        except KeyboardInterrupt:
            print("\nOptimisation interrupted by user. Initiating safe shutdown...")
        finally:
            self.total_wall_clock = time.perf_counter() - start_total
            try:
                shm_X.close()
                shm_X.unlink()
            except FileNotFoundError:
                pass

            try:
                shm_Var.close()
                shm_Var.unlink()
            except FileNotFoundError:
                pass

        self.run_result = {
            "params": self.global_best_params,
            "loss": self.global_best_loss,
            "es": self.es_strategies,
            "times_per_generation": self.times_per_generation,
            "run_records": self.run_records,
            "use_warm_start": self.use_warm_start,
            "sigma0": self.sigma0,
            "x0_list": self.x0_list,
            "num_cores": self.num_cores,
            "total_wall_clock": self.total_wall_clock,
        }
        return self.run_result
