import numpy as np
import time

"""
Warm start function to generate initial parameters for CMA-ES, based on Nomura et al.(2021).

Simulator class for optimisation routines, inspired by the solution to a 
StackOverflow question on how to track function evaluations and their results during optimization.

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
class Simulator:
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
