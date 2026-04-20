from scipy.special import gamma
from scipy import integrate
import numpy as np

def converge_test_val(nu, lam, al):
    H = al - 1 / 2
    return (nu / gamma(al)) ** 2 * gamma(2 * H) / (2 * lam) ** (2 * H)

def converge_test_obj(nu, lam, al):
    H = al - 1 / 2
    integral = (nu / gamma(al)) ** 2 * gamma(2 * H) / (2 * lam) ** (2 * H)
    return integral < 1.0

def converge_test(params, opt="closed-form"):
    al = params["al"]
    H = al - 1 / 2
    lam = params["lam"]
    nu = params["nu"]

    if opt == "closed-form":
        integral = (nu / gamma(al)) ** 2 * gamma(2 * H) / (2 * lam) ** (2 * H)
    elif opt == "quad":

        def kernel(x):
            return (nu / gamma(al)) * x ** (al - 1) * np.exp(-lam * x)

        integral = integrate.quad(lambda x: kernel(x) ** 2, 0, np.inf)[0]
    else:
        raise ValueError("Check value for opt.")

    assert integral < 1
    return integral