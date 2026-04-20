import numpy as np
def param_arr_to_dict(params):
    return {"c": params[0], "nu": params[1], "lam": params[2], "al": params[3]}

def const_param_arr_to_dict(params):
    return {"c": params[0], "nu": params[1], "lam": params[2], "al": params[3], "a": params[4], "b": params[5]}

def dict_to_param_arr(params_dict):
    return np.array([params_dict["c"], params_dict["nu"], params_dict["lam"], params_dict["al"]])

def const_dict_to_param_arr(params_dict):
    return np.array([params_dict["c"], params_dict["nu"], params_dict["lam"], params_dict["al"], params_dict["a"], params_dict["b"]])