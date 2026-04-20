import enum
import numpy as np
from scipy.stats import norm
    
def brownianPaths(num_paths, num_steps):
    return np.random.normal(size=(num_paths, num_steps))

class OptionType(enum.IntEnum):
    CALL = 1.0
    PUT = -1.0
