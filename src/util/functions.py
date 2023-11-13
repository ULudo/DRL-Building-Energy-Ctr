import os
import sys
import random
import numpy as np
import torch
from contextlib import contextmanager

C_IN_K = 273.15
KILO = 1000

s_to_hour = lambda x: x/3600

times_kilo = lambda x: x*KILO
through_kilo = lambda x: x/KILO

k_to_c = lambda x: x-C_IN_K
c_to_k = lambda x: x+C_IN_K

j_to_kwh = lambda x: x / (60**2 * KILO)
kwh_to_j = lambda x: x * 3.6e6

class JToKw:
    def __init__(self, min) -> None:
        self.min = min
    
    def __call__(self, x):
        return x / (self.min * 60 * KILO)

byte_to_gb = lambda x: x / 1024**3

min_to_s = lambda x: x * 60

def cyclic_encode(value, trigonometric_fun, max_val):
    if False:
        x = value if value < max_val else max_val
    else: 
        x = np.rint(value)
    trig = trigonometric_fun(2 * np.pi * x/max_val)
    return trig

def sin_encode_hour(value):
    return cyclic_encode(value, np.sin, 24)

def cos_encode_hour(value):
    return cyclic_encode(value, np.cos, 24)

def sin_encode_day(value):
    return cyclic_encode(value, np.sin, 7)

def cos_encode_day(value):
    return cyclic_encode(value, np.cos, 7)

def sin_encode_month(value):
    return cyclic_encode(value, np.sin, 12)

def cos_encode_month(value):
    return cyclic_encode(value, np.cos, 12)

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seeds(seed):
    # For Python random
    random.seed(seed)
    
    # For PyTorch RNG
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For Numpy RNG
    np.random.seed(seed)
    

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
