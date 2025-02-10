import numpy as np
import torch as th
from collections.abc import Sequence


# simulate data from two moons dataset
def two_moons(n: int, sigma: float = 1e-1):
    theta = 2 * th.pi * th.rand(n)
    label = (theta > th.pi).float()

    x = th.stack(
        (th.cos(theta) + label - 1 / 2,
         th.sin(theta) + label / 2 - 1 / 4),
        axis=-1)
    return th.normal(x, sigma), label

def gaussian_simulator(theta, n=1, cov=.99):
    if isinstance(theta, Sequence):
        return np.array([gaussian_simulator(t).squeeze() for t in theta])
    
    f = lambda t : (1.5 * t)**3 / 200
    x = np.random.normal(loc=f(theta), scale=cov, size=n)
    return x