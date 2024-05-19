#!/home/henu/brogramming/audio_stuff/.venv/bin/python3.12
"""
Discrete Fourier Transform functionality
"""

import numpy as np

from time import perf_counter
from typing import Any, Callable


def timer(f: Callable) -> Callable:
    """
    Decorator to time function calls
    :param f: Function to be timed
    :return:
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """
        :param args: Positional arguments for the timed function
        :param kwargs: Keyword arguments for the timed function
        :return:
        """
        s = perf_counter()
        ret = f(*args, **kwargs)
        e = perf_counter()
        print(f"{f.__name__} ran in {e - s:.3f} s")
        return ret
    return wrapper


def hann(y: np.ndarray) -> np.ndarray:
    """
    Hann window, see https://en.wikipedia.org/wiki/Window_function
    :param y:
    :return:
    """
    n = y.shape[0]
    nn = np.array(range(0, n))
    return y * .5 * (1 - np.cos(2 * np.pi * nn / n))


@timer
def rdft(y: np.ndarray) -> np.ndarray:
    """
    Is supposed to compute the Discrete Fourier Transform on the given real
    valued data

    See https://en.wikipedia.org/wiki/Discrete_Fourier_Transform
    :param y:
    :return:
    """
    n = y.shape[0]
    nhalf = n // 2 + 1
    out = np.zeros(shape=(nhalf,), dtype=complex)
    for i in range(nhalf):
        summa = 0
        for j in range(n):
            summa += y[j] * np.exp(-2j * np.pi * i / n * j)
        out[i] = summa
    return out
        
