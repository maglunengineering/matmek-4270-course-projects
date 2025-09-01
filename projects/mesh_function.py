import numpy as np
from collections.abc import Callable


def mesh_function(f: Callable[[float], float], t: np.ndarray) -> np.ndarray:
    f_t = np.zeros_like(t, dtype=float)
    for i, _t in enumerate(t):
        f_t[i] = f(_t)
    return f_t


def func(t: float) -> float:
    if 0 <= t and t <= 3:
        return np.exp(-t)
    elif 3 < t <= 4:
        return np.exp(-3*t)
    else:
        raise ValueError('')


def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)

if __name__ == "__main__":
    test_mesh_function()
