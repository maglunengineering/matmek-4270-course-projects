import numpy as np


def differentiate(u: np.ndarray, dt: float) -> np.ndarray:
    d_n = np.zeros_like(u, dtype=float)
    d_n[0] = (u[1] - u[0])/dt

    for n in range(1, len(u) - 1):
        d_n[n] = (u[n+1] - u[n-1]) / (2*dt)

    d_n[-1] = (u[-1] - u[-2])/dt

    return d_n

def differentiate_vector(u: np.ndarray, dt: float) -> np.ndarray:
    d_n = np.zeros_like(u)
    
    d_n[0] = (u[1] - u[0])/dt
    d_n[1:-1] = (u[2:] - u[:-2])/(2*dt)
    d_n[-1] = (u[-1] - u[-2])/dt

    return d_n

def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1, du2)

if __name__ == '__main__':
    test_differentiate()
    