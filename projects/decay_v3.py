import numpy as np
import matplotlib.pyplot as plt


def solver(
    I: float, a: float, T: float, dt: float, theta: float
) -> tuple[np.ndarray, float]:
    """Solve u'=-a*u, u(0)=I, for t in (0,T] with steps of dt."""
    Nt = int(T / dt)  # no of time intervals
    T = Nt * dt  # adjust T to fit time step dt
    u = np.zeros(Nt + 1)  # array of u[n] values
    t = np.linspace(0, T, Nt + 1)  # time mesh

    u[0] = I  # assign initial condition
    for n in range(0, Nt):
        u[n + 1] = (1 - (1 - theta) * a * dt) / (1 + theta * dt * a) * u[n]
    return u, t


def u_exact(t: float, I: float, a: float) -> float:
    return I * np.exp(-a * t)


def plot_numerical_and_exact(theta: float, I: float, a: float, T: float, dt: float):
    """Compare the numerical and exact solution in a plot."""
    u, t = solver(I, a, T, dt, theta)

    t_e = np.linspace(0, T, 1001)  # fine mesh for u_e
    u_e = u_exact(t_e, I, a)

    plt.plot(
        t,
        u,
        "r--o",  # red dashes w/circles
        t_e,
        u_e,
        "b-",
    )  # blue line for exact sol.
    plt.legend(["numerical", "exact"])
    plt.xlabel("t")
    plt.ylabel("u")
    plt.title("theta=%g, dt=%g" % (theta, dt))
    plt.savefig("plot_%s_%g.png" % (theta, dt))


def test_solver_three_steps():
    """Compare three steps with known manual computations."""
    theta, a, I, dt = 0.8, 2, 0.1, 0.8
    u_by_hand = np.array([I, 0.0298245614035, 0.00889504462912, 0.00265290804728])

    Nt = 3  # number of time steps
    u, t = solver(I, a, Nt * dt, dt, theta)

    tol = 1e-12  # tolerance for comparing floats
    diff = np.abs(u - u_by_hand).max()
    success = diff < tol
    assert success
