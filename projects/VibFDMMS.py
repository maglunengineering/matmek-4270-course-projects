import numpy as np
import sympy as sp
import VibFD

t = VibFD.t

class VibFD2MMF(VibFD.VibFD2):
    """
    Second order accurate solver using any (?)boundary conditions as long as they satisfy 
    the assumed solution, or something
    """

    def __init__(self, Nt: int, T: float, w: float = 0.35, assumed_u: sp.Expr = None) -> None:
        VibFD.VibSolver.__init__(self, Nt, T, w, assumed_u.subs(t, 0))

        self.assumed_u = assumed_u

        # We'll have something like assumed_u = t⁴ or exp(sin(t)),
        # boundary conditions can be anything that assumed_u can satisfy

        self.f:sp.Expr = assumed_u.diff(t).diff(t) + self.w**2 * assumed_u

    
    # The exact solution ue() is now (by definition?) assumed_u
    def ue(self) -> sp.Expr:
        return self.assumed_u
    
    def __call__(self):
        rhs = np.array([self.f.subs(t, _t) for _t in self.t], dtype=float)
        rhs[0] = self.assumed_u.subs(t, 0)
        rhs[-1] = self.assumed_u.subs(t, self.T)

        dt = self.T / self.Nt
        D = np.zeros((self.Nt + 1, self.Nt + 1))

        idiag = np.arange(self.Nt + 1)
        D[idiag, idiag] = -2.0 + (self.w * dt)**2 # Main diagonal is this (-g or whatever) 
        D[idiag[:-1], (idiag+1)[:-1]] = 1.0 # Neighboring diagonals are 1.0/dt²
        D[(idiag+1)[:-1], idiag[:-1]] = 1.0

        D *= 1.0/dt**2

        D[0, 0:2] = [1.0, 0.0]
        D[-1, -2:] = [0.0, 1.0]

        u = np.linalg.solve(D, rhs)
        return u


def test_order_mmf():
    w = 0.35

    VibFD2MMF(8, 1.0, w, t**4).test_order()
    VibFD2MMF(8, 1.0, w, sp.exp(sp.sin(t))).test_order()

if __name__ == '__main__':
    test_order_mmf()