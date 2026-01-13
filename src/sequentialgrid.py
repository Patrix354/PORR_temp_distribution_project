import numpy as np
from numba import njit

from grid import Grid


class SequentialGrid(Grid):
    def __init__(self, n: int):
        super().__init__(n)

    @staticmethod
    @njit(parallel=False)
    def _update(T, omega=1.7):
        max_diff = 0.0
        n = T.shape[0]
        for i in range(1, n-1):
            for j in range(1, n-1):

                old = T[i, j]
                avg = 0.25 * ( T[i+1, j] + T[i-1, j] +
                            T[i, j+1] + T[i, j-1] )

                new = (1 - omega) * old + omega * avg
                T[i, j] = new

    def update(self, omega=1.7):
        T_old = self.T.copy()
        SequentialGrid._update(self.T, omega)
        inner_old = T_old[1:-1, 1:-1]
        inner_new = self.T[1:-1, 1:-1]
        diff = np.max(np.abs(inner_new - inner_old))
        return diff


class AnalyticGrid(Grid):
    def __init__(self, n: int, n_terms: int = 200):
        super().__init__(n)
        self.n_terms = n_terms

    def compute(self):
        h = 1.0 / (self.n - 1)

        for i in range(self.n):
            x = i * h
            for j in range(self.n):
                # węzły brzegowe są ustalone
                if i == 0 or i == self.n - 1 or j == 0 or j == self.n - 1:
                    continue

                y = j * h
                self.T[i, j] = self._T_analytic(x, y)

    def _T_analytic(self, x: float, y: float):
        s = 0.0
        for n in range(1, self.n_terms + 1):
            # parzyste
            if n % 2 == 0:
                continue
            # nieparzyste
            Cn = -8.0 / (n * np.pi * (n**2 - 4) * np.sinh(n * np.pi))
            s += Cn * np.sinh(n * np.pi * (1.0 - x)) * np.sin(n * np.pi * y)

        return s

