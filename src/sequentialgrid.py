import numpy as np
from matplotlib import pyplot as plt
from grid import Grid
from numba import njit, prange

class SequentialGrid(Grid):
    def __init__(self, n: int):
        super().__init__(n)

    def update(self, omega=1.7):
        max_diff = 0.0

        for i in range(1, self.n-1):
            for j in range(1, self.n-1):

                old = self.T[i, j]
                avg = 0.25 * ( self.T[i+1, j] + self.T[i-1, j] +
                            self.T[i, j+1] + self.T[i, j-1] )

                new = (1 - omega) * old + omega * avg
                self.T[i, j] = new

                diff = abs(new - old)
                if diff > max_diff:
                    max_diff = diff

        return max_diff


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

class ParallelGrid(Grid):
    def __init__(self, n: int):
        super().__init__(n)

    @staticmethod
    @njit(parallel=True)
    def _red_sweep(T, omega):
        n = T.shape[0]
        for i in prange(1, n - 1):
            for j in range(1, n - 1):
                if (i + j) % 2 == 0:
                    old = T[i, j]
                    avg = 0.25 * (
                        T[i + 1, j] + T[i - 1, j] +
                        T[i, j + 1] + T[i, j - 1]
                    )
                    new = (1.0 - omega) * old + omega * avg
                    T[i, j] = new

    @staticmethod
    @njit(parallel=True)
    def _black_sweep(T, omega):
        n = T.shape[0]
        for i in prange(1, n - 1):
            for j in range(1, n - 1):
                if (i + j) % 2 == 1:
                    old = T[i, j]
                    avg = 0.25 * (
                        T[i + 1, j] + T[i - 1, j] +
                        T[i, j + 1] + T[i, j - 1]
                    )
                    new = (1.0 - omega) * old + omega * avg
                    T[i, j] = new

    def update(self, omega=0.96):
        T_old = self.T.copy()
        ParallelGrid._red_sweep(self.T, omega)
        ParallelGrid._black_sweep(self.T, omega)
        inner_old = T_old[1:-1, 1:-1]
        inner_new = self.T[1:-1, 1:-1]
        diff = np.max(np.abs(inner_new - inner_old))
        return diff