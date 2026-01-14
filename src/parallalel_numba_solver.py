import numpy as np
from numba import njit, prange

from grid import Grid


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
