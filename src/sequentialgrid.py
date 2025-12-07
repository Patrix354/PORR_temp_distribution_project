import numpy as np
from matplotlib import pyplot as plt
from grid import Grid

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