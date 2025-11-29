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
