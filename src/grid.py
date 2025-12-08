import numpy as np
from matplotlib import pyplot as plt
import time

class Grid:
    def __init__(self, n: int):
        if n < 2:
            raise ValueError("n musi być >= 2")

        self.n = n
        self.h = 1.0 / (n - 1)

        self.T = np.zeros((n, n), dtype=float)

        self.x = np.linspace(0, 1, n)
        self.y = np.linspace(0, 1, n)

        self.apply_boundary_conditions()

    def apply_boundary_conditions(self):
        for j in range(self.n):
            y = self.y[j]
            self.T[0, j] = np.sin(np.pi * y)**2

    def print_heatmap(self, delay=0.001):
        plt.ion()
        plt.clf()
        plt.imshow(self.T, cmap='viridis')
        plt.colorbar()
        plt.draw()
        plt.pause(delay)
    
    def __repr__(self):
        return f"Grid(n={self.n}, h={self.h:.4f})"