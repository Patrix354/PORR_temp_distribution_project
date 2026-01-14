import time
import numpy as np
import matplotlib.pyplot as plt

from sequentialgrid import SequentialGrid, AnalyticGrid
from parallalel_numba_solver import ParallelGrid
from parralel_multiprocessing_solver import ParallelSolver

EPS = 1e-12
N = 40
MAX_ITER_MULTIPROC = 100000

class ResultWrapper:
    def __init__(self, T_matrix):
        self.T = T_matrix


def run_iterative_solver(GridCls, n, eps, name):
    print(f"Running {name}...")
    g = GridCls(n=n)
    iters = 0
    t0 = time.perf_counter()
    while True:
        diff = g.update()
        iters += 1
        if diff < eps:
            break
    t1 = time.perf_counter()
    return g, t1 - t0, iters


def run_multiprocessing_solver(n, eps, name):
    print(f"Running {name}...")
    template = SequentialGrid(n=n)
    solver = ParallelSolver(n=n, initial_grid=template.T)
    T_result, duration, iters = solver.solve(n_procs=16, epsilon=eps, max_iter=MAX_ITER_MULTIPROC)
    return ResultWrapper(T_result), duration, iters


# --- MAIN ---

def main():
    print(f"=== SIMULATION START (N={N}, EPS={EPS}) ===\n")
    results = {}
    results["Sequential"] = run_iterative_solver(SequentialGrid, N, EPS, "Sequential")
    results["Parallel (Numba)"] = run_iterative_solver(ParallelGrid, N, EPS, "Parallel (Numba)")
    results["Parallel (MultiProc)"] = run_multiprocessing_solver(N, EPS, "Parallel (MultiProc)")

    print("\n" + "=" * 60)
    print(f"{'METHOD':<25} | {'TIME (s)':<12} | {'ITERATIONS'}")
    print("-" * 60)
    for name, (grid, duration, iters) in results.items():
        print(f"{name:<25} | {duration:<12.4f} | {iters}")
    print("=" * 60 + "\n")
    print("=== ERRORS VS ANALYTIC SOLUTION ===")

    ana = AnalyticGrid(n=N, n_terms=200)
    ana.compute()

    for name, (grid, duration, iters) in results.items():
        diff_matrix = np.abs(grid.T - ana.T)
        max_diff = diff_matrix.max()
        print(f"max |T_{name:<20} - T_analytic| = {max_diff:.6e}")

        plt.figure(figsize=(6, 5))
        plt.title(f"Error Distribution: {name}\nN={N}, Max Error={max_diff:.2e}")
        plt.imshow(grid.T - ana.T, cmap='viridis')  # Różnica ze znakiem
        plt.colorbar(label='T_calc - T_analytic')
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(f"error_{safe_name}_{N}.png")
        plt.close()

    print("\n=== CROSS VALIDATION (Consistency Check) ===")
    seq_grid = results["Sequential"][0]
    for name, (grid, _, _) in results.items():
        if name == "Sequential": continue
        diff = np.abs(seq_grid.T - grid.T).max()
        print(f"max |T_Sequential - T_{name:<18}| = {diff:.6e}")


if __name__ == "__main__":
    main()