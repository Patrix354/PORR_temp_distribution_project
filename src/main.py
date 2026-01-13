import time
import numpy as np
from matplotlib import pyplot as plt
from sequentialgrid import SequentialGrid, AnalyticGrid, ParallelGrid

EPS = 1e-12
N = 160


def run_until_convergence(GridCls, n=N, eps=EPS, name=""):
    print(f"=== Running {name or GridCls.__name__} ===")
    g = GridCls(n=n)
    iters = 0
    t0 = time.perf_counter()
    while True:
        diff = g.update()
        iters += 1
        if diff < eps:
            break
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"{name or GridCls.__name__}: "
          f"time = {elapsed:.4f} s, iterations = {iters}")
    print()
    return g, elapsed, iters


def main():
    print(f"n = {N}")
    results = {}
    g_seq, t_seq, it_seq = run_until_convergence(
        SequentialGrid, n=N, eps=EPS, name="sequential"
    )
    results["sequential"] = (g_seq, t_seq, it_seq)
    g_par, t_par, it_par = run_until_convergence(
        ParallelGrid, n=N, eps=EPS, name="parallel"
    )
    results["parallel"] = (g_par, t_par, it_par)
    print("=== SUMMARY: EXECUTION TIMES AND ITERATIONS ===")
    for name, (g, t, it) in results.items():
        print(f"{name:10s} -> time = {t:.4f} s, iterations = {it}")
    print()
    ana = AnalyticGrid(n=N, n_terms=200)
    ana.compute()
    print("=== ERRORS VS ANALYTIC SOLUTION ===")
    for name, (g, t, it) in results.items():
        diff = np.abs(g.T - ana.T)
        print(f"max |T_{name} - T_analytic| = {diff.max()}")

        diff_temp = g.T - ana.T
        plt.ion()
        plt.clf()
        plt.imshow(diff_temp, cmap='viridis')
        plt.colorbar()
        plt.savefig(f"roznica_{name}_{N}.png")
        plt.close()   

    diff_sp = np.abs(results["sequential"][0].T - results["parallel"][0].T)
    print()
    print("max |T_sequential - T_parallel| =", diff_sp.max())

if __name__ == "__main__":
    main()
