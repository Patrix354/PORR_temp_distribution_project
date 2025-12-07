import numpy as np
from sequentialgrid import SequentialGrid, AnalyticGrid

def main():
    g = SequentialGrid(n=40)
    diff = 1
    eps = 0.000000000001

    ana = AnalyticGrid(n=40, n_terms=200)
    ana.compute()

    while True:
        diff = g.update()
        # g.print_heatmap()
        if diff < eps:
            input('Press any key to continue')
            break

    diff = np.abs(g.T - ana.T)
    print("max |T_sequential - T_analytic| =", diff.max())

    ana.print_heatmap()
    input('Press any key to continue')

if __name__ == "__main__":
    main()
