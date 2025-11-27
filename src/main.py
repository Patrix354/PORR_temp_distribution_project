from sequentialgrid import SequentialGrid

def main():
    g = SequentialGrid(n=40)
    diff = 1
    eps = 0.000000000001

    while True:
        diff = g._update_()
        g._print_heatmap_()
        if diff < eps:
            input('Press any key to continue')
            break

if __name__ == "__main__":
    main()
