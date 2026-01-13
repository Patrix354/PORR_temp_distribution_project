# mpi_sor.py
from mpi4py import MPI
import numpy as np
from sequentialgrid import AnalyticGrid

def decompose_rows(n, size, rank):

    base = n // size
    rem = n % size

    if rank < rem:
        local_n = base + 1
        start = rank * local_n
    else:
        local_n = base
        start = rem * (base + 1) + (rank - rem) * base

    return start, local_n


def apply_local_boundary_conditions(T_old, i_start, n, h):

    local_n = T_old.shape[0] - 2

    for i_loc in range(1, local_n + 1):
        i_global = i_start + (i_loc - 1)

        if i_global == 0:
            for j in range(n):
                y = j * h
                T_old[i_loc, j] = np.sin(np.pi * y) ** 2


def exchange_halos(comm, T_old, rank, size):

    local_n = T_old.shape[0] - 2

    if rank < size - 1:
        comm.Send(T_old[local_n, :], dest=rank + 1, tag=0)
    if rank > 0:
        comm.Recv(T_old[0, :], source=rank - 1, tag=0)

    if rank > 0:
        comm.Send(T_old[1, :], dest=rank - 1, tag=1)
    if rank < size - 1:
        comm.Recv(T_old[local_n + 1, :], source=rank + 1, tag=1)


def mpi_weighted_jacobi(n, eps=1e-12, omega=1, max_iters=60000):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    h = 1.0 / (n - 1)

    i_start, local_n = decompose_rows(n, size, rank)

    T_old = np.zeros((local_n + 2, n), dtype=np.float64)
    T_new = np.zeros_like(T_old)

    apply_local_boundary_conditions(T_old, i_start, n, h)
    apply_local_boundary_conditions(T_new, i_start, n, h)

    iters = 0
    while True:
        iters += 1

        exchange_halos(comm, T_old, rank, size)

        T_new[:, :] = T_old

        max_diff_local = 0.0

        for i_loc in range(1, local_n + 1):
            i_global = i_start + (i_loc - 1)

            if i_global == 0 or i_global == n - 1:
                continue

            for j in range(1, n - 1):

                old = T_old[i_loc, j]
                avg = 0.25 * (
                    T_old[i_loc + 1, j] + T_old[i_loc - 1, j] +
                    T_old[i_loc, j + 1] + T_old[i_loc, j - 1]
                )

                new = (1.0 - omega) * old + omega * avg
                T_new[i_loc, j] = new

                diff = abs(new - old)
                if diff > max_diff_local:
                    max_diff_local = diff

        max_diff_global = comm.allreduce(max_diff_local, op=MPI.MAX)
        if max_diff_global < eps or iters >= max_iters:
            break

        T_old, T_new = T_new, T_old

    local_interior = T_old[1:local_n + 1, :].copy()

    sendcounts = comm.gather(local_n * n, root=0)

    if rank == 0:

        T_global = np.zeros((n, n), dtype=np.float64)

        displs = [0]
        for p in range(1, size):
            displs.append(displs[-1] + sendcounts[p - 1])

        recvbuf = np.empty(sum(sendcounts), dtype=np.float64)
    else:
        T_global = None
        recvbuf = None
        displs = None

    sendbuf = local_interior.ravel()

    comm.Gatherv(
        sendbuf,
        (recvbuf, sendcounts, displs, MPI.DOUBLE),
        root=0
    )

    if rank == 0:
        offset = 0
        for p in range(size):
            p_start, p_n = decompose_rows(n, size, p)
            block = recvbuf[offset: offset + p_n * n]
            T_global[p_start:p_start + p_n, :] = block.reshape((p_n, n))
            offset += p_n * n

        return T_global, iters
    else:
        return None, iters


if __name__ == "__main__":

    ns = [10, 40, 80, 160]
    eps = 1e-12
    omega = 0.96

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(" MPI (pamięć lokalna) ")
        print(f"eps={eps}, omega={omega}, procesy={size}")
        print()


    for n in ns:

        comm.Barrier()
        if rank == 0:
            t0 = MPI.Wtime()

        T_global, iters = mpi_weighted_jacobi(n, eps=eps, omega=omega)

        comm.Barrier()
        if rank == 0:
            t1 = MPI.Wtime()
            elapsed = t1 - t0

            ana = AnalyticGrid(n=n, n_terms=200)
            ana.compute()

            diff = np.abs(T_global - ana.T)
            max_diff = diff.max()

            print(f"n: {n:6d}")
            print(f"czas: {elapsed}")
            print(f"iteracje: {iters}")
            print(f"błąd max: {max_diff}")
            print()
