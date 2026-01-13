from dataclasses import dataclass
import time
from multiprocessing import Barrier, Value, Process
from multiprocessing.shared_memory import SharedMemory

import numpy as np


@dataclass
class SimulationConfig:
    n: int
    omega: float
    epsilon: float
    max_iter: int
    shm_name: str
    diff_shm_name: str


class WorkerTask:
    def __init__(self, config: SimulationConfig, start_row: int, end_row: int,
                 proc_id: int, barrier: Barrier, stop_flag: Value, shared_iters):
        self.cfg = config
        self.start_row = start_row
        self.end_row = end_row
        self.pid = proc_id
        self.barrier = barrier
        self.stop_flag = stop_flag
        self.shared_iters = shared_iters
        self._shm = None
        self._diff_shm = None
        self.T = None
        self.diffs = None

    def run(self):
        try:
            self._connect_memory()
            self._solve_loop()
        finally:
            self._cleanup()

    def _connect_memory(self):
        self._shm = SharedMemory(name=self.cfg.shm_name)
        self.T = np.ndarray((self.cfg.n, self.cfg.n), dtype=np.float64, buffer=self._shm.buf)
        self._diff_shm = SharedMemory(name=self.cfg.diff_shm_name)
        num_parties = self.barrier.parties
        self.diffs = np.ndarray((num_parties,), dtype=np.float64, buffer=self._diff_shm.buf)

    def _solve_loop(self):
        for iteration in range(1, self.cfg.max_iter + 1):
            if self.stop_flag.value:
                break
            diff_red = self._compute_sweep(parity=0)
            self.barrier.wait()
            diff_black = self._compute_sweep(parity=1)
            self.barrier.wait()
            self._check_convergence(max(diff_red, diff_black), iteration)
            self.barrier.wait()

    def _compute_sweep(self, parity: int) -> float:
        max_diff = 0.0
        cols = self.cfg.n
        omega = self.cfg.omega
        for r in range(self.start_row, self.end_row):
            for c in range(1, cols - 1):
                if (r + c) % 2 == parity:
                    old = self.T[r, c]
                    avg = 0.25 * (self.T[r + 1, c] + self.T[r - 1, c] + self.T[r, c + 1] + self.T[r, c - 1])
                    new_val = (1.0 - omega) * old + omega * avg
                    self.T[r, c] = new_val
                    diff = abs(new_val - old)
                    if diff > max_diff:
                        max_diff = diff
        return max_diff

    def _check_convergence(self, local_max: float, iteration: int):
        self.diffs[self.pid] = local_max
        self.barrier.wait()
        if self.pid == 0:
            global_max = np.max(self.diffs)
            self.shared_iters.value = iteration
            if global_max < self.cfg.epsilon:
                self.stop_flag.value = 1

    def _cleanup(self):
        if self._shm: self._shm.close()
        if self._diff_shm: self._diff_shm.close()


class ParallelSolver:
    def __init__(self, n: int, initial_grid: np.ndarray = None):
        self.n = n
        self.T_init = initial_grid if initial_grid is not None else np.zeros((n, n))
        self.shm = None
        self.diff_shm = None

    def solve(self, n_procs=4, omega=0.96, epsilon=1e-6, max_iter=1000):
        self._allocate_shared_memory(n_procs)
        barrier = Barrier(n_procs)
        stop_flag = Value('i', 0)
        shared_iters = Value('i', 0)
        config = SimulationConfig(self.n, omega, epsilon, max_iter, self.shm.name, self.diff_shm.name)
        processes = self._spawn_workers(n_procs, config, barrier, stop_flag, shared_iters)
        start_time = time.time()
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        duration = time.time() - start_time
        result = np.ndarray((self.n, self.n), dtype=np.float64, buffer=self.shm.buf).copy()
        total_iters = shared_iters.value
        self._cleanup()
        return result, duration, total_iters

    def _allocate_shared_memory(self, n_procs):
        self.shm = SharedMemory(create=True, size=self.T_init.nbytes)
        grid_view = np.ndarray(self.T_init.shape, dtype=np.float64, buffer=self.shm.buf)
        np.copyto(grid_view, self.T_init)
        dummy_diff = np.zeros(n_procs, dtype=np.float64)
        self.diff_shm = SharedMemory(create=True, size=dummy_diff.nbytes)

    def _spawn_workers(self, n_procs, config, barrier, stop_flag, shared_iters):
        processes = []
        rows_per_proc = (self.n - 2) // n_procs
        current_row = 1
        for i in range(n_procs):
            is_last = (i == n_procs - 1)
            end_row = (self.n - 1) if is_last else (current_row + rows_per_proc)
            p = Process(
                target=self._run_worker_wrapper,
                args=(config, current_row, end_row, i, barrier, stop_flag, shared_iters)
            )
            processes.append(p)
            current_row = end_row
        return processes

    @staticmethod
    def _run_worker_wrapper(config, start, end, pid, barrier, stop_flag, shared_iters):
        worker = WorkerTask(config, start, end, pid, barrier, stop_flag, shared_iters)
        worker.run()

    def _cleanup(self):
        if self.shm:
            self.shm.close()
            self.shm.unlink()
        if self.diff_shm:
            self.diff_shm.close()
            self.diff_shm.unlink()