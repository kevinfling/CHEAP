#!/usr/bin/env python3
"""Exploratory CHEAP DCT projection and PCG experiment (NumPy/SciPy).
Run: python3 experiments/dct_preconditioner.py
No performance or novelty claim; reports accuracy and iteration counts.
"""
import numpy as np
from scipy.fft import dct
from scipy.linalg import toeplitz
from scipy.sparse.linalg import LinearOperator, cg

def fbm_cov(n, h):
    t = np.arange(1, n + 1, dtype=float)
    return 0.5 * (t[:, None] ** (2*h) + t[None, :] ** (2*h)
                  - np.abs(t[:, None] - t[None, :]) ** (2*h))

def evaluate(n, label, a):
    q = dct(np.eye(n), type=2, norm="ortho", axis=0)
    w = np.diag(q @ a @ q.T).copy()
    approx = q.T @ (w[:, None] * q)
    err = np.linalg.norm(a - approx, "fro") / np.linalg.norm(a, "fro")
    rho = 0.1 * np.linalg.norm(a, 2)
    rhs = np.sin(np.arange(n) * 0.17) + 0.2
    reference = np.linalg.solve(a + rho * np.eye(n), rhs)
    direct = q.T @ ((q @ rhs) / (w + rho))
    solve_err = np.linalg.norm(reference - direct) / np.linalg.norm(reference)
    def matvec(v):
        return a @ v + rho * v
    def precond(v):
        return q.T @ ((q @ v) / (w + rho))
    counts = []
    for pre in (None, LinearOperator((n, n), matvec=precond)):
        iterations = [0]
        def callback(_):
            iterations[0] += 1
        x, info = cg(LinearOperator((n, n), matvec=matvec), rhs,
                     M=pre, rtol=1e-9, atol=0, callback=callback)
        residual = np.linalg.norm(matvec(x) - rhs) / np.linalg.norm(rhs)
        counts.append((iterations[0], info, residual))
    print(f"{label:12} N={n:4d} op_err={err:.6f} solve_err={solve_err:.6f} "
          f"CG={counts[0][0]} PCG={counts[1][0]} "
          f"PCG_residual={counts[1][2]:.2e} status={counts[1][1]}")

if __name__ == "__main__":
    for n in (64, 128, 256):
        evaluate(n, "AR1 rho=.9", toeplitz(0.9 ** np.arange(n)))
        evaluate(n, "fBm H=.7", fbm_cov(n, 0.7))
