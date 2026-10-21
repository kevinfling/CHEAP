#!/usr/bin/env python3
"""Finite-grid fBm covariance: reproducible DCT-PCG sweep.
Dependencies: numpy, scipy. Run: OPENBLAS_NUM_THREADS=1 python3 experiments/fbm_pcg_sweep.py
All iteration counts are for true covariance operator, not surrogate residuals.
The first-column circulant is a deliberately naive baseline, NOT a
literature-grade fBm preconditioner. Eigenvalue flooring is disclosed.
"""
import numpy as np
from scipy.fft import dct, idct, fft, ifft
from scipy.sparse.linalg import cg, LinearOperator

def trial(n, h, reg):
    t = np.arange(1, n + 1, dtype=float) / n
    a = 0.5 * (t[:, None] ** (2*h) + t[None, :] ** (2*h)
               - np.abs(t[:, None] - t[None, :]) ** (2*h))
    rho = reg * np.linalg.eigvalsh(a)[-1]
    diag = np.diag(a) + rho
    transformed = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")
    weights = np.diag(transformed).copy() + rho
    c = np.array([a[0, min(j, n-j)] for j in range(n)])
    eig = fft(c).real + rho
    floor = max(1e-12, 1e-9*np.max(np.abs(eig)))
    clipped = np.maximum(eig, floor)
    b = np.sin(np.arange(n)*0.17) + 0.2
    def mv(v):
        return a @ v + rho*v
    op = LinearOperator((n,n), matvec=mv)
    methods = [
        ("CG", None),
        ("Jacobi", lambda v: v/diag),
        ("DCT", lambda v: idct(dct(v, norm="ortho")/weights, norm="ortho")),
        ("naive_circulant_clipped", lambda v: ifft(fft(v)/clipped).real),
    ]
    output = []
    for name, fn in methods:
        it = [0]
        def callback(_):
            it[0] += 1
        pre = None if fn is None else LinearOperator((n,n), matvec=fn)
        x, info = cg(op, b, M=pre, rtol=1e-9, atol=0,
                     callback=callback, maxiter=3000)
        residual = np.linalg.norm(mv(x)-b)/np.linalg.norm(b)
        output.append((name, it[0], info, residual))
        if info or residual > 1.1e-9:
            raise RuntimeError(f"failed: {n=} {h=} {reg=} {name=} {info=} {residual=}")
    return eig.min(), output

if __name__ == "__main__":
    for n in (64, 128, 256):
        for h in (0.3, 0.7):
            for reg in (0.0001, 0.001, 0.01):
                minimum, result = trial(n, h, reg)
                print(f"N={n} H={h} rho_fraction={reg} "
                      f"min_naive_circulant_eigenvalue={minimum:.8g} "
                      f"results={result}", flush=True)
