#!/usr/bin/env python3
"""Exploratory sweep: DCT vs Strang circulant preconditioning.
Requires numpy and scipy. Run with OPENBLAS_NUM_THREADS=1.
Circulant on nonstationary fBm is NOT an established comparator:
its spectrum can be negative and is floored here solely to make an SPD
preconditioner. Report its minimum eigenvalue to expose this issue.
"""
import numpy as np
from scipy.fft import dct, idct, fft, ifft
from scipy.linalg import toeplitz
from scipy.sparse.linalg import LinearOperator, cg

def trial(n, kind, frac):
    t = np.arange(n, dtype=float)
    if kind == "AR1":
        a = toeplitz(0.9 ** t)
    else:
        x = t + 1
        a = .5 * (x[:, None] ** 1.4 + x[None, :] ** 1.4
                  - np.abs(x[:, None] - x[None, :]) ** 1.4)
    rho = frac * np.linalg.eigvalsh(a)[-1]
    s = dct(dct(a, type=2, norm="ortho", axis=0),
            type=2, norm="ortho", axis=1)
    w = np.diag(s).copy()
    def pd(v):
        return idct(dct(v, type=2, norm="ortho") / (w + rho),
                    type=2, norm="ortho")
    col = np.array([a[0, min(j, n-j)] for j in range(n)])
    eig = fft(col).real
    floor = 1e-10 * max(1., np.max(np.abs(eig + rho)))
    denom = np.maximum(eig + rho, floor)
    def pc(v):
        return ifft(fft(v) / denom).real
    def mv(v):
        return a @ v + rho * v
    b = np.sin(.17 * t) + .2
    op = LinearOperator((n, n), matvec=mv)
    results = []
    for name, fn in [("CG", None), ("DCT", pd), ("circulant", pc)]:
        counter = [0]
        def callback(_):
            counter[0] += 1
        m = None if fn is None else LinearOperator((n, n), matvec=fn)
        sol, info = cg(op, b, M=m, rtol=1e-9, atol=0,
                       callback=callback, maxiter=3000)
        residual = np.linalg.norm(mv(sol)-b)/np.linalg.norm(b)
        results.append((name, counter[0], residual, info))
    return eig.min(), results

if __name__ == "__main__":
    for kind in ("AR1", "fBm"):
        for n in (64, 128, 256):
            for frac in (.001, .01, .1):
                minimum, result = trial(n, kind, frac)
                print(kind, n, frac, "min_circulant_eigenvalue", minimum,
                      result)
