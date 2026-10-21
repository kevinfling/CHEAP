#!/usr/bin/env python3
"""Matrix-free finite-grid fBm PCG, with O(N log N) DCT projection setup.
Requires numpy/scipy. Run: OPENBLAS_NUM_THREADS=1 python3 experiments/fbm_matrixfree.py
Exploratory single-run timing, NOT controlled end-to-end performance data.
"""
import numpy as np
import time
from scipy.fft import fft, ifft, dct, idct
from scipy.sparse.linalg import cg, LinearOperator

def dct_diag_toeplitz(c):
    """diag(Q T(c) Q.T), Q orthonormal DCT-II; two length-2N FFTs."""
    n = len(c)
    a = np.zeros(2*n)
    b = np.zeros(2*n)
    a[1:n] = (n-np.arange(1,n))*c[1:]
    b[1:n] = c[1:]
    fa, fb = fft(a), fft(b)
    w = np.empty(n)
    w[0] = c[0] + 2*a.sum()/n
    k = np.arange(1,n)
    w[1:] = c[0] + (2/n)*(fa.real[k] + fb.imag[k]/np.sin(np.pi*k/n))
    return w

def run(n,h,reg):
    t = np.arange(1,n+1)/n
    u = .5*t**(2*h)
    c = -.5*(np.arange(n)/n)**(2*h)
    # fBm covariance A = Toeplitz(c) + u 1^T + 1 u^T.
    z = np.zeros(2*n)
    z[:n] = c
    z[n+1:] = c[1:][::-1]
    fz = fft(z)
    def mv0(v):
        a = np.zeros(2*n)
        a[:n] = v
        return ifft(fft(a)*fz).real[:n] + u*v.sum() + np.dot(u,v)
    w = dct_diag_toeplitz(c)
    # Rank-two correction contributes only to the DC DCT diagonal.
    w[0] += 2*u.sum()
    v = np.ones(n)/np.sqrt(n)
    for _ in range(50):
        z = mv0(v)
        v = z/np.linalg.norm(z)
    lam = np.dot(v,mv0(v))
    rho = reg*lam
    rhs = np.sin(np.arange(n)*.17)+.2
    def mv(v): return mv0(v)+rho*v
    op = LinearOperator((n,n),matvec=mv)
    methods = [
        ("CG",None),
        ("Jacobi",lambda v:v/(t**(2*h)+rho)),
        ("DCT",lambda v:idct(dct(v,norm="ortho")/(w+rho),norm="ortho")),
    ]
    results = []
    for name,fn in methods:
        it = [0]
        def cb(_): it[0] += 1
        M = None if fn is None else LinearOperator((n,n),matvec=fn)
        start = time.perf_counter()
        x,info = cg(op,rhs,M=M,rtol=1e-9,atol=0,callback=cb,maxiter=5000)
        elapsed = time.perf_counter()-start
        residual = np.linalg.norm(mv(x)-rhs)/np.linalg.norm(rhs)
        if info != 0 or residual > 1.1e-9:
            raise RuntimeError((n,h,reg,name,info,residual))
        results.append((name,it[0],elapsed,residual,info))
    return w.min(),results

def verify_projection():
    for n in (8,16,64,256):
        for h in (.3,.7):
            t = np.arange(1,n+1)/n
            u = .5*t**(2*h)
            c = -.5*(np.arange(n)/n)**(2*h)
            a = u[:,None]+u[None,:]+c[np.abs(np.subtract.outer(np.arange(n),np.arange(n)))]
            q = dct(np.eye(n),axis=0,norm="ortho")
            w = dct_diag_toeplitz(c)
            w[0] += 2*u.sum()
            np.testing.assert_allclose(w,np.diag(q@a@q.T),rtol=1e-9,atol=1e-10)

if __name__ == "__main__":
    verify_projection()
    print("PASS: FFT projection agrees with dense DCT reference",flush=True)
    for n in (256,1024,4096):
        for h in (.3,.7):
            for reg in (.0001,.001):
                minimum,results = run(n,h,reg)
                print(f"N={n} H={h} reg={reg} min_weight={minimum:.8g} results={results}",flush=True)
