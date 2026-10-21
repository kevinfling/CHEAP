# FFTW C benchmark: first exploratory run

A standalone C99 FFTW benchmark was compiled and run locally on 2026-10-21. The source is available as a conversation artifact pending repository integration. The benchmark uses FFTW real-to-complex convolution for exact symmetric Toeplitz matrix-vector products, FFTW DCT-II/III for DCT preconditioning, and an FFT circulant preconditioner.

For AR(1) covariance c[k]=0.9^k, N=256, rho=0.001 times a Gershgorin spectral-norm upper bound, relative tolerance 1e-9:

| Solver | Iterations | One measured solve (ms) | True relative residual |
| --- | ---: | ---: | ---: |
| CG | 88 | 1.009 | 5.40e-10 |
| DCT-PCG | 7 | 0.041 | 2.52e-10 |
| Strang circulant-PCG | 4 | 0.018 | 1.92e-13 |

Caveats: single measurements; times fluctuate at microsecond scale. FFTW MEASURE planning and dense construction of the Frobenius-optimal DCT diagonal are counted in setup, not solves. The prototype's DCT diagonal construction is cubic in N and therefore **not competitive in setup cost**. Setup measured roughly 170–400 ms for N=256. The comparison demonstrates iteration reduction, NOT an end-to-end speedup. The circulant preconditioner is faster on this stationary Toeplitz problem. Next: efficient DCT diagonal construction, proper setup amortization, fBm/nonstationary baselines, multiple repetitions and uncertainty estimates. Do not cite these numbers as a validated CHEAP library benchmark.
