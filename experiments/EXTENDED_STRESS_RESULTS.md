# Extended fBm solver stress test (2026-10-21)

The comparison is matrix-free SciPy FFT, not the native FFTW implementation. Solver timing excludes FFTW planning, spectrum construction, and preconditioner setup. Each entry uses a randomized normal RHS and is checked against the **original** regularized covariance equation to true relative residual <=1.1e-9. The increment-domain fGn–Chan solver uses a tighter transformed-system stopping tolerance (1e-12) because transformed residuals do not directly control original residuals. Timing samples are exploratory.

## Hurst exponent sweep

N=1024, random seed=42, rho/lambda_max=1e-6:

| H | CG iterations | DCT-PCG | fGn–Chan-PCG |
|---:|---:|---:|---:|
| 0.05 | 59 | 9 | 16 |
| 0.10 | 92 | 10 | 14 |
| 0.30 | 289 | 10 | 10 |
| 0.70 | 328 | 12 | 10 |
| 0.90 | 146 | 19 | 11 |
| 0.95 | 105 | 23 | 13 |

For rho/lambda_max=1e-4, N=1024, seed=42:

| H | CG iterations | DCT-PCG | fGn–Chan-PCG |
|---:|---:|---:|---:|
| 0.05 | 54 | 9 | 16 |
| 0.10 | 80 | 10 | 14 |
| 0.30 | 109 | 10 | 11 |
| 0.70 | 49 | 10 | 13 |
| 0.90 | 29 | 15 | 17 |
| 0.95 | 24 | 16 | 21 |

## Larger N

N=16384, rho/lambda_max=1e-6, seed=7:

| H | CG iterations | DCT-PCG | fGn–Chan-PCG | CG solve ms | DCT solve ms | fGn–Chan solve ms |
|---:|---:|---:|---:|---:|---:|---:|
| .30 | 920 | 11 | 11 | 604.68 | 10.50 | 11.93 |
| .70 | 308 | 12 | 12 | 193.75 | 10.47 | 15.16 |
| .95 | 101 | 23 | 19 | 61.91 | 18.30 | 18.27 |

These results disprove any blanket statement that DCT-PCG is uniformly the best of the tested methods: fGn–Chan uses fewer iterations at H=.9/.95 with very weak regularization. Conversely, the DCT approach remains competitive at H=.3/.7 even at N=16384.

## Reproduction

Use `experiments/fbm_increment_domain_comparison.py` or `experiments/fbm_matrixfree.py` for the underlying implementations. Change the sweep to N=1024, H=(.05,.1,.3,.7,.9,.95), rho=(1e-6,1e-4), seed=42, and N=16384, H=(.3,.7,.95), rho=1e-6, seed=7.

## What is still missing

- Native FFTW implementation of the fGn–Chan competitor and full end-to-end cost decomposition
- Benchmarks against stronger literature methods, controlled CPU timing distributions, hardware disclosure
- Formal spectral clustering or N-independent preconditioning analysis
- Independent reproduction and detailed prior-art review

Do not represent this work as submission-ready or state of the art.
