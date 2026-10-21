# Practitioner validation protocol: CHEAP fBm solvers

This is an engineering acceptance gate, not a novelty claim.

## Build and reproducibility
Native source artifact `cheap_fbm_fftw.c` is currently distributed separately from the repository. Before a release, commit it and build with an official `fftw3.h` and the FFTW3 development package, e.g.:

```sh
cc -std=c99 -O3 -Wall -Wextra -Wpedantic -D_POSIX_C_SOURCE=200809L \
  experiments/cheap_fbm_fftw.c -lfftw3 -lm -o cheap_fbm_fftw
```

Do not use an unverified minimal FFTW declaration header for a release build. The native benchmark currently includes CG and DCT-PCG, but **not** the fGn-Chan competitor. The Python implementation of the latter is in `experiments/fbm_increment_domain_comparison.py`.

## Numerical acceptance gates
- Verify the covariance operator against dense `A[i,j]=0.5*(t_i^(2H)+t_j^(2H)-abs(t_i-t_j)^(2H))`.
- Verify DCT weights against `diag(Q A Q^T)` with orthonormal DCT-II Q.
- Compare solutions against dense Cholesky at small N; check original-system relative residual at all N.
- Run H in {.01,.05,.1,.3,.5,.7,.9,.95,.99}, rho/lambda_max in {1e-8,1e-6,1e-4,1e-2}, N from 32 through 65536 where feasible; report failure as data.
- Use multiple independent random RHS plus adversarial RHS aligned with high/low DCT modes. Validate both forward and backward error on small N.
- Check DCT weights are positive after regularization; never silently floor negative eigenvalues and call the result the optimal projection.
- Compare CG, DCT-PCG, fGn-Chan, and a Woodbury/boundary-corrected preconditioner if available.
- Distinguish transformed residual from original residual for fGn; transformed tolerance must be selected based on the original accuracy goal.
- Use ASan/UBSan, leak detection, compiler variations, and at least one independent machine.

## Performance acceptance gates
- Separate common setup (covariance kernel, FFT plans, lambda estimate) from method-specific setup (DCT weights, DCT plans; fGn circulant and plans).
- Report cold-start setup, warm setup, one RHS, 10 RHS, 100 RHS, memory high-water, FFT counts, iterations, and true residual.
- Reuse FFTW plans; do not replan per solve. Record FFTW version, FFTW planning flags, CPU, clock policy, compiler, and flags.
- Use paired, randomized method ordering and at least 30 independent timing samples; report median, IQR and bootstrap 95% CI. Include both solve-only and full end-to-end timing.
- A practical selection policy should minimize predicted total cost: `setup(method)+RHS_count*solve(method)`, with accuracy/failure gating first.

## Theorem / proof gate
For SPD A and rho>0, B=Q^T diag(diag(Q A Q^T)) Q is SPD, M=B+rho I SPD. If delta=||M^(-1/2)(A-B)M^(-1/2)||_2<1, eigenvalues of the preconditioned matrix lie in [1-delta,1+delta]. This standard bound alone says nothing uniform in N or H. Establish stronger bounds or explicitly state the results are empirical. In particular the Wiener-class Toeplitz results of Kailath-Olshevsky do not automatically cover fBm's nonstationary low-rank correction or all long-memory regimes.

## Practical decision
Treat the DCT route as a useful tool with measured benefits in several tested fBm regimes, not a universal replacement. Near H->1 at weak regularization the increment-domain method can require fewer iterations. Preserve fallback to CG or an alternative preconditioner whenever construction or convergence diagnostics fail.
