# Independent reproducibility and limitations audit (2026-10-21)

## Scope
Independent local compilation and execution of the conversation artifact `cheap_fbm_fftw.c` (C99/FFTW3). This C file has **not** been committed to this repository. The FFTW3 shared library was installed, but its official development header was unavailable; compilation used a local minimal declaration header matching the FFTW3 functions and enum values used. This does not replace a clean build against official FFTW headers.

## Correctness
- Matrix-free fBm covariance product cross-checked against dense reference matrices for N=8,16,64, H=0.3,0.7: PASS.
- Fast FFT Toeplitz DCT projection plus rank-two DC correction cross-checked against dense orthonormal DCT-II projection for N=8,16,64, H=0.3,0.7: maximum absolute discrepancies on the order of 1e-14: PASS.
- Native benchmark compiled with `-std=c99 -O2 -Wall -Wextra -Wpedantic`: PASS.
- Native benchmark executed with `-fsanitize=address,undefined`, leak detection disabled: PASS on tested cases. This is **not** a full memory-safety or leak audit.
- Native solver verifies true-system relative residual <= 1.1e-9.

## Timing: 15 separate processes, milliseconds
Each row gives min / median / max across 15 process executions. CPU affinity, frequency scaling, and OS noise were not controlled. All solves use a fixed right-hand side; these are *not* 15 statistically independent problem instances. Timings exclude setup from individual solves.

| N | H | CG iterations | DCT iterations | CG min/median/max | DCT min/median/max | Setup min/median/max |
|---:|---:|---:|---:|---|---|---|
| 256 | .3 | 71 | 9 | .199/.200/.219 | .046/.046/.063 | 1.742/1.780/2.012 |
| 256 | .7 | 46 | 10 | .129/.130/.153 | .051/.051/.086 | .210/.216/.351 |
| 1024 | .3 | 100 | 10 | 1.110/1.140/1.207 | .203/.205/.218 | 2.242/2.302/2.590 |
| 1024 | .7 | 47 | 10 | .519/.535/.672 | .203/.206/.219 | .663/.702/1.038 |
| 4096 | .3 | 104 | 10 | 6.292/6.474/6.808 | 1.209/1.248/1.429 | 5.652/5.792/6.125 |
| 4096 | .7 | 47 | 10 | 2.834/2.920/3.442 | 1.217/1.240/1.357 | 3.487/3.535/3.809 |

**Interpretation warning:** The reported setup includes FFTW planning, covariance kernel construction, DCT weights, and 60 power iterations for estimating lambda_max, all of which are shared by both solvers. Therefore assigning *all* setup cost to DCT-PCG or inferring a fair break-even RHS count from this setup column is invalid. A controlled end-to-end comparison must separately account for common versus preconditioner-only costs.

## Mathematical facts and novelty
- fBm covariance has the exact representation `A=T(c)+u 1^T+1 u^T`, with `u_i=t_i^(2H)/2`, `c_d=-|d/N|^(2H)/2`.
- Only the DC **diagonal** entry of the rank-two term is nonzero in DCT coordinates; its off-diagonal DCT entries are generally nonzero. The DCT projection therefore remains an approximation.
- The optimal Frobenius DCT diagonal is the diagonal of `Q A Q^T`; this is standard orthogonal projection, not a new theorem.
- The FFT construction uses the Toeplitz lag structure and achieves O(N log N) arithmetic; its correctness was independently checked at small N.
- Near-constant observed PCG iterations are not proof of uniform spectral equivalence or mesh-independent convergence.

## Outstanding publication gates
1. Commit a fully buildable C benchmark and official FFTW3 build instructions; remove minimal-header dependency.
2. Test covariance matvec, projection and inverse preconditioner for random vectors, multiple H and N, including near endpoints H->0 and H->1.
3. Add a fair covariance-specific baseline: Woodbury-corrected Toeplitz/circulant preconditioner, fractional Gaussian noise/increment-domain methods, and applicable hierarchical or multigrid methods.
4. Separate setup components; use repeated solves with multiple RHS and compare total wall-clock cost including planning and preprocessing.
5. Test multiple compilers, optimization levels, ASan/UBSan with leak detection, and valgrind where available.
6. Extend N beyond 4096; vary regularization, H, grid conventions, and RHS; report failures and condition estimates.
7. Compare operator application count, FFT count, memory, residual histories, and wall-clock distributions with confidence intervals.
8. Derive a preconditioned spectral bound, or state precisely that the work is empirical.
9. Review prior art and establish whether the O(N log N) diagonal formula or fBm application has been published; do not claim novelty without this.
10. Publish hardware, FFTW version, flags, thread count, raw CSV, reproducible scripts, and a limitations section.

**Current status:** promising numerical engineering evidence, **not yet publication-ready**.
