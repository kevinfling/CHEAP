# Matched-accuracy evidence protocol

> **Operation and accuracy disclosure:** A DCT-diagonal application is a **surrogate apply**, not a solve of a generic Toeplitz system. Exact Toeplitz FFT matvec evaluates the original finite Toeplitz operator but is not itself a solve. Residual-controlled PCG solves the original SPD system only when its reported fresh true residual meets the requested tolerance. PSD-validated circulant draws are covariance-faithful only for an accepted embedding (within the documented eigenvalue tolerance). Positive DCT weights and a small surrogate residual do **not** imply a small residual on the original operator. See [benchmark protocol](BENCH.md#matched-accuracy-evidence-protocol).

## Evidence status

The historical measurements below are retained as **legacy, non-matched-accuracy observations**. They do not establish a speedup for solving the original Toeplitz system. Do not compare a single DCT surrogate apply to a dense solve as if both solved the same mathematical problem.

The opt-in `cheap_bench_evidence` harness measures setup, first execution, and 40 warm steady-state timings for an original Toeplitz FFT matvec, unpreconditioned CG, DCT-preconditioned PCG on that same original operator, and a PSD circulant draw. Its `pcg` mode compares the two solvers at the same true-residual target, with independent AR(1) residual verification after every solve. This remains a preliminary comparison of those two solver paths: structure-specific direct baselines and other problem/RHS families are not measured. The full mode also checks matvec against a dense oracle at N<=1024 and checks two covariance entries using 12,000 seeded Gaussian draws; full empirical covariance recovery remains missing. Timed sampler calls repeat fixed input and exclude RNG. Peak memory is process high-water RSS, not incremental per-operation memory.

The [2026-10-20 Jetson size sweep](results/jetson-2026-10-20/README.md) records both backends at N=17, 64, 257, 512, 1024 and 4096, including raw CSVs, accuracy checks, platform/build provenance and separate sanitizer validation. All twelve runs succeeded. FFTW3 had lower medians for all three operations at every size in that session; the results do not establish a dense/FFT crossover or a preconditioning speedup.

### Stronger-correlation CG/DCT-PCG experiment

The [2026-10-20 paired solver sweep](results/jetson-pcg-2026-10-20/README.md)
repeats the six sizes for rho=0.3, 0.9 and 0.99, using both FFT backends.
All 36 processes and 2,952 individually logged solves passed the common stopping
target and an independent original-system residual check. At N=4096, rho=0.99,
FFTW3's median solve time fell from 193.027 ms (854 iterations) to 2.636 ms
(6 iterations); Notorious's fell from 217.380 ms (853 iterations) to 2.941 ms
(6 iterations). Notorious DCT-PCG was slower at N=17 and N=257 despite fewer
iterations. The report includes those cases and separates common setup,
preconditioner setup, first calls and warm solves.

The numerical library is unchanged. The benchmark uses the existing
`cheap_init_from_toeplitz_projection` and PCG preconditioner adapter; its local
planner wrapper selects `FFTW_ESTIMATE` for DCT plans as well as FFT plans.

```sh
# Existing one-argument invocation remains supported; defaults: 257 0.3 all.
# all: matvec, CG, DCT-PCG and sampler; pcg: just the two original-system solvers.
./build-evidence-fftw/cheap_bench_evidence 257 0.99 pcg /tmp/solve-samples.csv
python3 tests/test_bench_evidence.py \
  build-evidence-fftw/cheap_bench_evidence \
  build-evidence-notorious/cheap_bench_evidence
python3 bench/run_pcg_sweep.py --output results/pcg-repeat \
  --fftw build-evidence-fftw/cheap_bench_evidence \
  --notorious build-evidence-notorious/cheap_bench_evidence
```

CLI: `[n [rho [all|pcg [solve_samples.csv]]]]`, with `2 <= n <= 65536` and
finite `0 <= rho < 1`. Both solvers start from zero, use
`atol=1e-10`, `rtol=1e-8`, and check the original Toeplitz equation.
The preconditioner is the inverse Frobenius-optimal DCT-II diagonal, not the
legacy DCT of the first column. The optional solve log contains every first and
steady call, including failures. Argument/setup/output errors and rejected
embeddings also produce a nonzero process exit; the runner retains failures.
Strong-rho sampler embeddings can be rejected at small N, so use `pcg` mode for
this experiment.

The original 18 summary CSV columns are retained; additional columns expose
rho, absolute tolerance, common and preconditioner setup, first relative and
independently checked relative residuals, maximum residuals over all 41 calls,
last-call operator/preconditioner counts, residual validity, first solver status,
failed-solve count, iteration range, padded FFT length and planner.
`setup_ms = common_setup_ms + preconditioner_setup_ms` for the solver rows.
The legacy `true_residual` field is the first solve's absolute norm; new relative
fields are explicitly named. The legacy `backend` field labels the operation
path; `configured_backend` identifies FFTW3 versus Notorious.
`max_relative_residual` and `max_independent_relative_residual` cover all calls;
last-call counts accompany the legacy last-call `iterations` field.
The sampler's two-entry covariance threshold (0.12) and matvec oracle threshold
(`1e-10`) differ from the CSV's generic solver relative tolerance (`1e-8`).

### Reproduction commands (Linux, C99, release)

```sh
cmake -S . -B build-evidence-fftw -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_FLAGS=-march=native -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -DCHEAP_FFT_BACKEND=FFTW3 -DCHEAP_BUILD_EVIDENCE_BENCH=ON
cmake --build build-evidence-fftw --target cheap_bench_evidence
./build-evidence-fftw/cheap_bench_evidence 257 > evidence-fftw-257.csv
cmake -S . -B build-evidence-notorious -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_FLAGS=-march=native -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -DCHEAP_FFT_BACKEND=NOTORIOUS -DCHEAP_BUILD_EVIDENCE_BENCH=ON
cmake --build build-evidence-notorious --target cheap_bench_evidence
./build-evidence-notorious/cheap_bench_evidence 257 > evidence-notorious-257.csv
```

Notorious-FFT is pinned in CMake at `681ae621f51731f84cded6872dcb1f1082105168`. FFTW planning uses `FFTW_ESTIMATE` in the relevant Toeplitz/sampler contexts. Record the local FFTW revision, compiler and flags, CPU, OS, precision, sanitizers, and commit SHA before publishing comparative results. Run correctness under Debug/ASan/UBSan separately from optimized timings. The CSV includes median, p95 and IQR over 40 timed calls after a first execution; it does not contain confidence intervals or cross-method matched-accuracy speedups.

## Publication gate

A final comparative table must include identical discretization and precision, verified true residuals for **every** solve, covariance discrepancy for sampling, structure-aware baselines, setup and first-call timing, median/p95/IQR, and high-water memory. Report invalid/rejected embeddings and nonconverged solves as failures, never omit them silently. Include at least one rectangular 2D and one noncubic 3D separable DCT case, clearly labeled surrogate-only.

---

## Historical benchmarks (not matched-accuracy evidence)



## 1D

### ARM64 (ARMv8 rev 1 @ ~1.5 GHz, GCC 11.4.0, `-O3 -march=native`)

| Algorithm | N=1,024 | N=8,192 | N=65,536 | Scaling |
|-----------|---------|---------|----------|---------|
| KRR solve | 0.022 ms | 0.210 ms | 2.94 ms | O(N log N) |
| Sinkhorn (50 iters) | 0.196 ms | 1.67 ms | 18.40 ms | O(N log N) |
| Reparameterization | 0.022 ms | 0.212 ms | 2.84 ms | O(N log N) |
| Toeplitz matvec | 0.022 ms | 0.210 ms | 2.94 ms | O(N log N) |
| RFF map (D=256) | 0.016 ms | — | — | O(D) |

### Statistical Benchmarks (30 trials, FFTW_PATIENT)

| Algorithm | N | Mean (ms) | CV% | Status |
|-----------|---|-----------|-----|--------|
| krr_solve | 65,536 | 5.588 | 0.09% | STABLE |
| reparam | 65,536 | 5.630 | 0.11% | STABLE |
| sinkhorn_50 | 65,536 | 33.11 | 0.16% | STABLE |
| toeplitz_matvec | 65,536 | 5.745 | 0.20% | STABLE |

All CV < 0.5%, 95% CI within 0.2% of mean.

## 2D

All times are median wall time (1000 iterations, FFTW_PATIENT, `-O3 -march=native`, NEON enabled).

| N_total | Algorithm | wall time | in-place wall time | in-place savings |
|---------|-----------|-----------|-------------------|------------------|
| 4,096 (64×64) | apply_krr_2d | 0.099 ms | 0.094 ms | 5% |
| 16,384 (128×128) | apply_krr_2d | 0.527 ms | 0.505 ms | 4% |
| 65,536 (256×256) | apply_krr_2d | 2.90 ms | 2.81 ms | 3% |
| 262,144 (512×512) | apply_krr_2d | 12.39 ms | 11.97 ms | 3% |

## 3D

| N_total | Algorithm | wall time | in-place wall time | in-place savings |
|---------|-----------|-----------|-------------------|------------------|
| 4,096 (16³) | apply_krr_3d | 0.113 ms | 0.109 ms | 4% |
| 32,768 (32³) | apply_krr_3d | 1.30 ms | 1.25 ms | 4% |
| 262,144 (64³) | apply_krr_3d | 12.68 ms | 12.26 ms | 3% |

## Weight Constructors (v0.3.0)

Weight constructors are pure O(N) spectral math — no FFTW plans, no ctx required. The DCT in `cheap_apply` at O(N log N) dominates total cost; weight construction is never the bottleneck.

### ARM64 (ARMv8 rev 1 @ ~1.5 GHz, GCC 11.4.0, `-O3 -march=native`, NEON enabled)

All times are median wall time in ms (1000 iterations). The `cntvct_el0` timer runs at ~31.25 MHz on this board; "timer ticks/el" figures are not CPU cycles (multiply by ~48 to convert at 1.5 GHz).

| Constructor | N = 1,024 | N = 8,192 | N = 65,536 | SIMD | Bottleneck |
|---|---|---|---|---|---|
| `matern_ev` (κ=1, ν=2.5) | 0.028 ms | 0.226 ms | 1.810 ms | scalar | `pow` |
| `heat_propagator_ev` (t=0.1) | 0.009 ms | 0.068 ms | 0.546 ms | scalar | `exp` |
| `biharmonic_ev` | 0.004 ms | 0.035 ms | 0.278 ms | NEON 2×f64 | div |
| `poisson_ev` | 0.004 ms | 0.034 ms | 0.274 ms | NEON 2×f64 | div |
| `higher_order_tikhonov_deconv_ev` (p=2, lap=NULL) | 0.024 ms | 0.189 ms | 1.508 ms | scalar | `pow`+`sin` |

| Constructor | N_2D = 16,384 (128×128) | SIMD | Bottleneck |
|---|---|---|---|
| `matern_2d` (κ=1, ν=2.5) | 0.975 ms | scalar | `pow` |
| `biharmonic_2d` | 0.352 ms | NEON (pass 2) | div |
| `poisson_2d` | 0.353 ms | NEON (pass 2) | div |

**NEON speedup (`biharmonic_ev` vs `heat_ev` per element):** 0.278 ms vs 0.546 ms at N=65,536 → ~2.0× faster despite similar element count, confirming NEON vectorization is effective.

**Regression baseline:** if `biharmonic_ev` exceeds 0.40 ms at N=65,536 on this board, check that `-march=native` enables NEON and that `CHEAP_SIMD_DISABLE` is not set.
