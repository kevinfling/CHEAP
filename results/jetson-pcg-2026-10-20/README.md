# Jetson CG versus DCT-PCG — 2026-10-20

Stronger correlation exposed a large iteration and solve-time benefit from the
existing DCT preconditioner on this Toeplitz family, with important size/backend
exceptions. All 36 processes completed successfully: 3 correlation parameters ×
6 sizes × 2 FFT backends. All 2,952 solves (first call plus 40 repetitions for each
method) passed both the solver's fresh original-system residual check and an
independent AR(1) residual check.

At N=4096, rho=0.99, FFTW3 reduced CG from 854 iterations and 193.027 ms to
6 iterations and 2.636 ms with DCT-PCG: a 73.2× steady-state median ratio.
Notorious reduced 853 iterations and 217.380 ms to 6 iterations and 2.941 ms
(73.9×). These are paired solve-time comparisons at the same stopping target,
for this matrix family and RHS on this Jetson, not a universal solver speedup.

No numerical library code was changed. The harness composes the existing
`cheap_toeplitz_operator_init`, `cheap_init_from_toeplitz_projection`,
`cheap_pcg_dct_init`, and `cheap_pcg_solve_opts` APIs. The only planning override
is local to the benchmark: its 1D r2r planner wrapper passes `FFTW_ESTIMATE`
in place of the production initializer's `FFTW_PATIENT` request. Library headers,
projection formulas, preconditioning and PCG recurrences remain unchanged.

## Matched problem and verification

- `A[i,j] = rho^abs(i-j)`; rho=0.3 control, 0.9 and 0.99 stronger correlations.
  No ridge term or other regularization is added.
- N=17, 64, 257, 512, 1024 and 4096; double precision, native AArch64.
- Identical `b[i] = sin(0.1*i) + 0.5`, zero initial guess for every solve,
  absolute tolerance `1e-10`, relative tolerance `1e-8`, at most `4*N+40`
  iterations. No periodic residual replacement; continue policy on a convergence
  check that requires a residual refresh, as in the initial sweep.
- CG uses the exact finite Toeplitz FFT operator with no preconditioner.
  DCT-PCG uses that same operator and the inverse of the Frobenius-optimal
  DCT-II diagonal `diag(Q A Q^T)` as its SPD preconditioner. No weights are
  floored to make this experiment pass.
- After **every** solve, an independent long-double geometric-sum recurrence
  computes `A*x` and checks `||b-A*x|| <= 1e-10 + 1e-8*||b||`. This oracle uses
  neither FFT nor DCT. For N<=1024 it is also cross-checked against an independent
  dense Toeplitz matvec. Verification and CSV logging occur outside solve timing.
- The largest independent relative residual across all 2,952 solves was
  `9.86788362e-9`, below the common target. Reaching different residuals below
  that target is allowed; neither method receives a looser tolerance.
- Sampler/covariance measurements were omitted in `pcg` mode; the recorded
  experiment compares two solvers, without introducing sampler embedding
  acceptance as a confounding condition at high rho and small N.

## Solve-time results

Every arrow means **unpreconditioned CG → DCT-PCG**. Times are median
milliseconds over 40 calls after the first call. Raw summary files retain
setup, first call, p95, IQR and memory; `*-samples.csv` records every call's
iterations, operator/preconditioner calls, residuals, solver status and acceptance.
The iteration counts were constant across first and steady calls within each row.

### rho=0.3

| N | FFTW3 iterations | FFTW3 median ms | Notorious iterations | Notorious median ms |
|---:|---:|---:|---:|---:|
| 17 | 13 → 6 | 0.013216 → 0.010048 | 13 → 6 | 0.018273 → 0.096931 |
| 64 | 14 → 5 | 0.037058 → 0.022881 | 14 → 5 | 0.047169 → 0.027873 |
| 257 | 13 → 4 | 0.284105 → 0.188822 | 13 → 4 | 0.357179 → 13.936323 |
| 512 | 13 → 4 | 0.344619 → 0.188533 | 13 → 4 | 0.417133 → 0.226295 |
| 1024 | 13 → 3 | 0.730423 → 0.333547 | 13 → 3 | 0.878507 → 0.399581 |
| 4096 | 13 → 3 | 3.560431 → 1.624899 | 13 → 3 | 4.012333 → 1.823577 |

### rho=0.9

| N | FFTW3 iterations | FFTW3 median ms | Notorious iterations | Notorious median ms |
|---:|---:|---:|---:|---:|
| 17 | 16 → 6 | 0.015777 → 0.010048 | 16 → 6 | 0.021825 → 0.094515 |
| 64 | 42 → 8 | 0.099459 → 0.033153 | 43 → 8 | 0.129044 → 0.040018 |
| 257 | 88 → 7 | 1.633123 → 0.292489 | 87 → 7 | 2.027199 → 24.379208 |
| 512 | 107 → 6 | 2.416507 → 0.253432 | 107 → 6 | 2.914746 → 0.303514 |
| 1024 | 120 → 6 | 5.726083 → 0.541600 | 120 → 6 | 6.857381 → 0.643524 |
| 4096 | 129 → 5 | 29.575434 → 2.283047 | 129 → 5 | 33.539462 → 2.568368 |

### rho=0.99

| N | FFTW3 iterations | FFTW3 median ms | Notorious iterations | Notorious median ms |
|---:|---:|---:|---:|---:|
| 17 | 19 → 5 | 0.018192 → 0.008769 | 19 → 5 | 0.025217 → 0.079267 |
| 64 | 48 → 6 | 0.112899 → 0.026369 | 49 → 6 | 0.146132 → 0.031937 |
| 257 | 146 → 8 | 2.687956 → 0.327674 | 143 → 8 | 3.293398 → 27.809907 |
| 512 | 248 → 8 | 5.577118 → 0.319866 | 247 → 8 | 6.626798 → 0.382924 |
| 1024 | 398 → 8 | 18.646309 → 0.682998 | 399 → 8 | 22.478156 → 0.809561 |
| 4096 | 854 → 6 | 193.026630 → 2.635842 | 853 → 6 | 217.379540 → 2.941307 |

At N=4096, FFTW3's median ratio grows from 2.19× at rho=0.3 to 12.95× at
rho=0.9 and 73.23× at rho=0.99. This supports the proposed hypothesis for
larger grids in this experiment: stronger correlation creates substantially more
room for this DCT preconditioner to reduce CG work.

The Notorious cases at N=17 and N=257 are slower with DCT-PCG for **all three**
rho values despite lower iteration counts. At N=257, rho=0.99, DCT-PCG takes
27.810 ms versus 3.293 ms for CG (8.44× slower). The preconditioner applies
length-N DCTs, whereas the Toeplitz matvec pads to a power of two. N=17 and
N=257 therefore exercise different DCT size classes from the other sizes.
The association with awkward DCT lengths is consistent with these timings;
no profiler measurement was made to attribute the slowdown to a specific backend
routine or to cache behavior.

## Setup and amortization

`common_setup_ms` is measured Toeplitz operator construction plus PCG workspace
allocation, shared by both paths. `preconditioner_setup_ms` separately includes
the DCT projection/context/plans and inverse weight construction. `setup_ms`
is common setup for CG and common plus preconditioner setup for DCT-PCG.
Do not charge the common setup twice when comparing the two choices.

At N=4096, rho=0.99, common setup was 3.305 ms (FFTW3) / 1.065 ms (Notorious),
and additional preconditioner setup was 1.408 / 1.541 ms. Adding measured setup
to first-call time gives 196.601 / 222.176 ms for CG and 7.556 / 5.706 ms for
DCT-PCG. Those sums are derived accounting estimates, not separate cold-process
end-to-end measurements: in each process, CG is run before DCT setup and PCG.

Setup matters at small sizes. At N=17, rho=0.99, FFTW3's lower DCT-PCG median
needs about 31 RHS solves to offset the measured additional setup and first-call
difference in a simple reuse model. At N=64, the estimate is 5 solves for FFTW3
and 1 for Notorious; at N=257 it is 1 for FFTW3, while Notorious's slower
steady solves do not amortize the preconditioner in this experiment.

[comparison.csv](comparison.csv) contains derived iteration counts, median ratios,
setup-plus-first-call sums and modeled reuse thresholds for every case. The model
compares `common + first_CG + (k-1)*median_CG` with
`common + pre_setup + first_PCG + (k-1)*median_PCG`. It repeats timings from the
same RHS; this does not measure a distribution of different RHSs or independently
validate the reuse threshold. Cases with no steady saving are marked `none`.

## Platform, execution and provenance

The same prepared Jetson Orin Nano Engineering Reference Developer Kit Super
was used: six Cortex-A78AE cores, L4T R36.4.4, Linux 5.15.148-tegra, 15 W mode 0,
CPU clocks fixed at 1.4976 GHz, GPU at 612 MHz and observed EMC at 2.133 GHz.
The runner only queried power/clocks and did not change settings.

Both optimized binaries used GCC 11.4.0, C99, `-O3 -DNDEBUG -march=native`,
no fast-math or sanitizers. The Notorious implementation translation unit received
the same optimization flags. FFTW was the system 3.3.8 shared library; Notorious
was pinned to `681ae621f51731f84cded6872dcb1f1082105168`. All plans used
`FFTW_ESTIMATE`. Build commands match the [initial report](../jetson-2026-10-20/README.md#reproduce).

Timing window: 2026-10-21 05:44:45–05:45:13 UTC
(2026-10-20 22:44:45–22:45:13 America/Vancouver). Cases ran sequentially,
rho ascending then N ascending, with FFTW3 immediately followed by Notorious
for each pair. Each process ran first CG + 40 CG calls, constructed the
preconditioner, then ran first DCT-PCG + 40 DCT-PCG calls. CPU affinity was not
pinned. The runner executed outside sandbox tracing, with no simultaneous build
or correctness-test workload.

Before/after snapshots for all 36 processes reported CPU clocks of 1.4976 GHz.
CPU temperature ranged from 46.468 to 47.375 °C; junction temperature ranged
from 47.937 to 48.562 °C. These are endpoint snapshots, not continuous telemetry.
Inactive CV sensors were recorded as unavailable. See [runs.json](runs.json),
[platform.json](platform.json), [power-before.json](power-before.json) and
[power-after.json](power-after.json).

Source provenance uses base commit `b4819e8`, the uncommitted harness change in
[harness.patch.gz](harness.patch.gz) (gzip-compressed unified diff), and [source-hashes.json](source-hashes.json),
including the unchanged numerical headers and backend adapter.
[builds.json](builds.json) records exact compile/link commands, binary hashes and
the Notorious commit. The report/documents were written after timing and are not
part of the measured source hash set.

These are 40 warm timings of a fixed RHS per method in one session, not 40
independent experimental sessions. Confidence intervals and randomized method
order were not measured. Setup and the first call remain distinct from steady
solve timing. `peak_bytes` is process high-water RSS, not incremental memory
for a method. The geometric Toeplitz family also admits structure-specific direct
methods; those are not timed here. A broader algorithm comparison still needs
those baselines, varied RHSs and problem families, and repeated sessions.

## Reproduce and validate

Build both backends with the explicit GCC/native settings from the initial
report, then run the new runner into a fresh directory:

```sh
python3 tests/test_bench_evidence.py \
  build-jetson-gcc-FFTW3/cheap_bench_evidence \
  build-jetson-gcc-NOTORIOUS/cheap_bench_evidence
python3 bench/run_pcg_sweep.py --output results/jetson-pcg-repeat
```

The runner records exit codes and retains partial output/failures. It validates
all summary statistics against the 82 per-process raw solve records and returns
nonzero if any process fails. It refuses to overwrite an existing result directory.
Power queries need permission to read the prepared device's state; unavailable
queries are explicitly recorded. An individual pair is:

```sh
./build-jetson-gcc-FFTW3/cheap_bench_evidence 4096 0.99 pcg /tmp/solve-samples.csv
```

Separate validation uses GCC `Asan` builds with
`-O1 -g -fsanitize=address,undefined`, assertions enabled, and
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1`,
outside sandbox tracing. Harness sanitizer cases cover prime and power-of-two
PCG paths, the full sampler path and expected rejection of a non-PSD embedding.
All eight harness sanitizer cases and all six existing projection/PCG contract
tests passed. See [FFTW3 harness](correctness-harness-FFTW3.txt),
[Notorious harness](correctness-harness-NOTORIOUS.txt),
[FFTW3 library](correctness-library-FFTW3.txt) and
[Notorious library](correctness-library-NOTORIOUS.txt) logs.
The standalone harness contract test also checks strict CLI parsing, file-open
failures, preservation of successful solve rows before a sampler failure,
per-call convergence/residuals and call counts, setup accounting and summary
statistics recomputed from the raw log.
