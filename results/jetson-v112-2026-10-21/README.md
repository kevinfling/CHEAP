# Notorious-FFT v1.1.2 evaluation — Jetson, 2026-10-21

v1.1.2 removes the observed non-power-of-two DCT performance cliff while
preserving all measured output/residual checks. The standalone preconditioner
at N=257 now takes 0.018208 ms, compared with 3.451683 ms in the archived
v1.1.1 session. At rho=0.99, N=257, complete Notorious DCT-PCG now takes
0.393837 ms versus 3.299063 ms for unpreconditioned CG in the same session;
the prior DCT-PCG time was 27.809907 ms. The iteration counts remain 143 → 8.

Both recorded sweeps succeeded: 12 isolated DCT size/backend cases with 3,444
checked calls, plus 36 solver configurations with 2,952 checked solves.
The standalone harness error threshold (`1e-9`) and original-system PCG stopping
rule (`||b-Ax|| <= 1e-10 + 1e-8*||b||`) are unchanged.

## Version and scope

At measurement time, the annotated `v1.1.2` tag object was
`931561b13818dada8708d4cc0bc42cbaddd28dcd`; it resolved to commit
`a6f8d0e12920107f89558ab484071f8c6655841d`. The [release record](release.json)
preserves the tag object and commit metadata obtained from the upstream Git
repository. The previous pin `681ae621f51731f84cded6872dcb1f1082105168`
reports version 1.1.1.

Upstream subsequently retagged v1.1.2 to
`a9a48e61eee91ee7182a013c32a4cae0f64f6da3` (tag object
`2636fcc71e2219021d6a2a7032d6ba852c0effa9`). CMake now pins that revision.
Only Markdown documentation and benchmark-result CSVs changed; the `include`,
`src`, `tests` trees and `CMakeLists.txt` are identical Git objects. The
[pin-update verification](release-pin-update.json) records the changed paths and
matching object IDs. Timings, build/source hashes and the original release
record remain as captured at `a6f8d0e`; measurements were not rerun for this
source-identical retag.

No CHEAP numerical headers or backend adapter
were changed. The existing harness timing boundaries, problems, tolerances,
reference oracles, repetition counts, FFTW planner settings and optimization
flags remain the same. The isolated harness only adds version/algorithm fields
to plan diagnostics. Its validator recognizes both old direct-fallback records
and the new FFT-backed algorithm identifiers. Source excerpts are now located
by function definition instead of fixed line numbers, so a release cannot
silently shift them to unrelated functions.

The new release was built in a separate directory; the original backend build
and all archived results remain intact. This comparison does not independently
establish the asymptotic complexity of arbitrary future versions or all sizes.

## Isolated preconditioner before/after

All times are median **milliseconds** over forty steady calls after a first
call. Old values come from the [archived v1.1.1 experiment](../jetson-dct-2026-10-20/README.md).
New v1.1.2 and FFTW3 values were collected together in this session.

| N | v1.1.1 archived ms | v1.1.2 ms | Old / new | Fresh FFTW3 ms |
|---:|---:|---:|---:|---:|
| 256 | 0.005441 | 0.005568 | 1.0× | 0.004928 |
| 257 | 3.451683 | 0.018208 | 189.6× | 0.016385 |
| 258 | 3.488340 | 0.011777 | 296.2× | 0.010368 |
| 511 | 13.474817 | 0.054658 | 246.5× | 0.029793 |
| 512 | 0.011728 | 0.011905 | 1.0× | 0.010337 |
| 513 | 13.568196 | 0.031201 | 434.9× | 0.016209 |

The four non-power-of-two sizes improve by approximately 190–435× relative to
the historical measurements. The power-of-two medians remain close to their
previous values (about 1.5–2.3% apart); these single-session comparisons do not
establish a statistically significant regression at those sizes.

v1.1.2 is about 1.11–1.15× slower than the fresh FFTW3 controls at N=256, 257,
258 and 512, and about 1.83–1.92× slower at N=511 and 513. The quadratic cliff
is gone; these timings do not establish backend performance parity.

The [derived comparison](dct-comparison.csv), raw summaries, all calls and plan
dumps are retained in [dct/](dct/). The largest relative output discrepancy in
that sweep was `1.17178759e-11`, below the unchanged `1e-9` threshold. References
are direct cosine sums accumulated in long double, with double-precision cosine
evaluation, as in the original diagnostic.

## Runtime plans confirm the new path

Both DCT-II and DCT-III report `direct_r2r=0`, `fft_present=1` at all six sizes.
The new `r2r_algo` diagnostic is 2 for the power-of-two real-DFT route and 3
for the default arbitrary-length route. Observed inner FFT fields are:

| DCT N | Inner FFT length | Algorithm field |
|---:|---:|---|
| 256 | 128 | power of two |
| 257 | 257 | Rader active |
| 258 | 129 | mixed radix 43 |
| 511 | 511 | mixed radix 7 |
| 512 | 256 | power of two |
| 513 | 513 | mixed radix 27 |

The DCT at N=257 remains length 257. Neither CHEAP nor the harness pads that
DCT to 256 or 512; the Toeplitz operator's separate power-of-two embedding is
unchanged. Every plan still requests `FFTW_ESTIMATE`. See, for example,
[the N=257 plan dump](dct/NOTORIOUS-257-plans.txt) and
[the version-specific source excerpts](dct/notorious-path.txt).
The backend-specific struct fields are diagnostic details of this pinned release,
not a portable public API.

## Matched original-system solver comparison

Same AR(1) family `A[i,j]=rho^abs(i-j)`, rho=0.3, 0.9 and 0.99; same fixed
`b[i]=sin(0.1*i)+0.5`, zero initial guess for every solve, `atol=1e-10`,
`rtol=1e-8`, at most `4*N+40` iterations, no periodic residual refresh, and
continue policy if a convergence check requires replacement. Both paths continue
to apply the original finite Toeplitz operator. The DCT projection is used only
as an SPD preconditioner. Every returned solution is independently checked by
an untimed long-double AR(1) matvec oracle.

Below, each arrow means **CG → DCT-PCG**, at rho=0.99. The complete three-rho
results, per-call records, first-call/setup times, p95 and IQR are in [pcg/](pcg/).

| Backend | N | Iterations | Median ms | CG / DCT-PCG |
|---|---:|---:|---:|---:|
| FFTW3 | 17 | 19 → 5 | 0.018209 → 0.008800 | 2.07× |
| FFTW3 | 64 | 48 → 6 | 0.112899 → 0.026305 | 4.29× |
| FFTW3 | 257 | 146 → 8 | 2.675267 → 0.327850 | 8.16× |
| FFTW3 | 512 | 248 → 8 | 5.508363 → 0.319434 | 17.24× |
| FFTW3 | 1024 | 398 → 8 | 18.654550 → 0.681653 | 27.37× |
| FFTW3 | 4096 | 854 → 6 | 193.206371 → 2.623330 | 73.65× |
| NOTORIOUS | 17 | 19 → 5 | 0.025233 → 0.012000 | 2.10× |
| NOTORIOUS | 64 | 49 → 6 | 0.145764 → 0.032097 | 4.54× |
| NOTORIOUS | 257 | 143 → 8 | 3.299063 → 0.393837 | 8.38× |
| NOTORIOUS | 512 | 247 → 8 | 6.640702 → 0.383196 | 17.33× |
| NOTORIOUS | 1024 | 399 → 8 | 22.551886 → 0.810313 | 27.83× |
| NOTORIOUS | 4096 | 853 → 6 | 217.825404 → 2.946347 | 73.93× |

Every one of the 36 same-session solver comparisons has a lower DCT-PCG
steady-state median (minimum ratio 1.33×). The previous Notorious exceptions at
N=17 and N=257 now benefit from preconditioning. At N=4096, rho=0.99, the
original large iteration reduction remains: 853 → 6 for Notorious, with a 73.93×
solve-time ratio. Backend-dependent arithmetic can affect CG iteration counts;
acceptance always uses the same original-system residual criterion.

The largest independent relative residual across all 2,952 solves was
`9.86788362e-9`. This is the same acceptance target as the earlier report,
not an accuracy relaxation used to obtain faster timings.

## Setup and reuse

[pcg-comparison.csv](pcg-comparison.csv) retains measured common setup,
preconditioner-only setup, first calls, derived setup-plus-first-call sums and
modeled reuse thresholds. At N=257, rho=0.99, Notorious common setup was
0.158437 ms and additional preconditioner setup was 0.230951 ms. Summing measured
setup and first-call timings gives 3.481325 ms for CG versus 0.790809 ms for
DCT-PCG. This derived accounting favors the preconditioner even on the first
RHS in this case.

Those sums are not independently measured cold-process end-to-end runs. Each
process times CG first, constructs the preconditioner, then times DCT-PCG.
The reuse model compares `common + first_CG + (k-1)*median_CG` with
`common + pre_setup + first_PCG + (k-1)*median_PCG`, reusing the same RHS timing.
Small-size setup remains material; N=17 Notorious at rho=0.99 reaches the modeled
break-even at approximately two RHSs despite its lower steady-state median.

## Correctness validation

Release CLI/contract checks passed for both isolated binaries and the v1.1.2
solver harness, including independent reference outputs, residual acceptance,
statistics recomputed from raw calls, file-open failures, and retained sampler
embedding rejection. The numerical library is also checked in separate Debug,
Asan and Ubsan builds, including the C++ wrapper, using the full 14-test CTest
suite in each configuration. All 42 tests passed (14 per configuration).
Build commands/logs and CTest logs are retained in
[correctness/](correctness/).

The dedicated Asan harness checks cover all six isolated DCT size classes plus
strong-correlation N=257 PCG and the full N=17 sampler path. They run with
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1` outside sandbox tracing. These tests are separate
from all optimized timing. All eight dedicated sanitizer harness cases passed.
Final statuses are recorded in the logs and [correctness summary](correctness/summary.json).

## Platform and provenance

Same prepared Jetson Orin Nano Engineering Reference Developer Kit Super,
six Cortex-A78AE cores, L4T R36.4.4, Linux 5.15.148-tegra, 15 W profile 0.
CPU clocks were 1.4976 GHz, GPU 612 MHz, observed EMC 2.133 GHz. The runners only
queried settings. All endpoint CPU frequency snapshots reported 1.4976 GHz.
CPU temperatures ranged 45.781–46.718 °C across the two sweeps; junction
47.312–47.937 °C. These are before/after snapshots, not continuous telemetry.

Native GCC/G++ 11.4.0, C99/C++17, double precision. Optimized benchmark and
Notorious implementation objects use `-O3 -DNDEBUG -march=native`, no fast-math
or sanitizers. FFTW3 is the unchanged system 3.3.8 shared library. Every first
call is recorded separately from forty steady calls. Timer is CLOCK_MONOTONIC.

DCT sweep: 2026-10-21 07:40:39–07:40:40 UTC
(2026-10-21 00:40:39–00:40:40 America/Vancouver).
PCG sweep: 07:40:41–07:41:07 UTC (00:40:41–00:41:07 local).
Cases ran sequentially with FFTW3 before Notorious at each size. PCG sweeps
use rho ascending, then N ascending, and CG before DCT-PCG within each process.
No simultaneous build/test workload, sandbox tracing, or CPU affinity pinning
was used during timing.

Each sweep retains `platform.json`, `builds.json` (compile/link commands, binary
hashes and the actual backend commit), `source-hashes.json`, `runs.json` (commands,
exit status, timestamps, temperature/frequency snapshots and validation status),
and power-before/after records. Source base is `fd39c43` plus the pinned release
and harness diagnostic/validator changes identified by those hashes. The
[PCG benchmark source](../../bench/bench_matched.c) itself is unchanged.

Old/new upgrade ratios compare separate sessions on the same configured Jetson;
the old backend was not rerun in this session. Fresh FFTW3 controls and the
same-session CG/DCT-PCG comparisons provide additional context, but do not turn
the historical ratios into randomized or confidence-bounded estimates. These
remain one new Jetson session with a fixed RHS and one Toeplitz family. The
structure-specific AR(1) direct baseline, full covariance recovery and
multidimensional surrogate timing remain separate work.

## Reproduce

CMake now pins the verified v1.1.2 commit. With existing prepared dependencies:

```sh
cmake -S . -B build-jetson-v112-NOTORIOUS -G 'Unix Makefiles' \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS=-march=native \
  -DCHEAP_FFT_BACKEND=NOTORIOUS -DCHEAP_BUILD_EVIDENCE_BENCH=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-jetson-v112-NOTORIOUS --target \
  cheap_bench_preconditioner cheap_bench_evidence -j2
python3 bench/run_preconditioner_sweep.py --output results/v112-repeat/dct \
  --notorious build-jetson-v112-NOTORIOUS/cheap_bench_preconditioner
python3 bench/run_pcg_sweep.py --output results/v112-repeat/pcg \
  --notorious build-jetson-v112-NOTORIOUS/cheap_bench_evidence
```

Both runners default to the explicit-GCC `build-jetson-gcc-FFTW3` control binary
from the previous setup; use `--fftw` to override. Use fresh output directories.
For correctness, the archived build commands configure `Debug`, `Asan` and
`Ubsan` separately with explicit GCC/G++, native flags, C++ enabled and the same
pinned source reused through FetchContent's source-directory override.
