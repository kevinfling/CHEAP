# Isolated DCT preconditioner investigation — Jetson, 2026-10-20

The awkward-size slowdown is reproduced by the existing DCT preconditioner
outside PCG. Runtime plan fields and the pinned backend source identify the
costly path: Notorious-FFT selects direct, quadratic cosine-sum DCT-II/III
transforms for non-power-of-two sizes. N=258 and N=513 exhibit the same effect
as N=257, so this is broader than a prime-length anomaly.

All twelve requested size/backend cases passed: N=256, 257, 258, 511, 512, 513,
both FFTW3 and Notorious, with identical `FFTW_ESTIMATE` planning. The harness
records seven operations per case, each with one first call and forty steady
calls: 84 summaries and 3,444 individually verified call records. No numerical
library or backend algorithms were changed.

## One preconditioner application

Median wall time in **milliseconds**, excluding setup and validation. The ratio
is Notorious time divided by FFTW3 time. Each application uses the existing
`cheap_pcg_dct_apply` API with inverse Frobenius-optimal DCT projection weights
for `A[i,j]=0.99^abs(i-j)` and fixed input `sin(0.1*i)+0.5`.

| N | FFTW3 ms | Notorious ms | Notorious / FFTW3 | Notorious selected path |
|---:|---:|---:|---:|---|
| 256 | 0.004960 | 0.005441 | 1.1× | FFT-backed |
| 257 | 0.016449 | 3.451683 | 209.8× | direct cosine sums |
| 258 | 0.010401 | 3.488340 | 335.4× | direct cosine sums |
| 511 | 0.029889 | 13.474817 | 450.8× | direct cosine sums |
| 512 | 0.010304 | 0.011728 | 1.1× | FFT-backed |
| 513 | 0.016225 | 13.568196 | 836.3× | direct cosine sums |

For Notorious, moving from 256 to 257 increases application time from about
5.44 microseconds to 3.452 milliseconds. At 511/512/513 the application times
are 13.475 / 0.011728 / 13.568 milliseconds. FFTW3 also varies with length, but
uses substantially faster plans at the four non-power-of-two sizes.

## Component costs and wrapper overhead

Notorious median milliseconds:

| N | Preconditioner adapter | Underlying cheap_apply | DCT-II alone | DCT-III alone |
|---:|---:|---:|---:|---:|
| 256 | 0.005441 | 0.005441 | 0.002592 | 0.002368 |
| 257 | 3.451683 | 3.449186 | 1.746914 | 1.691056 |
| 258 | 3.488340 | 3.485635 | 1.764211 | 1.707281 |
| 511 | 13.474817 | 13.481009 | 6.831523 | 6.608029 |
| 512 | 0.011728 | 0.011809 | 0.005488 | 0.004992 |
| 513 | 13.568196 | 13.562836 | 6.872740 | 6.656062 |

At the four non-power-of-two sizes, the sum of separately measured DCT-II and
DCT-III medians is about 99.5–99.7% of the complete adapter's median. Validation
plus input copying, spectral multiplication, and normalization/output copying
are each below one microsecond. The adapter and direct `cheap_apply` timings
are close, so the large cost is already present in the transform pipeline
without PCG or its adapter checks.

Component medians are measured separately, with fixed input reset outside the
timed region. Their sums are descriptive accounting estimates, not an exact
profile of a complete application: timer overhead, input values, cache state
and measurement order can differ. Full application timings include the two
transforms, validation/copying, weights and normalization; stage timings use
the same context plans and existing multiplication/scale helpers.

As a consistency check against the [paired solver sweep](../jetson-pcg-2026-10-20/README.md),
eight preconditioner calls at N=257 cost approximately `8*3.451683=27.613 ms`
using this isolated median, close to the prior complete DCT-PCG solve median of
27.810 ms. This is a cross-run accounting check, not a direct PCG profile or an
exact measurement of the remaining solve overhead.

## Selected plans and source evidence

Every case retains `FFTW3-<N>-plans.txt` and `NOTORIOUS-<N>-plans.txt`.
FFTW3 plans are printed by its `fftw_fprint_plan` API. For example, the N=257
plans contain `dht-rader-257/256`; the complete [forward and inverse plan dump](FFTW3-257-plans.txt)
is retained rather than inferring the selected algorithm from N alone.

Notorious has exposed plan structures at the pinned revision. Its diagnostics
record rank, N, requested flags, transform kind, in-place layout, the axis
auxiliary's `direct_r2r` flag and whether it owns an FFT plan. Where present,
the FFT length and mixed-radix/Bluestein/Rader fields are also recorded.
These fields are backend-specific diagnostics, not a portable public ABI.

- N=256 and 512: both DCT plans have `direct_r2r=0`, `fft_present=1`, with
  inner FFT lengths 128 and 256 respectively.
- N=257, 258, 511 and 513: both DCT plans have `direct_r2r=1`, `fft_present=0`.
- All plans request `FFTW_ESTIMATE` (numeric flag 64 in the Notorious dumps).

The [pinned-source excerpts](notorious-path.txt) record why these fields matter.
`notorious_fft_mkaux_t2t3_1d` sets `direct_r2r=1` for N=1 or a non-power-of-two
length. Both DCT entry points dispatch on that flag to
`notorious_fft_direct_t2t3`, whose nested N-length loops evaluate cosine sums.
The measured runtime flags and isolated transform costs therefore identify an
actual quadratic fallback in this revision, beyond the size-only inference
available in the previous sweep. No attribution to cache misses or a particular
hardware instruction is made; no hardware-counter profile was collected.

The fallback remains unchanged. This experiment diagnoses the path; it does
not benchmark a repaired backend or a different mathematical preconditioner.

## Verification and measurement scope

The benchmark evaluates direct cosine sums outside the timed region as an
independent reference for DCT-II, DCT-III and the complete inverse-weighted
application. Reference sums accumulate in long double, with cosine arguments
and evaluations in double precision. Copy, multiply and scale stages also have
independent expected outputs. Every call must return success, produce finite
output and meet relative error `1e-9`; the largest measured discrepancy was
`1.42954496e-11`. This checks preconditioner application accuracy, not a solve
residual or covariance recovery.

The seven timed operations are, in order: `preconditioner_apply`, `cheap_apply`,
`validate_copy`, `dct2_execute`, `weight_multiply`, `dct3_execute`, `scale_copy`.
Raw `<backend>-<N>.csv` files report setup, first call, median, p95, IQR, maximum
reference error and process high-water RSS. `*-samples.csv` contains all first
and steady calls with individual errors/status. Setup is repeated as context
metadata across stages and must not be summed across rows.
[comparison.csv](comparison.csv) contains derived median ratios and stage sums.

Each isolated component starts with fixed b copied into the workspace outside
timing. This prevents cumulative transformation/overflow and measures one
operation per timer interval, rather than a dependent sequence. Stage transforms
therefore use fixed coefficients that differ from those inside a complete apply.
Every expected output is checked after its timer stops. References, sample
logging, resetting and plan printing are untimed. Tiny copy/arithmetic timings
are sensitive to timer/call overhead; no overhead was subtracted.

Separate GCC Asan/UBSan builds checked all seven operations and plan inspection
at N=256 and N=257 for each backend, with assertions and leak detection enabled.
All four sanitizer cases passed; see [FFTW3](correctness-FFTW3.txt) and
[Notorious](correctness-NOTORIOUS.txt) logs. The release harness contract test
also passed strict CLI/error handling, reference checks, raw-log statistics and
plan-field validation on both backends.

## Platform and provenance

Same prepared Jetson Orin Nano Engineering Reference Developer Kit Super,
six Cortex-A78AE cores, Linux 5.15.148-tegra, L4T R36.4.4; 15 W mode 0,
CPU clocks at 1.4976 GHz, GPU 612 MHz, observed EMC 2.133 GHz. No settings changed.
Both native C99 binaries and the compiled Notorious implementation used GCC
11.4.0, `-O3 -DNDEBUG -march=native`, no fast-math or sanitizers. FFTW is the
system shared library 3.3.8; Notorious is pinned at
`681ae621f51731f84cded6872dcb1f1082105168`.

Final recorded sweep: 2026-10-21 06:04:29–06:04:34 UTC
(2026-10-20 23:04:29–23:04:34 America/Vancouver). Cases ran sequentially in
ascending N, FFTW3 followed by Notorious per N, outside sandbox tracing and with
no simultaneous build or test workload. An initial diagnostic capture preceded
this final capture; the latter includes a corrected DCT-III source excerpt.
Results here come exclusively from the final capture, not pooled samples or
selected minima. No randomized order, repeated-session confidence intervals or
hardware-counter measurements are claimed. CPU affinity was not pinned.

Endpoint snapshots recorded CPU clocks at 1.4976 GHz throughout. CPU temperature
was 46.125–46.687 °C; junction temperature 47.656–48.156 °C. These are per-process
before/after snapshots, not continuous telemetry; inactive CV sensors are marked
unavailable. See [runs.json](runs.json), [platform.json](platform.json),
[power-before.json](power-before.json) and [power-after.json](power-after.json).

Source base is `af7df71`; the new benchmark/runner/test files and CMake target are
identified by [source-hashes.json](source-hashes.json) and the
[CMake patch](cmake.patch.gz). [builds.json](builds.json) records compile/link
commands, binary hashes and the backend commit. The numerical headers and
backend adapter match the base commit.

## Reproduce

Use the existing GCC/native release builds described in the
[paired solver report](../jetson-pcg-2026-10-20/README.md), with
`CHEAP_BUILD_EVIDENCE_BENCH=ON`, and reconfigure to discover the new target:

```sh
for backend in FFTW3 NOTORIOUS; do
  cmake -S . -B "build-jetson-gcc-${backend}"
  cmake --build "build-jetson-gcc-${backend}" --target cheap_bench_preconditioner -j2
done
python3 tests/test_bench_preconditioner.py \
  build-jetson-gcc-FFTW3/cheap_bench_preconditioner \
  build-jetson-gcc-NOTORIOUS/cheap_bench_preconditioner
python3 bench/run_preconditioner_sweep.py --output results/jetson-dct-repeat
```

The runner reuses the prior runner's platform/thermal query helpers, queries
power/clocks without changing settings, validates all rows against the raw logs
and retains failures. It refuses to overwrite existing result directories.
Override binaries with `--fftw` / `--notorious`. One case is:

```sh
./build-jetson-gcc-NOTORIOUS/cheap_bench_preconditioner 257 /tmp/notorious-257
```

CLI: `N OUTPUT_PREFIX [rho]`, N=2..4096, finite `0 <= rho < 1`, default rho=0.99.
The independent dense-transform oracle makes this deliberately a small-size
microbenchmark. Summary CSV goes to stdout; samples and plan diagnostics go to
`OUTPUT_PREFIX-samples.csv` and `OUTPUT_PREFIX-plans.txt`.
