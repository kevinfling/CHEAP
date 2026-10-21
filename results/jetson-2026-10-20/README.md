# Jetson initial size sweep — 2026-10-20

The twelve runs prescribed by [BENCHMARK.md](../../BENCHMARK.md) completed
successfully: both backends at N=17, 64, 257, 512, 1024, and 4096, yielding
36 successful operation rows. FFTW3 had lower steady-state medians in all
three measured operations at every size in this session. These are preliminary
backend timings for one problem family, not a final matched-accuracy comparison
against other algorithms.

## Configuration and provenance

- Device: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super,
  six Cortex-A78AE cores, native AArch64, L4T R36.4.4, Linux 5.15.148-tegra.
- Power profile: 15 W (mode 0); CPU clocks fixed at 1.4976 GHz, GPU at 612 MHz,
  EMC observed at 2.133 GHz. No power or clock settings were changed for this run.
- Source: `4cfff8d3cd3a951ffad4aa1db8f81c4bf8f4cace`, branch
  `feature/matched-accuracy-benchmarks`; benchmark and library sources unmodified.
  The only initial untracked file was the supplied `BENCHMARK.md`.
- Compiler: GCC 11.4.0 for both backends; C99, double precision,
  `-O3 -DNDEBUG -march=native`, no fast-math or sanitizers in timed binaries.
  The Notorious implementation translation unit also received these flags.
- FFTW: system-provided shared library 3.3.8, not rebuilt for this session.
  Notorious-FFT: `681ae621f51731f84cded6872dcb1f1082105168`.
- Planner: `FFTW_ESTIMATE`; 40 individually timed calls after the first call,
  using `CLOCK_MONOTONIC`. The CSV reports setup, first call, median, p95 and IQR.
- Run order: FFTW3 ascending sizes, then Notorious ascending sizes, one process
  per size, no concurrent benchmark processes. CPU affinity was not pinned.
- Timing window: 2026-10-21 05:28:14–05:28:37 UTC
  (2026-10-20 22:28:14–22:28:37 America/Vancouver).
- Snapshots before and after every process showed CPU clocks at 1.4976 GHz;
  CPU temperatures ranged from 45.593 to 46.218 °C and junction temperatures
  from 46.875 to 47.625 °C. These are endpoint snapshots, not continuous monitoring.
  Inactive CV thermal sensors were unavailable and are recorded as such.

See [platform.txt](platform.txt), [builds.json](builds.json) (compiler/link commands,
binary hashes and dependency commit), [runs.json](runs.json) (commands, timestamps,
exit codes, stderr and thermal/CPU frequency snapshots), and
[power-before.txt](power-before.txt) / [power-after.txt](power-after.txt).

## Steady-state median wall times

All values are **milliseconds**. Each pair shows FFTW3 then Notorious.
The raw `<backend>-<N>.csv` files retain setup, first-call and distribution data.

| N | Matvec FFTW3 | Matvec Notorious | CG FFTW3 | CG Notorious | Draw FFTW3 | Draw Notorious |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.000672 | 0.001024 | 0.012961 | 0.018016 | 0.000448 | 0.000592 |
| 64 | 0.001600 | 0.002208 | 0.036097 | 0.046209 | 0.001216 | 0.001472 |
| 257 | 0.015456 | 0.019952 | 0.280327 | 0.354394 | 0.009184 | 0.011072 |
| 512 | 0.016801 | 0.021313 | 0.337321 | 0.411179 | 0.010176 | 0.012064 |
| 1024 | 0.036257 | 0.045377 | 0.719012 | 0.867720 | 0.024417 | 0.028577 |
| 4096 | 0.181797 | 0.211430 | 3.511728 | 3.989452 | 0.117251 | 0.126947 |

The ratio of Notorious median time to FFTW3 median time ranged from 1.16–1.52
for matvec, 1.14–1.39 for CG, and 1.08–1.32 for a sampler draw. These describe
this session; there are no confidence intervals or independent repeated sessions.
The smallest operations take less than a microsecond, making timer overhead
material to their interpretation.

Both operators pad to the next power of two at least `2*N-1`: transform lengths
are 64, 128, 1024, 1024, 2048, and 8192. In particular, N=257 and N=512 use the
same FFT length. The jump from N=64 to N=257 also changes padding substantially;
this sweep does not isolate a cache-related latency discontinuity.

## Accuracy and checks

The matrix is the symmetric Toeplitz covariance `T[i,j] = 0.3^abs(i-j)`;
`b[i] = sin(0.1*i) + 0.5`. CG starts from zero with no preconditioner, at most
`4*N+40` iterations, absolute tolerance `1e-10` and relative tolerance `1e-8`.
Every first and timed solve must report convergence with a valid fresh true
residual or the harness exits with failure. Each CSV retains the first solve's
absolute residual and the final timed solve's iteration count.

The following errors use the larger value across the two backends. CG relative
residual is derived from the CSV's absolute residual divided by `||b||_2`.

| N | CG iterations (both) | CG relative residual | Matvec relative error vs dense oracle | Two-entry covariance discrepancy |
|---:|---:|---:|---:|---:|
| 17 | 13 | 2.423e-09 | 1.779e-16 | 0.037211 |
| 64 | 14 | 3.786e-09 | 1.692e-16 | 0.024336 |
| 257 | 13 | 9.868e-09 | 1.841e-16 | 0.015768 |
| 512 | 13 | 8.525e-09 | 2.424e-16 | 0.015768 |
| 1024 | 13 | 6.476e-09 | 2.270e-16 | 0.012243 |
| 4096 | 13 | 3.276e-09 | not checked | 0.006462 |

Matvec uses an independent dense long-double oracle at N<=1024 with an
acceptance threshold of `1e-10`. N=4096 has no dense oracle check in this harness.

Sampler timing uses a fixed deterministic input and excludes RNG. A separate
seeded check draws 12,000 Gaussian samples and estimates only `Cov(y[0],y[0])`
and `Cov(y[0],y[1])`, subtracting empirical means. The reported discrepancy is
`||(estimated entries) - (1,0.3)||_2 / ||(1,0.3)||_2`, with an acceptance threshold
of 0.12. All embeddings were accepted with approximate clipping disabled and
PSD tolerance `1e-12`. This is not a full covariance matrix validation. Identical
N=257/512 discrepancies follow from their identical FFT length and seeded stream.

The raw CSV's generic `tolerance=1e-8` describes CG's relative tolerance, not
these separate matvec/covariance acceptance thresholds. Its `true_residual`
column holds an absolute residual for CG, relative dense-oracle error for matvec,
and NaN for the draw. NaN marks an inapplicable or unmeasured quantity.
Peak memory is process high-water RSS (10.11–10.48 MiB here), not incremental
operation memory. CG setup includes the shared Toeplitz setup plus workspace
allocation; setup entries across rows must not be summed as independent costs.

Separate correctness validation uses assertion-enabled GCC `Asan` builds
(`-O1 -g -fsanitize=address,undefined`) for `embedding_diagnostics`, `pcg_contract`,
`pcg`, and `toeplitz_operator` in each backend. All eight tests passed with
leak detection enabled. See
[correctness-FFTW3.txt](correctness-FFTW3.txt) and
[correctness-NOTORIOUS.txt](correctness-NOTORIOUS.txt).
The initial sandbox executions failed because LeakSanitizer cannot run under
ptrace; their `correctness-*-sandbox.txt` logs are retained. The tests were rerun
outside the sandbox with `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1`.

## What these results support

The measurements compare two FFT backends for exact Toeplitz matvec,
residual-controlled unpreconditioned CG, and FFT-only PSD embedding draws under
the recorded Jetson configuration. They do not measure a dense/FFT crossover,
preconditioning benefit, RNG-inclusive sampling throughput, full covariance
recovery, 2D/3D cases, or end-to-end application speedups. Structure-aware baselines
and the remaining [publication gates](../../BENCH.md#publication-gate) are still
required before a final comparative claim.

## Reproduce

Ninja was unavailable, so these builds used Unix Makefiles. Compiler selection
is explicit because the local environment defaults to Clang. Setting
`CMAKE_C_FLAGS` also gives the separately compiled Notorious implementation
`-march=native`.

```sh
for backend in FFTW3 NOTORIOUS; do
  cmake -S . -B "build-jetson-gcc-${backend}" -G 'Unix Makefiles' \
    -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS=-march=native \
    -DCHEAP_FFT_BACKEND="${backend}" -DCHEAP_BUILD_EVIDENCE_BENCH=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  cmake --build "build-jetson-gcc-${backend}" --target cheap_bench_evidence -j2
done
mkdir -p results/jetson-repeat
for backend in FFTW3 NOTORIOUS; do
  for n in 17 64 257 512 1024 4096; do
    ./build-jetson-gcc-${backend}/cheap_bench_evidence "$n" \
      > "results/jetson-repeat/${backend}-${n}.csv" || exit "$?"
  done
done
```

Record the platform, power/clock state and temperatures again for a new session;
do not overwrite these archived measurements. Build and run sanitizer checks
separately from timing:

```sh
for backend in FFTW3 NOTORIOUS; do
  cmake -S . -B "build-jetson-asan-${backend}" -G 'Unix Makefiles' \
    -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_BUILD_TYPE=Asan \
    -DCMAKE_C_FLAGS=-march=native -DCHEAP_FFT_BACKEND="${backend}" \
    -DCHEAP_BUILD_CPP=OFF
  cmake --build "build-jetson-asan-${backend}" --target \
    test_toeplitz_operator test_pcg test_pcg_contract test_embedding_diagnostics -j2
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    ctest --test-dir "build-jetson-asan-${backend}" --output-on-failure \
      -R '^(toeplitz_operator|pcg|pcg_contract|embedding_diagnostics)$'
done
```

For the actual Notorious sanitizer build, FetchContent reused the pinned source
already downloaded by the release build through
`-DFETCHCONTENT_SOURCE_DIR_NOTORIOUS_FFT=$PWD/build-jetson-gcc-NOTORIOUS/_deps/notorious_fft-src`.
