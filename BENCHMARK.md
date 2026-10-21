Jetson benchmark configuration

I’d use native AArch64 builds, both FFT backends, double precision, and identical optimization flags.

Parameter	Configuration
Architecture	AArch64 / Cortex-A78AE
Compiler	GCC, same version for both backends
Optimization	-O3 -DNDEBUG -march=native
FFTW planning	FFTW_ESTIMATE
Notorious-FFT	Pinned commit 681ae621
Timing	40+ steady-state repetitions
Accuracy	Identical true-residual and covariance targets
Thermal/power state	Recorded and held as consistently as possible

1. Prepare the Jetson

sudo apt update
sudo apt install -y \
  git cmake ninja-build build-essential \
  pkg-config libfftw3-dev python3

For a dedicated benchmark session, with adequate cooling and power:

sudo nvpmodel -q
sudo jetson_clocks --show

If appropriate for your power supply and thermal setup, select a supported performance profile and lock clocks:

sudo nvpmodel -m <supported-mode-id>
sudo jetson_clocks

The mode ID is device- and JetPack-dependent; check sudo nvpmodel -q --verbose rather than assuming a particular value.

2. Build both backends

Native AArch64 build

Copy

git clone https://github.com/kevinfling/CHEAP.git
cd CHEAP
git fetch origin feature/matched-accuracy-benchmarks
git checkout feature/matched-accuracy-benchmarks
for backend in FFTW3 NOTORIOUS; do
  cmake -S . -B "build-${backend}" -G Ninja \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_FLAGS=-march=native -DCMAKE_BUILD_TYPE=Release \
    -DCHEAP_FFT_BACKEND="${backend}" \
    -DCHEAP_BUILD_EVIDENCE_BENCH=ON
  cmake --build "build-${backend}" \
    --target cheap_bench_evidence
done

3. Run the initial size sweep

mkdir -p results
for backend in FFTW3 NOTORIOUS; do
  for n in 17 64 257 512 1024 4096; do
    ./build-${backend}/cheap_bench_evidence "$n" \
      > "results/${backend}-${n}.csv"
  done
done

These are preliminary timings from the current harness. They are useful for spotting ARM-specific behavior, but they are not yet the final matched-accuracy comparison: the structure-aware baselines and full covariance measurements remain to be implemented. The current harness does check two covariance entries using 12,000 seeded draws, separately from FFT-only draw timing.

4. Capture provenance

{
  uname -a
  lscpu
  gcc --version | head -1
  cmake --version | head -1
  pkg-config --modversion fftw3
  git rev-parse HEAD
  cat /etc/nv_tegra_release 2>/dev/null || true
  sudo nvpmodel -q
  sudo jetson_clocks --show
} > results/platform.txt

What I’m particularly interested in

The A78AE’s cache hierarchy and the Jetson’s shared-memory architecture could make this a useful test of whether our planned FFT matvec remains competitive across awkward sizes.

I’d especially watch the crossover between dense matvec and FFT-based Toeplitz matvec, the effect of preconditioning on time-to-residual, and whether the larger FFT working sets produce a visible latency discontinuity.

We should not interpret any crossover as universal: it will be specific to this Jetson, the compiler, FFT backend, planner settings, and problem family.

Start with the build and size sweep above. Once the results are available, we can incorporate the measured data into PR #5 and determine which performance claims the evidence actually supports.


## Recorded Jetson results — 2026-10-20

The [initial size sweep report](results/jetson-2026-10-20/README.md) includes all
twelve successful backend/size runs, raw CSVs, build and platform provenance,
power/clock captures, per-run thermal snapshots and separate sanitizer logs.
FFTW3 had lower median times for all measured operations at every size.
At N=4096, FFTW3/Notorious medians were 0.182/0.211 ms for exact Toeplitz matvec,
3.512/3.989 ms for unpreconditioned CG, and 0.117/0.127 ms for FFT-only draws.
CG converged in 13–14 iterations across the sweep.

The prepared board stayed in 15 W mode with recorded CPU clocks at 1.4976 GHz.
The actual build used Unix Makefiles because Ninja was unavailable, and explicitly
selected GCC 11.4.0 for both backends. See the report for exact reproduction
commands, residual-field interpretation and covariance-check limitations.


## Follow-up: stronger-correlation DCT preconditioning

The [paired solver report](results/jetson-pcg-2026-10-20/README.md) compares
unpreconditioned CG with the existing DCT-preconditioned PCG implementation at
rho=0.3, 0.9 and 0.99 for both backends and the same six sizes. Only the harness
was expanded; no numerical library algorithms were modified. All 2,952 solves
passed the same true-residual target and an independent original-system check.

At N=4096, rho=0.99, FFTW3 went from 854 iterations / 193.027 ms to
6 iterations / 2.636 ms. Notorious went from 853 iterations / 217.380 ms to
6 iterations / 2.941 ms. Notorious's N=17 and N=257 DCT-PCG cases were slower
despite lower iteration counts; those measurements are retained in the report.

Rebuild both evidence binaries, then run sequentially into a fresh directory:

```sh
python3 bench/run_pcg_sweep.py --output results/jetson-pcg-repeat
```

The runner defaults to the `build-jetson-gcc-FFTW3` and
`build-jetson-gcc-NOTORIOUS` binaries used on this Jetson. Override them with
`--fftw` and `--notorious` if using the build directories above. It captures
platform/build/source provenance, power/clock queries, per-process snapshots,
exit status, raw per-solve data and validated summaries. Use `pcg` mode for an
individual pair (for example, `cheap_bench_evidence 257 0.99 pcg samples.csv`);
the optional log records every first and steady solve. See [BENCH.md](BENCH.md)
for CLI and CSV field definitions.


## Focused DCT preconditioner investigation

The [isolated application/stage report](results/jetson-dct-2026-10-20/README.md)
measures N=256, 257, 258, 511, 512 and 513 with both backends and
`FFTW_ESTIMATE`, including selected plans. All 12 cases / 3,444 calls passed
independent output checks. The runtime Notorious plans select `direct_r2r=1`
at every tested non-power-of-two length; the pinned source implements that
path as quadratic cosine sums. The isolated DCT-II/III timings account for
nearly all of the observed slow preconditioner application.

At N=257, standalone Notorious preconditioning takes 3.452 ms versus
0.016449 ms for FFTW3; at N=256 the corresponding medians are
0.005441 / 0.004960 ms. The library and FFT backend algorithms remain unchanged.
The new `cheap_bench_preconditioner` target is enabled by the existing
`CHEAP_BUILD_EVIDENCE_BENCH` option; `bench/run_preconditioner_sweep.py` captures
the sequential sweep, raw calls, plan diagnostics and provenance. See the
report for build commands, stage timing boundaries and interpretation limits.
