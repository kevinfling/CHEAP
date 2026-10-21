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
