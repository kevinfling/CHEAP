# Evidence, reproduction and research status

[Overview and lessons](README.md) · [Numerical contracts](API.md)

This is the consolidated guide to the recorded measurements and research audits, as of 2026-10-21. Raw CSVs, per-call samples, plan dumps, provenance, patches and correctness logs remain at their original paths in `results/`. Superseded narratives are backed up locally in `old/` and tracked originals remain in Git history.

## What is being measured

| Operation | Work after setup | Acceptance / interpretation |
| --- | --- | --- |
| DCT multiplier | O(N log N) | Exact for the chosen DCT operator; a surrogate for a different matrix |
| Original Toeplitz FFT product | O(L log L), L>=2N-1 | Finite original matvec, not a solve |
| CG / DCT-PCG | k iterations of original matvec plus optional preconditioner | Fresh `||b-Ax|| <= atol+rtol*||b||`; k depends on the problem |
| Circulant draw | O(L log L) | PSD-accepted embedding; clipping tolerance changes covariance |
| Dense explicit-cost transport | O(N²) per sweep | Row/column marginal acceptance; separate from legacy spectral scaling |

Do not compare a DCT surrogate application with a dense original-system solve as interchangeable tasks. Setup includes allocation, spectrum construction and planning; report it separately from warm application and include it in any end-to-end comparison. The legacy microbenchmark tables mixed these interpretations and are no longer the headline evidence.

## Recorded evidence index

| Session / directory | Scope and conclusion | Evidence to inspect |
| --- | --- | --- |
| [Initial Jetson sweep](results/jetson-2026-10-20/) | N=17,64,257,512,1024,4096; both backends, rho=0.3. All 12 processes succeeded. FFTW3 had lower medians for each measured operation at each size. | `<backend>-<N>.csv`, `platform.txt`, `builds.json`, `runs.json`, power and correctness logs |
| [Correlated Toeplitz solver sweep](results/jetson-pcg-2026-10-20/) | rho=0.3,0.9,0.99, same six sizes and two backends. All 36 configurations / 2,952 solves passed. DCT-PCG reduced iterations, but older Notorious was slower at N=17/257. | `comparison.csv`, summary/sample CSVs, `source-hashes.json`, `runs.json`, `harness.patch.gz` |
| [Isolated DCT investigation](results/jetson-dct-2026-10-20/) | N=256,257,258,511,512,513. All 12 cases / 3,444 calls passed. Older Notorious selected quadratic direct DCTs at all four non-power-of-two sizes. | `comparison.csv`, per-stage/sample CSVs, plan dumps, `notorious-path.txt`, build/source provenance |
| [Notorious v1.1.2 follow-up](results/jetson-v112-2026-10-21/) | Fresh controls and both sweeps: all 3,444 isolated calls and 2,952 solves passed. Runtime DCT plans are FFT-backed at all six sizes. All 36 solver comparisons have lower DCT-PCG warm medians. | `dct-comparison.csv`, `pcg-comparison.csv`, `dct/`, `pcg/`, `correctness/`, release records |

The 2026-10-20 sessions used Notorious v1.1.1 at `681ae621f51731f84cded6872dcb1f1082105168`. The follow-up measured v1.1.2 at `a6f8d0e12920107f89558ab484071f8c6655841d`. CMake pins its documentation-only retag `a9a48e61eee91ee7182a013c32a4cae0f64f6da3`; [pin verification](results/jetson-v112-2026-10-21/release-pin-update.json) records identical numerical source/build-input trees. These are recorded revisions, not claims about the latest upstream release.

### Latest isolated preconditioner results

Median milliseconds over 40 warm calls after a separately recorded first call:

| N | Archived v1.1.1 | v1.1.2 | Fresh FFTW3 |
| ---: | ---: | ---: | ---: |
| 256 | 0.005441 | 0.005568 | 0.004928 |
| 257 | 3.451683 | 0.018208 | 0.016385 |
| 258 | 3.488340 | 0.011777 | 0.010368 |
| 511 | 13.474817 | 0.054658 | 0.029793 |
| 512 | 0.011728 | 0.011905 | 0.010337 |
| 513 | 13.568196 | 0.031201 | 0.016209 |

Source: [derived DCT comparison](results/jetson-v112-2026-10-21/dct-comparison.csv). The awkward-size cliff disappeared in this follow-up; FFTW3 remains faster at every listed size. Old/new comparisons span sessions, while new FFTW3/Notorious controls share a session. They do not establish performance parity or universal scaling.

The earlier component measurements attributed roughly 99.5–99.7% of the slow application to isolated DCT-II/III medians. Those component medians were measured separately, so their sum is descriptive accounting rather than a full-call profile. Plan dumps and pinned-source excerpts identify the actual selected path; length alone is not proof of the algorithm.

### Latest original-system solve results

At rho=0.99, every arrow denotes **unpreconditioned CG → DCT-PCG**. Times are warm median milliseconds; setup is excluded here and retained in the CSVs.

| Backend | N | Iterations | Median ms | CG / DCT-PCG |
| --- | ---: | ---: | ---: | ---: |
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

Source: [paired solver comparison](results/jetson-v112-2026-10-21/pcg-comparison.csv). All three correlations, distributions and individual calls are in [pcg/](results/jetson-v112-2026-10-21/pcg/). The largest independent relative residual over all 2,952 solves was `9.86788362e-9`. Different backend arithmetic can change iteration counts; the acceptance target is identical.

This demonstrates benefit over unpreconditioned CG for these problems, not superiority over a structure-specific direct solver. The minimum same-session ratio across all 36 configurations was 1.33×. In older runs Notorious lost time despite fewer iterations at N=17/257; those negative observations remain in the historical CSVs.

## Problem, timing and validation contracts

The solver matrix is `A[i,j]=rho^abs(i-j)` with rho=0.3,0.9,0.99 and no ridge. The fixed RHS is `b[i]=sin(0.1*i)+0.5`. Every solve starts at zero, uses `atol=1e-10`, `rtol=1e-8`, at most `4*N+40` iterations, no periodic replacement, and continue policy if a convergence-candidate check requires replacement.

CG and DCT-PCG use the same original Toeplitz FFT matvec. Only the preconditioner differs: inverse Frobenius-optimal DCT projection weights, without flooring to force success. Each first/warm solution is independently checked outside timing by an untimed long-double AR(1) geometric recurrence, cross-checked against dense matvec for N<=1024.

The full evidence mode also checks matvec against a dense oracle at N<=1024 (threshold `1e-10`) and checks two covariance entries with 12,000 seeded Gaussian draws (discrepancy threshold 0.12). Two entries are not full covariance recovery. Timed sampler calls repeat fixed Gaussian input and exclude RNG. Strong-correlation embeddings can legitimately be rejected; use `pcg` mode for the solve-only sweep.

The isolated DCT harness measures seven operations: adapter apply, underlying cheap_apply, validate/copy, forward DCT, multiplication, inverse DCT, normalization/copy. Each has an independent reference, first call and 40 warm calls. Its output discrepancy threshold is `1e-9`; maximum in the follow-up was `1.17178759e-11`. References use long-double sums with double-precision cosine evaluation.

These harnesses locally request `FFTW_ESTIMATE` for DCT plans, overriding production `FFTW_PATIENT` **only inside the benchmark**. Toeplitz/sampler FFT plans also use ESTIMATE. Numerical library behavior is unchanged.

CSV interpretation:

- `setup_ms = common_setup_ms + preconditioner_setup_ms` for solver rows. Setup is not charged entirely to the preconditioner.
- First execution, warm median, p95 and IQR are separate. The legacy `true_residual` column is the first solve's absolute norm; explicitly named relative/max fields describe their own scope.
- Per-call sample logs retain first and all 40 warm calls, failures, original residual acceptance and operator/preconditioner counts. Legacy summary iterations/counts refer to the last call; maximum residual fields cover all calls.
- `backend` may label the operation path; `configured_backend` identifies the FFT library. Peak memory is process high-water RSS, not incremental operation memory.
- Argument, setup, rejected embedding and output errors cause nonzero process exits. Runners retain failures. Never omit them from a comparison.

### Platform and setup interpretation

Recorded Jetson sessions used the Orin Nano Engineering Reference Developer Kit Super, six Cortex-A78AE cores, AArch64, Linux 5.15.148-tegra/L4T R36.4.4, 15 W profile 0, observed CPU 1.4976 GHz, GPU 612 MHz and EMC 2.133 GHz. GCC/G++ 11.4.0, C99/C++17, double precision; timed objects use `-O3 -DNDEBUG -march=native`, without fast-math/sanitizers. FFTW3 is the system 3.3.8 shared library. Timer: CLOCK_MONOTONIC. The runners queried power/clocks without changing them.

Follow-up DCT sweep: 2026-10-21 07:40:39–07:40:40 UTC; PCG: 07:40:41–07:41:07 UTC. Cases ran sequentially, FFTW3 before Notorious at each size; PCG order was rho ascending, size ascending, CG then DCT-PCG. No affinity pinning or simultaneous build/test load. Endpoint CPU temperatures were 45.781–46.718 °C; snapshots do not establish continuous thermal stability. Full per-session details remain in platform/build/run/power records.

At N=257/rho=0.99, Notorious common setup was 0.158437 ms and preconditioner setup 0.230951 ms. Measured setup-plus-first-call sums are 3.481325 ms for CG and 0.790809 ms for PCG. These are derived sums, not independent cold-process timings. Modeled reuse compares:

```text
CG:  common + first_CG + (RHS_count-1)*median_CG
PCG: common + pre_setup + first_PCG + (RHS_count-1)*median_PCG
```

This uses repeated timings of one RHS, not independent RHS workloads. At N=17/rho=0.99, Notorious reaches modeled break-even at roughly two RHSs. Small-case setup and timer overhead remain material.

## Reproduce builds and checks

Default FFTW3 build instructions are in README.md. To build both optimized evidence harnesses, with installed dependencies and Git/network access for the pinned alternative backend:

```sh
for backend in FFTW3 NOTORIOUS; do
  cmake -S . -B "build-evidence-${backend}" \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCHEAP_FFT_BACKEND="${backend}" -DCHEAP_BUILD_EVIDENCE_BENCH=ON
  cmake --build "build-evidence-${backend}" --parallel 2 \
    --target cheap_bench_evidence cheap_bench_preconditioner
done
```

Individual runs:

```sh
./build-evidence-FFTW3/cheap_bench_evidence 257 0.99 pcg /tmp/cheap-solves.csv
./build-evidence-FFTW3/cheap_bench_evidence 17 0.3 all
./build-evidence-FFTW3/cheap_bench_preconditioner 257 /tmp/cheap-dct 0.99
```

Evidence CLI: `[n [rho [all|pcg [solve_samples.csv]]]]`, defaults `257 0.3 all`, 2<=n<=65536, finite 0<=rho<1. Isolated CLI: `n output_prefix [rho]`, 2<=n<=4096, default rho=0.99. Summary CSV goes to stdout; the isolated prefix names sample/plan files.

Python standard-library validators are separate from CTest:

```sh
python3 tests/test_bench_evidence.py \
  build-evidence-FFTW3/cheap_bench_evidence \
  build-evidence-NOTORIOUS/cheap_bench_evidence
python3 tests/test_bench_preconditioner.py \
  build-evidence-FFTW3/cheap_bench_preconditioner \
  build-evidence-NOTORIOUS/cheap_bench_preconditioner
```

Run sweeps **sequentially into fresh directories**. Explicit binary arguments avoid the runners' historical Jetson build-directory defaults:

```sh
python3 bench/run_preconditioner_sweep.py --output results/repeat-dct \
  --fftw build-evidence-FFTW3/cheap_bench_preconditioner \
  --notorious build-evidence-NOTORIOUS/cheap_bench_preconditioner
python3 bench/run_pcg_sweep.py --output results/repeat-pcg \
  --fftw build-evidence-FFTW3/cheap_bench_evidence \
  --notorious build-evidence-NOTORIOUS/cheap_bench_evidence
```

Runners preserve build/link commands, binary hashes, actual dependency revision, source hashes, command/status records, raw calls and available platform/power/thermal snapshots. They require both binaries and their build trees for provenance. Missing Jetson-specific queries on other systems are recorded rather than assumed equivalent.

Correctness runs should be separate from timing:

```sh
cmake -S . -B build-check -DCMAKE_BUILD_TYPE=Debug -DCHEAP_FFT_BACKEND=FFTW3
cmake --build build-check --parallel 2
ctest --test-dir build-check --output-on-failure
```

Repeat with `Asan` (address+undefined) and `Ubsan`, and with NOTORIOUS. Use `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and `UBSAN_OPTIONS=halt_on_error=1` where supported. The [workflow](.github/workflows/dual-backend-correctness.yml) defines both backends × Debug/Asan/Ubsan. Its push trigger still names the historical research branch; pull requests/manual runs also trigger it. This definition is not proof that a current remote run passed.

The archived v1.1.2 evaluation records 14 CTests passing in each Notorious Debug/Asan/Ubsan build (42 total), plus eight dedicated sanitizer harness cases with leak detection. Earlier backend failures were historical blockers; the pinned follow-up's recorded results supersede them for the tested scope. No blanket all-platform memory-safety claim follows.

Two independent reference sources are not registered in CTest. Run explicitly:

```sh
cc -std=c99 -O2 -Wall -Wextra -pedantic tests/test_spectral_reference.c \
  -lm -o /tmp/cheap-spectral-reference
/tmp/cheap-spectral-reference
cc -std=c99 -O2 -Wall -Wextra -pedantic tests/test_toeplitz_dct_diagonal.c \
  -lm -o /tmp/cheap-projection-reference
/tmp/cheap-projection-reference
```

Legacy executables `bench_cheap`, `bench_cheap_scalar`, `bench_cheap_stats`, `bench_cheap_cpp` remain available for local exploration. Their results are not matched original-system solve evidence. Different formulas taking different times do not establish SIMD speedup; compare the same workload with/without SIMD. ARM timer ticks are not CPU cycles.

## fBm research and inherited lessons

Executable NumPy/SciPy experiments remain in [`experiments/`](experiments/). They are exploratory research code, not native-library performance evidence or CTest coverage:

| Script | Purpose |
| --- | --- |
| `dct_preconditioner.py` | Dense covariance/projection/solution error and iteration comparison |
| `fbm_matrixfree.py` | FFT Toeplitz-plus-low-rank fBm application and DCT preconditioning |
| `fbm_pcg_sweep.py`, `preconditioner_sweep.py` | Parameter/conditioning sweeps |
| `fbm_increment_domain_comparison.py` | Original-system CG/DCT-PCG versus transformed fGn-Chan |

For example, with NumPy/SciPy installed:

```sh
OPENBLAS_NUM_THREADS=1 python3 experiments/dct_preconditioner.py
OPENBLAS_NUM_THREADS=1 python3 experiments/fbm_increment_domain_comparison.py
```

For t_i=(i+1)/N, the fBm covariance has exact representation `A=T(c)+u*1^T+1*u^T`, where `u_i=t_i^(2H)/2`, `c_d=-abs(d/N)^(2H)/2`. The low-rank term contributes only to the DC **diagonal** in DCT coordinates; its off-diagonal entries generally remain. The DCT projection is therefore still approximate.

Let D be the nonsingular first-difference matrix with first row e_0^T and G=DAD^T the stationary fGn covariance. Then `(A+rho*I)x=b` is equivalent to `(G+rho*D*D^T)y=D*b`, `x=D^T*y`. A transformed residual does not directly control the original residual. The exploratory comparison used transformed tolerance `1e-12` for fGn-Chan versus `1e-9` for original-system solvers, accepting every result only at original relative residual <=`1.1e-9`.

Archived stress notes reported DCT-PCG competitive around H=.3/.7, while fGn-Chan required fewer iterations near H=.9/.95 with very weak regularization. These notes do not establish universal superiority. Reports referring to an external `cheap_fbm_fftw.c` artifact are not reproducible from this checkout: that file is absent. The committed native evidence harness tests AR(1), not the full fBm comparison. Uncommitted-artifact timings are not promoted to current benchmark results.

For B the DCT projection and M=B+rho*I SPD, define `delta=||M^(-1/2)(A-B)M^(-1/2)||_2`. If delta<1, the preconditioned eigenvalues lie in `[1-delta,1+delta]`, giving condition bound `(1+delta)/(1-delta)`. This standard perturbation bound supplies no N/H-uniform estimate. Nearly constant observed iteration counts are not a mesh-independence theorem.

Transform-algebra projection, cosine preconditioning, DCT/fBm approximate eigenbasis relationships, and stationary-increment sampling are established research areas. The archived prior-art map prioritizes Kailath–Olshevsky on trigonometric preconditioners, Gupta–Joshi on DCT/fBm eigenvectors, Chan/Strang on circulant preconditioning, Lindgren–Rue–Lindström on SPDE fields, and Cuturi/Solomon on Sinkhorn/convolutional OT. Full-text overlap/priority review remains incomplete; no novelty claim is made here.

## Open validation and publication work

1. Add a structure-specific AR(1) direct baseline, established circulant/boundary-corrected competitors, and native fGn-Chan for fBm. Compare equal discretizations, precision and original-system accuracy.
2. Separate common and method-specific setup; measure cold/warm setup and actual 1/10/100 independent RHS workloads. Record memory, operator/FFT counts, iterations, residual histories and all failures.
3. Broaden N, matrix families, random/adversarial RHSs, conditioning and regularization. For fBm include H near .01/.99 and scaled ridge from 1e-8 to 1e-2; verify dense covariance/projection/Cholesky at small N.
4. Use paired randomized method ordering, at least 30 independent timing samples, median/IQR and bootstrap confidence intervals, with multiple machines/compilers. Current ordered repeated-call sessions provide neither independent problems nor confidence bounds.
5. Measure full empirical covariance with uncertainty, not only two entries. Treat rejected/materially clipped embeddings as failures/approximations. Test rectangular 2D/noncubic 3D surrogate timings with their boundaries disclosed.
6. For OT compare explicit-cost log-domain reference and structured positive-kernel methods at matched marginal/objective error; spectral positivity alone is insufficient. Certify any proposed fast kernel's entrywise positivity and quantify its approximation error.
7. Expand independent weight/covariance checks, ND axis-exchange tests and sanitizer coverage; migrate legacy Matérn/spectral scaling names without silently changing results.
8. Complete prior-art review and any fBm-specific spectral bounds before claiming novelty, universal convergence, publication readiness or state of the art.
