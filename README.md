# CHEAP

CHEAP (Circulant Hessian Efficient Algorithm Package) is a C99 toolkit for fast structured linear operators on regular grids, with a C++17 wrapper. Its core applies a chosen spectral multiplier using DCT-II/III transforms in O(N log N) work and O(N) storage. Companion headers provide original Toeplitz products, residual-controlled iterative solves, covariance sampling, and dense entropic transport.

This is a research branch with tested numerical components and several legacy interfaces. The useful idea is reusable transforms and explicit operator contracts. The earlier claim that a DCT directly solves arbitrary Toeplitz or finite-grid fractional Brownian covariance systems was too broad.

| Start here | Contents |
| --- | --- |
| [README.md](README.md) | Build, repository map, lessons learned, strengths and limitations |
| [API.md](API.md) | Operator definitions, lifecycle, numerical contracts and API selection |
| [BENCH.md](BENCH.md) | Recorded evidence, raw data index, reproduction and research gaps |

## Choose the operation

| Goal | Use | What the result means |
| --- | --- | --- |
| Apply a known DCT filter or discrete PDE operator | `cheap.h`, `cheap_weights_verified.h` | The specified discrete operator, up to floating-point error |
| Multiply by a finite symmetric Toeplitz matrix | `cheap_toeplitz_operator.h` | The original matrix-vector product through padded FFT convolution |
| Solve an original SPD system | `cheap_pcg.h` with the original operator callback | A solve accepted after a fresh original-system residual check; DCT preconditioning is optional |
| Draw a stationary 1D Gaussian with specified covariance lags | `cheap_circulant_sampler.h` | Covariance-faithful only when the embedding passes its PSD check, subject to the chosen tolerance |
| Solve entropic OT for an explicit cost matrix | `cheap_transport.h` | Dense log-domain Sinkhorn, O(N²) per sweep |

A Toeplitz product is not a solve. A DCT surrogate solve is exact for its chosen surrogate; its residual on another matrix can still be large. PCG costs depend on iteration count, and sampling costs depend on the accepted embedding length.

## Build and try it

The default backend is FFTW3. Install its development package (`libfftw3-dev` on Debian/Ubuntu), a C/C++ compiler and CMake. CMake searches locally first and downloads FFTW 3.3.10 if needed; that fallback requires network access. C++ targets need C++17 and can be disabled with `-DCHEAP_BUILD_CPP=OFF`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
./build/poisson_solver_2d
./build/toeplitz_solve
```

The example named `toeplitz_solve` demonstrates a DCT surrogate. For the original Toeplitz equation, see the complete [PCG example](API.md#solve-an-original-toeplitz-system).

Save this standalone example as `heat.c`. It smooths initialized data with the heat semigroup of the unit-spacing discrete Neumann Laplacian. The supplied weights determine the operator; this example does not use the context's default covariance spectrum.

```c
#include <stdio.h>
#include "cheap.h"

int main(void) {
    enum { N = 32 };
    cheap_ctx ctx = {0};
    double input[N] = {0}, weights[N], output[N];
    input[N / 2] = 1.0;
    int rc = cheap_init(&ctx, N, 0.5);
    if (rc == CHEAP_OK)
        rc = cheap_weights_laplacian(N, weights);
    if (rc == CHEAP_OK) {
        for (int k = 0; k < N; ++k)
            weights[k] = exp(-0.25 * weights[k]);
        rc = cheap_apply(&ctx, input, weights, output);
    }
    if (rc == CHEAP_OK) printf("center = %.9g\n", output[N / 2]);
    cheap_destroy(&ctx);
    return rc == CHEAP_OK ? 0 : 1;
}
```

```sh
cc -std=c99 -O2 -Wall -Wextra -pedantic -Iinclude heat.c -lfftw3 -lm -o heat
./heat
```

Applications can include the headers directly. In this repository, CMake exposes `cheap` and `cheap_cpp` interface targets. There is currently no install/export package configuration.

An alternative backend is selected at configure time:

```sh
cmake -S . -B build-notorious -DCMAKE_BUILD_TYPE=Debug \
  -DCHEAP_FFT_BACKEND=NOTORIOUS
cmake --build build-notorious --parallel 2
ctest --test-dir build-notorious --output-on-failure
```

CMake fetches Notorious-FFT v1.1.2 at pinned commit `a9a48e61eee91ee7182a013c32a4cae0f64f6da3` and compiles its implementation once. This needs network access unless the dependency is supplied locally. Backend selection is not available at runtime.

## Repository map

| Path | Purpose and first place to look |
| --- | --- |
| [`include/cheap.h`](include/cheap.h) | Main header: contexts, DCT application, legacy weights, spectral scaling and random Fourier features |
| [`include/cheap.hpp`](include/cheap.hpp) | Move-only RAII wrappers for 1D/2D/3D contexts and RFF |
| [`include/cheap_weights_verified.h`](include/cheap_weights_verified.h) | Explicit covariance versus sampling weights, anisotropy, resolvent, wave and Wiener constructors |
| [`include/cheap_toeplitz_operator.h`](include/cheap_toeplitz_operator.h), [`include/cheap_pcg.h`](include/cheap_pcg.h) | Original Toeplitz operator and iterative solver |
| [`include/cheap_circulant_sampler.h`](include/cheap_circulant_sampler.h), [`include/cheap_spectral_diagnostics.h`](include/cheap_spectral_diagnostics.h) | PSD-checked stationary sampling and surrogate diagnostics |
| [`include/cheap_transport.h`](include/cheap_transport.h), [`include/cheap_sinkhorn_divergence.h`](include/cheap_sinkhorn_divergence.h), [`include/cheap_gumbel_transport.h`](include/cheap_gumbel_transport.h) | Dense transport and its debiased/perturbed variants |
| [`include/cheap_toeplitz_dct_diagonal.h`](include/cheap_toeplitz_dct_diagonal.h) | Standalone O(N²) projection reference; main header also has an FFT implementation |
| [`src/`](src/), [`include/cheap_notorious_nd.h`](include/cheap_notorious_nd.h) | Notorious implementation translation unit and ND compatibility adapter |
| [`examples/c/`](examples/c/), [`examples/cpp/`](examples/cpp/) | Matching examples; interpretation follows the selected operator, even when filenames use application names |
| [`tests/`](tests/) | C/C++ correctness, independent references and Python harness validators |
| [`bench/`](bench/) | Evidence harnesses and sequential sweep runners; use for accuracy-qualified comparisons |
| [`benchmarks/`](benchmarks/) | Legacy microbenchmarks; some Release targets use fast-math and cannot establish input-validation behavior |
| [`experiments/`](experiments/) | Exploratory NumPy/SciPy fBm comparisons, outside CTest |
| [`experimental/timberwolf.h`](experimental/timberwolf.h) | Separate C11 application prototype requiring CHEAPDAN; outside the supported build/test surface |
| [`results/`](results/) | Recorded CSVs, plans, logs, hashes and platform metadata; indexed in BENCH.md |
| [`CMakeLists.txt`](CMakeLists.txt), [`cmake/`](cmake/), [CI workflow](.github/workflows/dual-backend-correctness.yml) | Targets, dependency discovery and dual-backend correctness matrix |

## Lessons learned

1. **Start with the matrix and its boundary conditions.** The DCT exactly diagonalizes the specified reflected-boundary discrete Laplacian and operators built from it. Generic Toeplitz covariances and nonstationary fBm levels need approximation or iterative correction. The default Flandrin-inspired spectrum is a model, not an exact finite-grid fBm eigendecomposition.
2. **Measure error in the original problem.** Positive surrogate weights establish an SPD surrogate, not accuracy or positive definiteness of the original matrix. A small surrogate or transformed residual cannot replace `||b-Ax||`. A sampler needs covariance validation, not just finite outputs.
3. **Covariance and sampling weights differ.** Sampling uses the square root of a covariance spectrum. Legacy Matérn helpers conflate the raw exponent with conventional smoothness, and their convenience samplers apply covariance weights directly. Use the additive corrected constructors; see [migration notes](API.md#legacy-names-and-migration).
4. **A spectral exponential is not an entrywise Gibbs kernel.** Legacy `cheap_sinkhorn` scales a spectral operator. That operator need not be entrywise positive, so arbitrary-cost entropic OT claims do not follow. The explicit-cost transport header provides a separate dense solver.
5. **Fewer iterations do not guarantee less time.** An older backend's direct cosine-sum fallback overwhelmed preconditioning gains at awkward sizes. Isolated timings and selected plans identified the issue; the v1.1.2 follow-up removed that measured cliff. Setup, first call, reuse and backend paths all matter.
6. **Compare equal tasks and retain failures.** Recent AR(1) PCG comparisons check every original-system residual. They still use one matrix family and fixed RHS, and omit a structure-specific direct baseline. Established transform projections and preconditioning theory do not become novel because they share one implementation.

## Strengths and current limits

CHEAP is useful when a regular-grid DCT operator is the model you want, transforms and spectra can be reused, or a spectral approximation is an effective preconditioner for a separately supplied SPD operator. It supports rectangular 2D and noncubic 3D transforms, flat spectral arrays, AVX2/NEON pointwise kernels, and cached plans. The exact Toeplitz apply, PCG loop and circulant draw reuse allocated scratch.

The recorded Jetson follow-up found lower DCT-PCG warm medians in all 36 AR(1) comparisons. At N=4096 and correlation 0.99, FFTW3 CG took 193.206 ms versus 2.623 ms for DCT-PCG at the same residual target. This is a single-session comparison against unpreconditioned CG, with setup reported separately. [BENCH.md](BENCH.md) preserves context, older counterexamples and raw evidence.

- **Model scope:** DCT grid boundaries and separable/tensor models do not represent arbitrary scattered data, nonseparable covariance or general PDE boundaries. Exact Toeplitz multiplication and PSD circulant sampling are currently 1D; 2D/3D Toeplitz constructors remain legacy surrogates.
- **Numerics:** validation varies by API. Legacy fast paths may skip checks, and `cheap_apply` does not validate every supplied weight or output. Callers must supply finite valid parameters and weights. No universal approximation rate, mesh-independent PCG convergence or state-of-the-art claim is established.
- **Lifecycle:** initialization/planning can be expensive (`FFTW_PATIENT` for core DCT contexts). Contexts own mutable scratch, cannot be copied, and cannot be shared concurrently. Serialize backend planning/destruction as required by the backend.
- **Portability:** native Linux GCC/Clang on AArch64 and x86 are the intended build environment. CMake adds `-march=native` and GNU-style flags; this is not a portable binary or MSVC build configuration. `cheap_rdtsc` returns architecture-dependent ticks, not a universal CPU-cycle count.
- **Compatibility:** the header reports `0.3.0-tensor-weights`, while CMake still declares `0.2.0`. Legacy names and some source comments overstate mathematical interpretations. API.md describes this branch without silently changing numerical behavior.
- **Evidence:** optimized timings, sanitizer validation and Python experiments are different evidence. Backend parity at all sizes, full covariance recovery, strong direct/structured baselines and broad independent reproduction remain open.

Next work should prioritize the missing direct baseline and independent accuracy checks, legacy API migration, consistent versioning, and fair setup-inclusive multi-RHS experiments. fBm research needs a native increment-domain competitor, broader parameter/RHS sweeps and a completed prior-art review before publication claims.

## Documentation archive and license

Documentation was consolidated on 2026-10-21 into these three guides. Superseded manuscripts, audits, tables, plans and per-run narratives were copied with their original paths into local `old/`, with a SHA-256 manifest, before removal. Original versions of rewritten guides were backed up too. `old/` is excluded through `.git/info/exclude`, so this archive stays on this checkout; tracked originals remain in Git history. Raw evidence and executable source retain their existing paths.

CHEAP is [MIT licensed](LICENSE). FFT backend dependencies carry their own licenses; consult the dependency distribution when redistributing a linked application.
