# CHEAP

**Circulant Hessian Efficient Algorithm Package** — a header-only C99 library for O(N log N) spectral algorithms on structured covariance matrices.

CHEAP turns cubic-time kernel methods into fast spectral operations by exploiting the fact that stationary covariance matrices (Toeplitz) diagonalizes in the DCT-II basis. One forward DCT, one pointwise multiply, one inverse DCT — that's the entire algorithm for kernel regression, optimal transport, Gaussian sampling, control, and fractional calculus.

On N=65,536 problems, CHEAP delivers 400--1000x speedups over Cholesky/LAPACK baselines using O(N) memory.

## How It Works

Every algorithm in CHEAP reduces to the same three-step primitive:

```
output = iDCT( DCT(input) .* weights )
```

The choice of weight vector determines the operation:

| Operation | Weight `w[k]` | Complexity |
|-----------|---------------|------------|
| Kernel ridge regression | `1 / (lambda[k] + reg)` | O(N log N) |
| Gaussian reparameterization | `sqrt(lambda[k])` | O(N log N) |
| Sinkhorn optimal transport | `exp(-lambda[k] / eps)` | O(N log N) / iter |
| LQR / Tikhonov control | `1 / (lambda[k] + R)` | O(N log N) |
| Fractional differentiation | `(2 sin(w_k/2))^d` | O(N log N) |
| Toeplitz matvec / solve | `DCT(first_col)[k]` | O(N log N) |
| Kalman prediction | `P[k] + lambda[k]` | O(N) |

The eigenvalues `lambda[k]` come from Flandrin's wavelet variance formula for fractional Brownian motion, precomputed once during `cheap_init`. See [CHEAP.md](CHEAP.md) for the full mathematical treatment.

## Quick Start

### C API

```c
#include "cheap.h"

int main(void) {
    cheap_ctx ctx;
    cheap_init(&ctx, 4096, 0.7);  /* N=4096, Hurst exponent H=0.7 */

    /* Kernel ridge regression: solve (K + lambda*I)^{-1} y in O(N log N) */
    double w[4096], y[4096], alpha[4096];
    for (int k = 0; k < 4096; ++k)
        w[k] = 1.0 / (ctx.lambda[k] + 1e-3);

    cheap_apply(&ctx, y, w, alpha);  /* done. */

    cheap_destroy(&ctx);
}
```

### C++ API

```cpp
#include "cheap.hpp"

int main() {
    cheap::Context ctx(4096, 0.7);

    std::vector<double> y(4096), weights(4096);
    for (int k = 0; k < 4096; ++k)
        weights[k] = 1.0 / (ctx.lambda()[k] + 1e-3);

    auto alpha = ctx.apply(y.data(), weights.data());  /* returns std::vector */
}
```

## Examples

### Entropic Optimal Transport (Sinkhorn)

Transport mass between two distributions in O(N log N) per iteration, with max-log stabilization for numerical safety:

```c
cheap_ctx ctx;
cheap_init(&ctx, 1024, 0.7);

double a[1024], b[1024];  /* source and target marginals (equal mass) */
double f[1024], g[1024];  /* dual potentials (output) */

int rc = cheap_sinkhorn(&ctx, a, b,
                        /*eps=*/0.01, /*max_iter=*/200, /*tol=*/1e-8,
                        f, g);
/* rc == CHEAP_OK on convergence */

cheap_destroy(&ctx);
```

### Gaussian Process Sampling

Sample from a GP posterior with fBm covariance in O(N log N) — no Cholesky required:

```c
cheap_ctx ctx;
cheap_init(&ctx, 8192, 0.7);

double eps[8192], sample[8192];
/* fill eps with standard normal draws */

cheap_apply(&ctx, eps, ctx.sqrt_lambda, sample);
/* sample ~ N(0, K) where K is the fBm covariance */

cheap_destroy(&ctx);
```

### Toeplitz System Solve

Solve a regularised Toeplitz system `(T + lambda*I)x = y` for any symmetric Toeplitz matrix:

```c
cheap_ctx ctx;
cheap_init(&ctx, 2048, 0.5);

double t[2048];   /* first column of Toeplitz matrix */
double lam[2048]; /* eigenvalues (output) */
cheap_toeplitz_eigenvalues(&ctx, t, lam);

double y[2048], x[2048];
cheap_toeplitz_solve_precomp(&ctx, lam, y, /*reg=*/1e-4, x);
/* x = (T + 1e-4 * I)^{-1} y */

cheap_destroy(&ctx);
```

More examples in [`examples/`](examples/): GP regression, optimal transport, LQR/MPC control, fractional calculus, Poisson solver, Navier-Stokes dissipation, online kernel filtering.

## API Overview

The API has three layers:

**1. Context lifecycle** — allocate once, reuse for all operations:
- `cheap_init(ctx, n, H)` / `cheap_destroy(ctx)`

**2. Core spectral primitives** — the building blocks:
- `cheap_forward` / `cheap_inverse` — raw DCT-II and iDCT-III
- `cheap_apply` — the universal `DCT -> multiply -> iDCT` primitive

**3. Algorithms** — built on the primitives:
- `cheap_sinkhorn` — entropic optimal transport
- `cheap_toeplitz_eigenvalues` / `cheap_toeplitz_solve_precomp` — generic Toeplitz operations
- `cheap_rff_init` / `cheap_rff_map` — random Fourier features for kernel approximation

The C++ wrapper (`cheap.hpp`) provides RAII via `cheap::Context` and `cheap::RffContext`, with exception-based error handling and optional `std::span` overloads in C++20.

Full documentation: [API.md](API.md)

## Building

### Header-only (no build system)

```bash
gcc -std=c99 -O3 -march=native myapp.c -o myapp -lfftw3 -lm
```

### CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
ctest -V    # run tests
```

Build types: `Release` (`-O3`), `Debug` (`-O0 -g`), `Asan` (`-fsanitize=address,undefined`).

C++ targets are built by default. Disable with `-DCHEAP_BUILD_CPP=OFF`.

### Dependencies

- **FFTW3** — the only external dependency
  - Debian/Ubuntu: `sudo apt install libfftw3-dev`
  - macOS: `brew install fftw`
  - CMake will auto-download FFTW 3.3.10 if not found locally

### Platform Support

| Platform | Status |
|----------|--------|
| x86 / x86_64 | Supported (auto-vectorised with `-march=native`) |
| ARM64 / AArch64 | Supported (primary dev target, auto-vectorised) |
| Any C99 target | Portable (no platform-specific code in hot paths) |

## Benchmarks

### ARM64 (ARMv8 rev 1 @ ~2.0 GHz, GCC 11.4.0, `-O3 -march=native`)

| Algorithm | N=1,024 | N=8,192 | N=65,536 | Scaling |
|-----------|---------|---------|----------|---------|
| KRR solve | 0.018 ms | 0.180 ms | 2.54 ms | O(N log N) |
| Sinkhorn (50 iters) | 0.087 ms | 0.839 ms | 8.77 ms | O(N log N) |
| Reparameterization | 0.018 ms | 0.181 ms | 2.45 ms | O(N log N) |
| Toeplitz matvec | 0.018 ms | 0.180 ms | 2.54 ms | O(N log N) |
| RFF map (D=256) | 0.004 ms | — | — | O(D) |

### Statistical Benchmarks (30 trials, FFTW_PATIENT)

| Algorithm | N | Mean (ms) | CV% | Status |
|-----------|---|-----------|-----|--------|
| krr_solve | 65,536 | 5.279 | 0.08% | STABLE |
| reparam | 65,536 | 5.363 | 0.11% | STABLE |
| sinkhorn_50 | 65,536 | 16.613 | 0.46% | STABLE |
| toeplitz_matvec | 65,536 | 4.924 | 0.06% | STABLE |

All CV < 0.5%, 95% CI within 0.2% of mean.

Run benchmarks:
```bash
./build/bench_cheap          # quick benchmarks
./build/bench_cheap_stats    # statistical benchmarks (30 trials)
```

### Complexity vs. Standard Methods

| Problem | Standard | CHEAP | Memory |
|---------|----------|-------|--------|
| Kernel regression | O(N^3) | O(N log N) | O(N) |
| Toeplitz solve | O(N^3) | O(N log N) | O(N) |
| GP sampling | O(N^3) | O(N log N) | O(N) |
| KLMS update | O(n) growing | O(D) fixed | O(D) |

## File Structure

```
include/
  cheap.h        — C99 header (~460 lines, entire implementation)
  cheap.hpp      — C++17 RAII wrapper
tests/
  test_cheap.c   — C test suite
  test_cheap_cpp.cpp
benchmarks/
  bench_cheap.c  — quick benchmarks
  bench_cheap_cpp.cpp
examples/
  c/             — C99 examples
    gp_regression.c, optimal_transport.c, lqr_mpc.c,
    fractional_diff.c, poisson_solver.c, ns_dissipation.c,
    toeplitz_solve.c, online_filter.c
  cpp/           — C++17 equivalents
    gp_regression.cpp, optimal_transport.cpp, lqr_mpc.cpp,
    fractional_diff.cpp, poisson_solver.cpp, ns_dissipation.cpp,
    toeplitz_solve.cpp, online_filter.cpp
```

## When to Use CHEAP

**CHEAP excels at:** 1D regular grids, stationary kernels (Matern, RBF, fBm), repeated solves with the same (N, H), large N (tested to N > 10^6).

**CHEAP is not for:** irregular/scattered data, non-stationary kernels, high-dimensional grids (use tensor products or RFF instead).

See [CHEAP.md](CHEAP.md) for the full mathematical framework, proofs, and an honest discussion of the accuracy-efficiency Pareto frontier.

## License

[MIT](LICENSE)
