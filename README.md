# CHEAP

**Circulant Hessian Efficient Algorithm Package:** a header-only C99 library for O(N log N) spectral algorithms on structured covariance matrices. Supports 1D, 2D, and 3D grids via separable DCT diagonalization.

CHEAP turns cubic-time kernel methods into fast spectral operations by exploiting the fact that stationary covariance matrices (Toeplitz) diagonalize in the DCT-II basis. One forward DCT, one pointwise multiply, one inverse DCT: that's the entire algorithm for kernel regression, optimal transport, Gaussian sampling, control, and fractional calculus.

On N=65,536 problems, CHEAP delivers 400-1000x speedups over Cholesky/LAPACK baselines using O(N) memory.

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
| Wiener filter | `lk / (lk + sigma^2)` | O(N log N) |
| 2D/3D Laplacian | sum of `4sin²` terms | O(N log N) |
| 2D/3D fractional Laplacian | `(laplacian)^α` | O(N log N) |
| Spectral normalization | `1 / sqrt(lk + eps)` | O(N log N) |
| Kernel PCA (hard/soft) | `I(k < K)` / soft variant | O(N log N) |
| Mandelbrot multifractal | `abs(Gamma(H+it)/Gamma(1-H+it))` | O(N log N) |
| RMT denoising | Marchenko-Pastur threshold | O(N log N) |

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

### 2D Poisson solver (C)

```c
cheap_ctx_2d ctx;
cheap_init_2d(&ctx, 128, 128, 0.5, 0.5);

double f[128*128], phi[128*128], w[128*128];
/* ... fill RHS f ... */

cheap_weights_laplacian_2d(128, 128, w);
w[0] = 1.0;  /* DC regularization */

cheap_forward_2d(&ctx, f);
for (int i = 0; i < 128*128; ++i) ctx.workspace[i] /= w[i];
cheap_inverse_2d(&ctx, phi);  /* solves -Δφ = f */

cheap_destroy_2d(&ctx);
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

Solve a regularized Toeplitz system `(T + lambda*I)x = y` for any symmetric Toeplitz matrix:

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

More examples in [`examples/`](examples/): GP regression, optimal transport, LQR/MPC control, fractional calculus, Poisson solver (2D/3D), Navier-Stokes dissipation (2D/3D), online kernel filtering, Wiener denoising (2D), RMT denoising, GP regression (2D), Toeplitz solve (2D).

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

**4. Spectral weight constructors** — compute weight vectors for `cheap_apply`:
- `cheap_weights_fractional` — fractional integration/differentiation
- `cheap_weights_wiener` / `cheap_weights_wiener_ev` — Wiener filter (Laplacian or custom eigenvalues)
- `cheap_weights_specnorm` / `cheap_weights_specnorm_ev` — spectral normalization / whitening
- `cheap_weights_kpca_hard` / `cheap_weights_kpca_soft` — kernel PCA projection
- `cheap_weights_mandelbrot` — Mandelbrot multifractal weights (complex Gamma ratio)
- `cheap_weights_rmt_hard` / `cheap_weights_rmt_shrink` — RMT denoising (Marchenko-Pastur)
- `cheap_weights_laplacian` — discrete Laplacian eigenvalues

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
| x86 / x86_64 | Supported (auto-vectorized with `-march=native`) |
| ARM64 / AArch64 | Supported (primary dev target, auto-vectorized) |
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
  cheap.h        — C99 header (~2200 lines, entire implementation)
  cheap.hpp      — C++17 RAII wrapper (1D + 2D + 3D)
tests/
  test_cheap.c   — C test suite (1D)
  test_cheap_2d.c — 2D correctness + SIMD equivalence
  test_cheap_3d.c — 3D correctness
  test_cheap_cpp.cpp — C++ wrapper tests
benchmarks/
  bench_cheap.c  — quick benchmarks (1D + 2D + 3D)
  bench_cheap_cpp.cpp
examples/
  c/             — C99 examples
    gp_regression.c, optimal_transport.c, lqr_mpc.c,
    fractional_diff.c, poisson_solver.c, ns_dissipation.c,
    toeplitz_solve.c, online_filter.c, wiener_denoise.c,
    rmt_denoise.c,
    poisson_solver_2d.c, poisson_solver_3d.c,
    wiener_denoise_2d.c,
    ns_dissipation_2d.c, ns_dissipation_3d.c,
    gp_regression_2d.c, toeplitz_solve_2d.c
  cpp/           — C++17 equivalents
    (same names as c/ with .cpp extension)
```

## When to Use CHEAP

**CHEAP excels at:** 1D regular grids, stationary kernels (Matern, RBF, fBm), repeated solves with the same (N, H), large N (tested to N > 10^6).

**CHEAP is not for:** irregular/scattered data, non-stationary kernels, high-dimensional grids (use tensor products or RFF instead).

See [CHEAP.md](CHEAP.md) for the full mathematical framework, proofs, and an honest discussion of the accuracy-efficiency Pareto frontier.

## License

[MIT](LICENSE)
