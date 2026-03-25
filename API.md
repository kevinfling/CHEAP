# CHEAP API Reference

Complete documentation for the CHEAP C99 and C++17 APIs.

## C API (`cheap.h`)

### Constants

#### Error Codes

| Code | Value | Meaning |
|------|-------|---------|
| `CHEAP_OK` | 0 | Success |
| `CHEAP_EINVAL` | -1 | Invalid argument (null pointer, out-of-range H, etc.) |
| `CHEAP_ENOMEM` | -2 | Allocation failed |
| `CHEAP_ENOCONV` | -3 | Sinkhorn did not converge within `max_iter` |
| `CHEAP_EDOM` | -4 | NaN or Inf detected in input data |
| `CHEAP_EUNINIT` | -5 | Context not initialized |

#### Numerical Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `CHEAP_EPS_LOG` | `1e-12` | Floor for `sin`/`log` arguments |
| `CHEAP_EPS_DIV` | `1e-300` | Floor for divisors to prevent division by zero |
| `CHEAP_EPS_LAMBDA` | `1e-15` | DC eigenvalue regularization |

#### Version

```c
CHEAP_VERSION_MAJOR   // 0
CHEAP_VERSION_MINOR   // 1
CHEAP_VERSION_PATCH   // 0
CHEAP_VERSION         // "0.1.0"
```

---

### Types

#### `cheap_ctx`

Main context holding FFTW plans, precomputed eigenvalues, and workspace buffers.

```c
typedef struct {
    int n;                        // problem size
    int is_initialized;           // 1 after successful cheap_init
    double* restrict lambda;      // Flandrin eigenvalues [n]
    double* restrict gibbs;       // Gibbs weights exp(-lambda/eps) [n]
    double* restrict sqrt_lambda; // sqrt(lambda) [n]
    double* restrict workspace;   // FFTW in-place buffer [n]
    fftw_plan plan_fwd;           // DCT-II plan (FFTW_REDFT10)
    fftw_plan plan_inv;           // DCT-III plan (FFTW_REDFT01)
    double current_eps;           // cached epsilon for Gibbs weights
    double current_H;             // Hurst exponent
} cheap_ctx;
```

#### `cheap_rff_ctx`

Context for Random Fourier Features.

```c
typedef struct {
    int D;               // feature dimension (must be even, >= 2)
    int d_in;            // input dimension
    double sigma;        // kernel bandwidth
    double* omega;       // (D/2) * d_in frequencies, row-major
    double* bias;        // D/2 biases in [0, 2*pi)
    double scale;        // sqrt(2.0 / D)
    int is_initialized;
} cheap_rff_ctx;
```

---

### Context Lifecycle

#### `cheap_init`

```c
int cheap_init(cheap_ctx* ctx, int n, double H);
```

Allocate buffers, precompute Flandrin eigenvalues, and create `FFTW_PATIENT` plans.

**Parameters:**
- `ctx` — pointer to uninitialized context
- `n` — problem size (must be >= 2)
- `H` — Hurst exponent, strictly in (0, 1)

**Returns:** `CHEAP_OK` on success, `CHEAP_EINVAL` or `CHEAP_ENOMEM` on failure.

**Notes:**
- First call is slow due to `FFTW_PATIENT` planning. Persist FFTW wisdom for faster startup.
- Call once per (n, H) pair; reuse the context for all subsequent operations.

#### `cheap_destroy`

```c
void cheap_destroy(cheap_ctx* ctx);
```

Free all FFTW plans and buffers. Safe to call on a zeroed or already-destroyed context.

---

### Core Spectral Primitives

#### `cheap_forward`

```c
int cheap_forward(cheap_ctx* ctx, const double* input);
```

Compute DCT-II of `input` into `ctx->workspace`. After this call, `ctx->workspace` contains the spectral coefficients.

**Parameters:**
- `ctx` — initialized context
- `input` — array of `n` doubles (validated with `isfinite`)

**Returns:** `CHEAP_OK`, `CHEAP_EUNINIT`, `CHEAP_EINVAL`, or `CHEAP_EDOM`.

#### `cheap_inverse`

```c
int cheap_inverse(cheap_ctx* ctx, double* output);
```

Compute iDCT-III of `ctx->workspace` into `output`, applying `1/(2N)` normalization.

**Parameters:**
- `ctx` — initialized context (workspace must contain spectral data)
- `output` — array of `n` doubles to write result

**Returns:** `CHEAP_OK`, `CHEAP_EUNINIT`, or `CHEAP_EINVAL`.

#### `cheap_apply`

```c
int cheap_apply(cheap_ctx* ctx, const double* input,
                const double* weights, double* output);
```

The universal spectral primitive: `output = iDCT(DCT(input) * weights) / (2N)`.

Every spectral algorithm in CHEAP reduces to this operation with an appropriate weight vector:

| Algorithm | Weight `weights[k]` |
|-----------|---------------------|
| KRR solve | `1 / (lambda[k] + lambda_reg)` |
| Reparameterization | `sqrt(lambda[k])` |
| LQR / Tikhonov | `1 / (lambda[k] + R)` |
| Fractional diff | `(2 * sin(omega_k / 2))^d` |
| Toeplitz matvec | `DCT(first_column)[k]` |

**Parameters:**
- `ctx` — initialized context
- `input` — array of `n` doubles
- `weights` — array of `n` spectral weights
- `output` — array of `n` doubles for result (may alias `input`)

**Returns:** `CHEAP_OK`, `CHEAP_EUNINIT`, `CHEAP_EINVAL`, or `CHEAP_EDOM`.

---

### Sinkhorn Optimal Transport

#### `cheap_recompute_gibbs`

```c
int cheap_recompute_gibbs(cheap_ctx* ctx, double eps);
```

Recompute the Gibbs weights `exp(-lambda[k] / eps)` and cache in `ctx->gibbs`. Only recomputes if `eps` has changed since the last call.

**Parameters:**
- `ctx` — initialized context
- `eps` — regularization strength (must be > 0)

#### `cheap_apply_hybrid_log`

```c
void cheap_apply_hybrid_log(cheap_ctx* ctx, const double* f, double* out);
```

Apply the Gibbs kernel in log domain with max-log stabilization to prevent underflow. Computes `out[i] = log(sum_j K[i,j] * exp(f[j])) ` using the precomputed `ctx->gibbs` weights.

#### `cheap_sinkhorn`

```c
int cheap_sinkhorn(cheap_ctx* ctx, const double* a, const double* b,
                   double eps, int max_iter, double tol,
                   double* f, double* g);
```

Solve the entropic optimal transport problem between marginals `a` and `b`.

**Parameters:**
- `ctx` — initialized context
- `a`, `b` — source and target marginals (must sum to equal mass within `1e-8` relative tolerance)
- `eps` — entropic regularization (must be > 0)
- `max_iter` — maximum Sinkhorn iterations
- `tol` — convergence tolerance on dual potentials
- `f`, `g` — output dual potentials (arrays of `n` doubles)

**Returns:** `CHEAP_OK` on convergence, `CHEAP_ENOCONV` if `max_iter` reached, or other error codes.

---

### Toeplitz Utilities

These functions reuse `cheap_ctx` for FFTW plans and workspace but ignore `ctx->lambda` — the caller supplies the Toeplitz structure directly.

#### `cheap_toeplitz_eigenvalues`

```c
int cheap_toeplitz_eigenvalues(cheap_ctx* ctx, const double* t,
                               double* lambda_out);
```

Compute eigenvalues of a symmetric Toeplitz matrix by taking the DCT-II of its first column `t`.

#### `cheap_toeplitz_solve_precomp`

```c
int cheap_toeplitz_solve_precomp(cheap_ctx* ctx, const double* lambda_t,
                                 const double* y, double lambda_reg,
                                 double* x);
```

Solve `(T + lambda_reg * I) x = y` given precomputed eigenvalues `lambda_t`.

**Parameters:**
- `ctx` — initialized context
- `lambda_t` — precomputed eigenvalues from `cheap_toeplitz_eigenvalues`
- `y` — right-hand side
- `lambda_reg` — Tikhonov regularization (must be >= 0)
- `x` — solution output

---

### Random Fourier Features (RFF)

Explicit feature maps for Gaussian kernel approximation via Bochner's theorem: `k(x,y) = exp(-||x-y||^2 / (2*sigma^2)) ~ z(x)^T z(y)`.

#### `cheap_rff_init`

```c
int cheap_rff_init(cheap_rff_ctx* rctx, int D, int d_in,
                   double sigma, uint64_t seed);
```

Initialize RFF context by sampling `D/2` random frequencies from the spectral density.

**Parameters:**
- `rctx` — pointer to uninitialized RFF context
- `D` — feature dimension (must be even, >= 2)
- `d_in` — input dimension (>= 1)
- `sigma` — kernel bandwidth (> 0)
- `seed` — PRNG seed for reproducibility

#### `cheap_rff_destroy`

```c
void cheap_rff_destroy(cheap_rff_ctx* rctx);
```

Free RFF context. Safe to call on a zeroed or already-destroyed context.

#### `cheap_rff_map`

```c
int cheap_rff_map(const cheap_rff_ctx* rctx, const double* x_in,
                  double* z_out);
```

Map a single input vector `x_in` (length `d_in`) to feature space `z_out` (length `D`).

#### `cheap_rff_map_batch`

```c
int cheap_rff_map_batch(const cheap_rff_ctx* rctx, const double* X_in,
                        int N, double* Z_out);
```

Map `N` input vectors (row-major, each of length `d_in`) to feature space (row-major, each of length `D`).

---

### Utility

#### `cheap_rdtsc`

```c
uint64_t cheap_rdtsc(void);
```

Read the hardware cycle counter. Supports x86 (`rdtsc`) and ARM64 (`cntvct_el0`). Returns 0 on unsupported architectures.

---

## C++ API (`cheap.hpp`)

Header-only C++17 wrapper. RAII semantics, exception-based error handling, optional `std::span` overloads (C++20).

### `cheap::Context`

RAII wrapper for `cheap_ctx`.

```cpp
// Construction allocates and plans; destruction cleans up.
cheap::Context ctx(1024, 0.7);

// Throwing API
ctx.forward(input);
ctx.inverse(output);
ctx.apply(input, weights, output);
std::vector<double> result = ctx.apply(input, weights);

// Non-throwing API (returns error code)
int rc = ctx.try_forward(input);
int rc = ctx.try_apply(input, weights, output);

// Sinkhorn
ctx.sinkhorn(a, b, eps, max_iter, tol, f, g);

// Toeplitz
ctx.toeplitz_eigenvalues(t, lambda_out);
std::vector<double> lam = ctx.toeplitz_eigenvalues(t);
ctx.toeplitz_solve_precomp(lambda_t, y, lambda_reg, x);
std::vector<double> sol = ctx.toeplitz_solve_precomp(lambda_t, y, lambda_reg);

// Accessors
int n = ctx.n();
double H = ctx.H();
const double* lam = ctx.lambda();
const double* sq = ctx.sqrt_lambda();
```

Move-only (no copy). C++20 builds additionally get `std::span` overloads with bounds checking.

### `cheap::RffContext`

RAII wrapper for `cheap_rff_ctx`.

```cpp
cheap::RffContext rff(256, 3, 1.0, 42);

// Map single vector
rff.map(x_in, z_out);
std::vector<double> z = rff.map(x_in);

// Batch
rff.map_batch(X_in, N, Z_out);
std::vector<double> Z = rff.map_batch(X_in, N);

// Accessors
int D = rff.D();
int d = rff.d_in();
```

### `cheap::Error`

Exception thrown on non-OK return codes. Inherits `std::runtime_error`.

```cpp
try {
    cheap::Context ctx(0, 0.5);  // n < 2
} catch (const cheap::Error& e) {
    // e.code() == cheap::ErrorCode::einval
    // e.what() == "cheap: invalid argument"
}
```

### `cheap::ErrorCode`

Enum class mirroring the C error codes: `einval`, `enomem`, `enoconv`, `edom`, `euninit`.
