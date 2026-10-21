# CHEAP API and numerical contracts

[Overview and build](README.md) · [Evidence and reproduction](BENCH.md)

This guide covers the implemented C99 headers and C++17 wrapper on this branch. Definitions in `include/` give complete parameter lists; links below identify the owning header. Legacy names retain their numerical behavior, with interpretation limits collected at the end.

## The spectral operator

Use the orthonormal DCT-II convention:

```text
Q[k,j] = sqrt((2 - delta[k,0])/N) * cos(pi*k*(j+1/2)/N)
B(w)   = Q^T diag(w) Q
```

`cheap_apply` evaluates `B(w) input`. The backend uses unnormalized DCT-II (`FFTW_REDFT10`) and DCT-III (`FFTW_REDFT01`); inverse normalization makes their pair equivalent to this orthonormal multiplier. Raw `cheap_forward` coefficients are not orthonormal coefficients.

Every real B(w) is symmetric, PSD if all weights are nonnegative, and SPD if all are strictly positive. Those statements describe B; they do not establish accuracy or definiteness of an original matrix A.

For symmetric A, the Frobenius-optimal DCT approximation has `w=diag(Q A Q^T)`. Its squared Frobenius error equals the sum of squares of the off-diagonal entries of `Q A Q^T`. This is an orthogonal projection, not an eigendecomposition of A.

The unit-spacing, cell-centered Neumann second-difference matrix has diagonal `[1,2,...,2,1]`, nearest neighbors -1, and spectrum `mu[k]=4*sin(pi*k/(2N))^2`. This operator is exactly DCT-diagonal. In 2D/3D its spectrum is the **sum** of per-axis spectra. For spacing h scale a 1D spectrum by `1/h²`; scale each axis separately for unequal spacings. L² has the same constant nullspace as L; it is not an arbitrary fourth-derivative boundary-value operator.

## Lifecycle and status codes

Zero-initialize C contexts/workspaces, initialize once, reuse, and destroy before reinitializing. Do not copy live owning structs by value. Destruction of zeroed/destroyed contexts is supported. Failure semantics vary in legacy initialization code; always check return codes before use.

One context owns mutable scratch and cannot execute concurrently with itself. Independent contexts have separate execution scratch. Plan creation/destruction needs the backend's synchronization rules; independent execution does not imply concurrent planning is safe.

| Code | Value | Meaning |
| --- | ---: | --- |
| `CHEAP_OK` | 0 | Success |
| `CHEAP_EINVAL` | -1 | Invalid argument or unsupported size/overlap |
| `CHEAP_ENOMEM` | -2 | Allocation or plan creation failed |
| `CHEAP_ENOCONV` | -3 | Iteration budget exhausted in scaling/transport |
| `CHEAP_EDOM` | -4 | Invalid numerical data/domain, including rejected spectrum |
| `CHEAP_EUNINIT` | -5 | Unusable/uninitialized context |

PCG returns its own result/status enum. The standalone O(N²) projection reference returns 0 or -1 rather than the full status family.

Constants in `cheap.h`: `CHEAP_EPS_LOG=1e-12`, `CHEAP_EPS_DIV=1e-300`, `CHEAP_EPS_LAMBDA=1e-15`. These are implementation floors, not accuracy guarantees. Header version: `CHEAP_VERSION="0.3.0-tensor-weights"` (numeric components 0, 3, 0); CMake still declares 0.2.0.

Finite/domain validation is API-specific. `cheap_apply` checks input but does not sweep all weights or check every output; raw in-place primitives skip input checks. Some legacy scalar checks do not reject NaN. Supply finite, correctly sized, valid data even when a function can return `CHEAP_EDOM`. `CHEAP_DEBUG_CONTRACTS` adds checks, and fast-math can undermine finite-value guards.

## DCT contexts and primitives

Defined in [`include/cheap.h`](include/cheap.h).

| Context | Shape/layout | Inverse normalization |
| --- | --- | --- |
| `cheap_ctx` | `n` | `1/(2*n)` |
| `cheap_ctx_2d` | flat `nx*ny`, index `j*ny+k` | `1/(4*nx*ny)` |
| `cheap_ctx_3d` | flat `nx*ny*nz`, index `(j*ny+k)*nz+l` | `1/(8*nx*ny*nz)` |

Each axis must have at least 2 entries. `cheap_init(ctx,n,H)` and `cheap_init_2d/3d` require finite H strictly in (0,1). They allocate aligned `lambda`, `sqrt_lambda`, `gibbs` and `workspace` arrays and create `FFTW_PATIENT` DCT plans. Gibbs weights are cached when epsilon changes.

For k>0, the default 1D model is `N^(2H)*sin(pi*k/(2N))^(-(2H+1))`. DC is extrapolated from the first two positive modes (or doubled from mode 1 at N=2). This is a Flandrin-inspired model, not the exact finite-grid fBm spectrum. ND defaults multiply axis spectra; flat ordering need not be monotone. Laplacian spectra instead add across axes.

`cheap_init_from_eigenvalues_2d/3d` accepts a flat caller spectrum. Legacy `cheap_init_from_toeplitz` and ND variants use DCTs of first columns; consult migration notes. A Frobenius projection constructor exists in 1D.

| Primitive | Effect |
| --- | --- |
| `cheap_forward(ctx,input)` | Validates/copies input and executes raw DCT-II into workspace |
| `cheap_inverse(ctx,output)` | Executes DCT-III on workspace and copies normalized output; workspace is overwritten |
| `cheap_apply(ctx,input,weights,output)` | Forward transform, pointwise multiply, normalized inverse |
| `cheap_forward_inplace`, `cheap_inverse_inplace`, `cheap_apply_inplace` | Operate on `ctx->workspace`; caller populates and validates it |
| `_2d` / `_3d` forms | Same over flat ND arrays, e.g. `cheap_apply_2d`, `cheap_forward_inplace_3d` |
| `cheap_workspace_2d/3d` | Access ND workspace; 1D exposes `ctx->workspace` directly |

Use distinct input, weight and output buffers for ordinary apply calls and keep weights separate from workspace. These functions use `restrict`; use explicit workspace in-place interfaces when overlap is needed. No general partial-overlap guarantee is made for legacy DCT functions.

Choose reciprocal weights for a model solve and square-root covariance weights for a model Gaussian draw. `cheap_apply` does not use `ctx->lambda` unless the caller constructs weights from it. Application names such as KRR, LQR/MPC and dissipation in examples describe compositions; they do not provide general control or inference solvers.

## Weight constructors

Weight construction is O(N), without FFT planning. Applying the result is a separate transform operation. `_ev` accepts flat supplied spectra; check signatures because argument order differs between families. Arrays must match context shape.

Prefer explicit constructors in [`include/cheap_weights_verified.h`](include/cheap_weights_verified.h) for new covariance/filtering code:

| Constructor | Weight and contract |
| --- | --- |
| `cheap_weights_matern_covariance_ev(n,mu,d,kappa,nu,amplitude,out)` | `amplitude*(kappa²+mu)^(-(nu+d/2))` |
| `cheap_weights_matern_sampling_ev(n,mu,d,kappa,nu,amplitude,out)` | Square root of the preceding covariance spectrum |
| `cheap_weights_anisotropic_matern_2d_verified` | Positive mass `kappa²`, separate positive directional coefficients, exponent `-(nu+1)` |
| `cheap_weights_resolvent_ev(n,mu,shift,out)` | `1/(mu+shift)`, nonnegative mu and positive shift |
| `cheap_weights_wave_cos_ev(n,mu,t,out)` | `cos(t*sqrt(mu))`, nonnegative mu; displacement propagator with zero initial velocity |
| `cheap_weights_wiener_signal_ev(n,signal_power,noise_power,out)` | `S/(S+N)`, nonnegative PSDs and nonzero denominator |

Matérn constructors accept dimensions 1–3, positive kappa/nu, and nonnegative amplitude. They define a discrete spectral SPDE model; spacing, boundaries, continuum equivalence and marginal variance calibration remain caller responsibilities. Apply the **sampling** weights to independent unit-variance spatial white noise. Orthogonal transformation preserves white noise covariance, yielding `Q^T diag(s) Q`.

For example, construct unit-spacing 2D sampling weights with:

```c
/* ctx initialized for NX by NY; noise contains independent N(0,1) draws. */
int rc = cheap_weights_laplacian_2d(NX, NY, mu);
if (rc == CHEAP_OK)
    rc = cheap_weights_matern_sampling_ev(NX * NY, mu, 2,
                                         1.0, 1.5, 1.0, weights);
if (rc == CHEAP_OK)
    rc = cheap_apply_2d(&ctx, noise, weights, sample);
```

The main header retains additional constructors:

| Family | Functions / multiplier | Interpretation |
| --- | --- | --- |
| Laplacian | `cheap_weights_laplacian`, `_ev`, `_2d`, `_3d` | Unit-spacing reflected-boundary spectrum mu |
| Fractional | `cheap_weights_fractional`: floored `(2*sin(pi*k/(2N)))^d`; `cheap_weights_fractional_laplacian_2d/3d`: floored `mu^alpha` | Regularized powers; DC differs from a true pseudoinverse |
| Heat | `cheap_weights_heat_propagator_ev/2d/3d`: `exp(-t*mu)` | Semigroup of the specified discrete operator, t>=0 |
| Poisson | `cheap_weights_poisson_ev/2d/3d`: DC=0; other modes `1/(mu+eps)` | For eps=0, Neumann pseudoinverse; RHS must be zero mean for an exact equation solve |
| Biharmonic | `cheap_weights_biharmonic_ev/2d/3d`: `1/(mu²+eps)` | Regularized L² inverse; DC=`1/eps` unless caller projects it out |
| Deconvolution | `cheap_weights_wiener_deconv_ev`: `psi/(psi²+noise_power)` | Assumes real transform-diagonal blur and the stated variance ratio |
| Higher-order Tikhonov | `cheap_weights_higher_order_tikhonov_deconv_ev`: `psi/(psi²+alpha*mu^p+eps)` | Penalty `mu^p` squares the action of `L^(p/2)` |
| Whitening | `cheap_weights_specnorm`, `_ev`: `1/sqrt(s+eps)` | Covariance whitening only if s is the intended covariance spectrum |

Complex blur filtering needs conjugation and `abs(psi)²`; the real API does not provide it. Wiener denoising requires a signal PSD and noise PSD. RMT, KPCA, Gamma-ratio and legacy Matérn names are qualified in the migration table.

## Original Toeplitz operator and DCT projection

[`include/cheap_toeplitz_operator.h`](include/cheap_toeplitz_operator.h) represents the original finite symmetric Toeplitz operator `A[i,j]=column[abs(i-j)]`, n>=1. A need not be SPD for multiplication.

```c
int cheap_toeplitz_operator_init(cheap_toeplitz_operator *op,
                                int n, const double *column);
int cheap_toeplitz_operator_apply(cheap_toeplitz_operator *op,
                                 const double *x, double *y);
void cheap_toeplitz_operator_destroy(cheap_toeplitz_operator *op);
cheap_linear_operator cheap_toeplitz_as_linear_operator(
    cheap_toeplitz_operator *op);
```

The embedding length is the smallest power of two L>=2n-1, with row `[c0,c1,...,c(n-1),zeros,c(n-1),...,c1]`. Cached FFT convolution and `1/L` inverse normalization return the first n outputs. Construction/application take O(L log L) arithmetic, with backend planning additional at construction. Apply does not allocate or replan.

Column and vectors must be finite and sizes representable. Exact `x==y` is supported; partial overlap is rejected. Failed apply leaves y unchanged. Initialize a zeroed/destroyed context; failed init leaves it empty. FFT buffers use `8L + 32*(floor(L/2)+1)` bytes, excluding plans and struct overhead.

For a **DCT approximation**, `cheap_init_from_toeplitz_projection(ctx,n,column)` in `cheap.h` computes `diag(Q A Q^T)` in O(n log n) arithmetic and initializes a DCT context. It rejects nonpositive/nonfinite projected weights; it does not certify A is SPD. For repeated column updates, use:

- `cheap_toeplitz_projection_workspace_init` / `_destroy`;
- `cheap_toeplitz_projection_with_workspace(ws,column,weights)`.

The workspace path requires nonoverlapping column/weights, reuses all scratch, and leaves weights unchanged on failed validation/positivity checks. [`cheap_toeplitz_dct_projection_diagonal`](include/cheap_toeplitz_dct_diagonal.h) is the separate O(n²), FFT-free reference formula; it does not enforce positive weights.

## PCG

[`include/cheap_pcg.h`](include/cheap_pcg.h) solves the original caller-supplied SPD operator. `cheap_linear_operator` contains `{n, context, apply}`, with callback type `int (*)(void *,const double *,double *)`. Return `CHEAP_OK` after computing output or a failure code. Context must be nonnull. Optional Minv applies an SPD inverse preconditioner; NULL selects CG.

```c
cheap_pcg_result cheap_pcg_solve_opts(
    const cheap_linear_operator *A, const cheap_linear_operator *Minv,
    const double *b, double *x, cheap_pcg_workspace *workspace,
    const cheap_pcg_options *options);
```

Initialize/destroy scratch with `cheap_pcg_workspace_init(w,n)` / `cheap_pcg_workspace_destroy(w)`. The solver itself does not allocate or plan. Keep b, x and scratch disjoint. Unless `zero_initial_guess` is set, x is the initial guess and must be initialized.

Options fields: `max_iterations`, `atol`, `rtol`, `residual_recompute_interval`, `zero_initial_guess`, `replacement_policy`. Acceptance requires a freshly evaluated:

```text
||b - A*x|| <= atol + rtol*||b||
```

The returned `reason` must equal `CHEAP_PCG_CONVERGED`. `residual_valid=1` means `true_residual` is a norm of a successfully evaluated original-system residual. A failed final A callback invalidates it and yields NaNs even if an earlier residual was known. `relative_residual` uses 0 for zero RHS by convention; use the absolute criterion then. Results also record iterations, operator/preconditioner calls, callback status and replacement policy.

`CHEAP_PCG_REFRESH_RESTART` discards the previous direction after replacement; this is restarted PCG. `CHEAP_PCG_REFRESH_CONTINUE` replaces the residual while retaining the direction and previous inner product; this finite-precision heuristic has no general conjugacy guarantee. Interval 0 disables periodic replacement but retains entry, convergence-candidate and termination checks. Legacy `cheap_pcg_solve` selects restart policy and uses the supplied initial guess.

Other reasons: `CHEAP_PCG_MAX_ITER`, `CHEAP_PCG_INVALID`, `CHEAP_PCG_BREAKDOWN_CURVATURE` (nonpositive `p^T A p`), `CHEAP_PCG_BREAKDOWN_NONFINITE`, `CHEAP_PCG_PRECOND_INVALID`, `CHEAP_PCG_CALLBACK_FAILED`. A valid exit residual does not itself mean convergence. Neither policy certifies arbitrary callbacks are SPD.

`cheap_pcg_dct_init(p,ctx,weights)` requires strictly positive finite weights and caches reciprocals. It borrows ctx and owns only the reciprocal array. Keep ctx alive until after `cheap_pcg_dct_destroy(p)`. `cheap_pcg_dct_as_operator(p)` exposes inverse apply as Minv; it never replaces A.

### Solve an original Toeplitz system

Save this complete example as `solve.c` and compile with the standalone command in README.md, substituting the filename. Initialization and the solver's original-system stopping status are checked.

```c
#include <stdio.h>
#include "cheap_pcg.h"

int main(void) {
    enum { N = 32 };
    cheap_toeplitz_operator toeplitz = {0};
    cheap_ctx dct = {0};
    cheap_pcg_dct_preconditioner pre = {0};
    cheap_pcg_workspace work = {0};
    double column[N], b[N], x[N] = {0};
    for (int i = 0; i < N; ++i) {
        column[i] = pow(0.9, i);
        b[i] = sin(0.1 * i) + 0.5;
    }
    int rc = cheap_toeplitz_operator_init(&toeplitz, N, column);
    if (rc == CHEAP_OK)
        rc = cheap_init_from_toeplitz_projection(&dct, N, column);
    if (rc == CHEAP_OK) rc = cheap_pcg_dct_init(&pre, &dct, dct.lambda);
    if (rc == CHEAP_OK) rc = cheap_pcg_workspace_init(&work, N);
    if (rc == CHEAP_OK) {
        cheap_linear_operator A = cheap_toeplitz_as_linear_operator(&toeplitz);
        cheap_linear_operator M = cheap_pcg_dct_as_operator(&pre);
        cheap_pcg_options opts = {200, 1e-10, 1e-8, 0, 1,
                                 CHEAP_PCG_REFRESH_RESTART};
        cheap_pcg_result result = cheap_pcg_solve_opts(&A, &M, b, x,
                                                      &work, &opts);
        rc = result.reason == CHEAP_PCG_CONVERGED && result.residual_valid
             ? CHEAP_OK : CHEAP_ENOCONV;
        if (rc == CHEAP_OK)
            printf("iterations=%d, true residual=%.9g\n",
                   result.iterations, result.true_residual);
    }
    cheap_pcg_workspace_destroy(&work);
    cheap_pcg_dct_destroy(&pre);
    cheap_destroy(&dct);
    cheap_toeplitz_operator_destroy(&toeplitz);
    return rc == CHEAP_OK ? 0 : 1;
}
```

## Circulant Gaussian sampling

[`include/cheap_circulant_sampler.h`](include/cheap_circulant_sampler.h) samples stationary **1D** covariance lags using the symmetric zero-padded embedding, separately from the DCT model sampler.

```c
int cheap_circulant_sampler_init(cheap_circulant_sampler *s,
    const double *lags, size_t n, size_t L, double tol, int backend);
int cheap_circulant_sampler_draw(cheap_circulant_sampler *s,
    const double *z, double *out);
void cheap_circulant_sampler_destroy(cheap_circulant_sampler *s);
```

L=0 selects the minimal power of two >=2n-1; explicit L must meet that bound and be a power of two. `backend` must be 0 (configured backend). Lags must be finite with nonnegative variance at lag 0.

With tol=0, any computed negative eigenvalue rejects initialization. With tol>0, negative eigenvalues within `tol*max(abs(min_eig),abs(max_eig))` are clipped to zero. This perturbs covariance by that spectral threshold; enum `CHEAP_EMBED_EXACT` includes this tolerance interpretation. Material negatives return `CHEAP_EDOM` and `info.status=CHEAP_EMBED_NOT_PSD`, retaining rejection diagnostics without allocated resources.

`cheap_circulant_sampler_init_approx_clip` permits material clipping and marks `CHEAP_EMBED_APPROX_CLIP`; such draws are approximate. `cheap_circulant_sampler_init_search` tries powers of two up to max_L, retaining the last rejection. **Zero-padding fixed lags to larger powers of two retains old Fourier frequencies and cannot repair an already negative mode.** Extending covariance lags would be a different construction, outside this API.

Supply exactly L independent N(0,1) values in z: DC at `z[0]`, interior real/imaginary components at `z[2*k-1]` / `z[2*k]`, Nyquist at `z[L-1]` for even L. Interior amplitudes use `sqrt(lambda/2)`, endpoints `sqrt(lambda)`. C2R output is divided by `sqrt(L)` and its first n values returned. Use separate z/output buffers. No RNG is owned; draw does not allocate or plan.

An accepted unmodified embedding yields the requested Toeplitz covariance up to floating-point error. Not every stationary covariance has a PSD embedding under this fixed-lag construction. fGn increments are stationary; fBm levels are nonstationary and are not directly represented by this API.

## Spectral diagnostics

[`include/cheap_spectral_diagnostics.h`](include/cheap_spectral_diagnostics.h) reports descriptive quantities in `cheap_spectral_diag`:

- `cheap_diag_from_weights`: finite min/max, negative/zero counts, condition estimate only for strictly positive weights. Undefined quantities are NaN.
- `cheap_diag_projection_error`: independent dense O(n³) relative Frobenius comparison of Toeplitz matrix and DCT surrogate, n<=256; not an SPD certificate for A.
- `cheap_diag_stochastic_probe`: seeded Rademacher RMS discrepancy `sqrt(sum||(A-B)v||² / sum||Av||²)`. It allocates temporary vectors and reports an empirical discrepancy, not an upper bound. Supply a nonzero seed and record probe count.

## Explicit-cost entropic transport

[`include/cheap_transport.h`](include/cheap_transport.h) provides:

```c
int cheap_transport_sinkhorn(int n, const double *a, const double *b,
    const double *cost, double epsilon, int max_iter, double tol,
    double *f, double *g, double *plan_out);
int cheap_transport_cost(int n, const double *cost, const double *plan,
                         double epsilon, double *value);
```

Supply finite row-major n*n costs, strictly positive finite marginals, equal mass within `1e-12` relative tolerance, epsilon>0, max_iter>=1 and tol>0. This is log-domain Sinkhorn for **entrywise** `exp(-cost[i,j]/epsilon)`. Each sweep costs O(n²). It resets f/g on entry; they are log scaling vectors, not epsilon-scaled dual potentials. Optional plan_out is filled on convergence.

Both row/column marginal errors must be <=`tol*max(1,total_mass)`. Objective: `<P,C> + epsilon*sum(P*(log(P)-1))`. Use representable n*n sizes and distinct input/scratch/output buffers; these dense reference helpers do not supply comprehensive large-size overflow guards.

[`cheap_sinkhorn_divergence`](include/cheap_sinkhorn_divergence.h) solves cross and two self problems with explicit cab/caa/cbb, returning `OT(a,b)-0.5*OT(a,a)-0.5*OT(b,b)`. Supply f[n], g[n], plan[n*n]. `cheap_sinkhorn_richardson` computes `2*S(epsilon/2)-S(epsilon)`; an accuracy-order claim needs a problem-specific error expansion.

[`cheap_gumbel_transport`](include/cheap_gumbel_transport.h) independently perturbs cost entries with seeded Gumbel draws, then solves the dense transport problem. Supply separate n*n cost scratch. It is a stochastic relaxation, not an exact matching sampler or Gaussian-copula field.

## RFF and C++ wrappers

`cheap_rff_init(rctx,D,d_in,sigma,seed)`, `cheap_rff_map`, `cheap_rff_map_batch`, `cheap_rff_destroy` in `cheap.h` implement seeded random Fourier features for a Gaussian kernel. D must be even and >=2, d_in>=1 and sigma finite/positive. Batch X/Z are row-major with widths d_in/D. Per-point work is O(D*d_in); this is separate from the DCT operator and does not solve a growing kernel system itself.

[`include/cheap.hpp`](include/cheap.hpp) supplies `cheap::Context`, `Context2D`, `Context3D`, `RffContext`. They own C contexts, destroy automatically and support moves rather than copying. Methods generally offer throwing and `try_*` status forms; `cheap::Error` derives from `std::runtime_error`, with `ErrorCode` mirroring C codes. C++20 adds checked `std::span` overloads unless `CHEAP_NO_SPAN` is defined. Raw pointers cannot check buffer lengths. The wrapper does not add RAII classes for companion PCG/sampler/transport headers.

```cpp
#include "cheap.hpp"

int main() {
    cheap::Context ctx(32, 0.5);
    std::vector<double> input(32, 0.0), weights(32, 1.0);
    input[0] = 1.0;
    auto output = ctx.apply(input.data(), weights.data());
    return output.size() == 32 ? 0 : 1;
}
```

`cheap_rdtsc()` reads x86 rdtsc, AArch64 cntvct_el0, or returns zero elsewhere. ARM generic timer ticks are not CPU cycles. Use wall-clock timings for comparisons.

## Legacy names and migration

These compatibility interfaces retain their numerical behavior. Some source comments and example labels still use older application claims.

| Legacy interface / label | Actual behavior and guidance |
| --- | --- |
| `cheap_toeplitz_eigenvalues`, `cheap_init_from_toeplitz` | DCT-II of first column, not generic Toeplitz eigenvalues. Prefer projection for a surrogate; exact FFT operator plus PCG for original-system solves. |
| `cheap_init_from_toeplitz_2d/3d` | Products of per-axis first-column DCTs; neither general BTTB/BTTTB diagonalization nor an original ND solver. |
| `cheap_toeplitz_solve_precomp` | Divides in DCT basis by supplied weights plus ridge; solves that surrogate. Check original-system residual separately. |
| `cheap_sinkhorn`, `cheap_recompute_gibbs`, `cheap_apply_hybrid_log` | Spectral operator exponential/log scaling heuristic; positive weights do not ensure entrywise-positive K. Use explicit-cost transport for dense EOT. Legacy convergence uses dual changes, not the dense solver's marginal contract. |
| `cheap_weights_matern_ev/2d/3d` | Exponent -nu, not conventional `-(nu+d/2)`. Use the corrected covariance constructor. |
| `cheap_sample_matern_2d/3d` | Applies legacy covariance weights to noise, squaring that spectrum in resulting covariance. Use corrected sampling weights and apply explicitly. |
| Legacy anisotropic Matérn | Axis coefficients/tiny DC floor rather than separate positive mass. Prefer verified 2D constructor; no verified 3D anisotropic helper exists yet. |
| `cheap_weights_wiener` | Laplacian `mu/(mu+sigma²)`: high-pass response unless mu intentionally denotes signal power. Use `cheap_weights_wiener_signal_ev` with actual PSDs. |
| `cheap_weights_specnorm` | Regularized inverse-square-root multiplier; not neural-network spectral norm normalization. |
| `cheap_weights_kpca_hard/soft` | Index cutoff/shrinkage. PCA needs a covariance eigenbasis and ordering; soft shrinkage is not a projector. |
| `cheap_weights_mandelbrot` | Complex-Gamma-ratio magnitude multiplier; filtering Gaussian noise remains Gaussian and does not establish multifractality. |
| `cheap_weights_rmt_hard/shrink` | User-spectrum formulas; require independently justified covariance/singular-value model, aspect ratio and loss. DCT indices are not sample covariance eigenvalues. |
| `experimental/timberwolf.h` | C11/CHEAPDAN prototype outside CMake/CTest; not a supported spectral API or validated application. |
