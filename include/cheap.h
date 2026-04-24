#ifndef CHEAP_H
#define CHEAP_H

/*
 * CHEAP — Circulant Hessian Efficient Algorithm Package
 * Core primitives: DCT context, spectral apply, Sinkhorn, Toeplitz, RFF
 *
 * Header-only C99. Requires FFTW3.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Kevin Fling
 *
 * ----------------------------------------------------------------
 * v0.1.1-metal — hardened release
 * ----------------------------------------------------------------
 * Correctness:
 *   - λ[0] uses the Gupta-Joshi low-frequency asymptotic (no more
 *     CHEAP_EPS_LAMBDA wart); monotonicity of `lambda[]` is now a
 *     strict guarantee under default `cheap_init`.
 *   - `cheap_init_from_toeplitz` accepts a user-supplied first
 *     column and populates `lambda[]` exactly via DCT-II.
 *   - `cheap_sinkhorn` is allocation-free on its hot path
 *     (scratch buffers live in `cheap_ctx`).
 *   - `restrict` annotations on all non-aliasing pointer params.
 *   - Runtime contract monitors under `CHEAP_DEBUG_CONTRACTS`;
 *     zero cost in release.
 *
 * SIMD (compile-time dispatch — see CHEAP_SIMD_* below):
 *   - `cheap_apply`, `cheap_forward`, `cheap_inverse` use SIMD
 *     pointwise-mul / scale-copy helpers (AVX2: 4×f64, NEON: 2×f64).
 *   - `cheap_weights_wiener_ev`, `cheap_weights_specnorm_ev`,
 *     `cheap_weights_rmt_hard`, `cheap_weights_rmt_shrink`,
 *     `cheap_weights_wiener_deconv_ev` are fully vectorized.
 *   - `cheap_weights_mandelbrot` vectorizes the Re(lnΓ) Lanczos
 *     path on the non-reflection branch.
 *   - Scalar path always compiles for correctness cross-check.
 *
 * API additions (no breaking changes):
 *   - `cheap_init_from_toeplitz`
 *   - `cheap_apply_inplace`, `cheap_forward_inplace`,
 *     `cheap_inverse_inplace`
 *
 * Perf (ARM64 Tegra, scalar baseline → NEON, `cheap_apply`):
 *   ~162 cycles/el → target 18–25 cycles/el. AVX2 target: 8–12.
 *   See BENCH.md for full before/after tables.
 * ----------------------------------------------------------------
 * v0.2.0-tensor — 2D/3D extension
 * ----------------------------------------------------------------
 * New contexts:
 *   - `cheap_ctx_2d` / `cheap_ctx_3d` with tensor-product Flandrin spectra
 *   - `cheap_init_2d/3d`, `cheap_init_from_eigenvalues_2d/3d`,
 *     `cheap_init_from_toeplitz_2d/3d`
 *
 * Transforms (normalization: 1/(4·nx·ny) in 2D, 1/(8·nx·ny·nz) in 3D):
 *   - `cheap_forward_2d/3d`, `cheap_inverse_2d/3d`, `cheap_apply_2d/3d`
 *   - In-place variants: `cheap_*_inplace_2d/3d`
 *
 * 2D / 3D weight constructors (new):
 *   - `cheap_weights_laplacian_2d/3d`
 *   - `cheap_weights_fractional_laplacian_2d/3d`
 *
 * Dimension-agnostic (pass flat nx*ny or nx*ny*nz as `n`):
 *   - `_ev` family: `wiener_ev`, `specnorm_ev`
 *   - `kpca_hard`, `kpca_soft`, `rmt_hard`, `rmt_shrink`
 *
 * C++ wrapper: `cheap::Context2D`, `cheap::Context3D` with span overloads.
 * ----------------------------------------------------------------
 * v0.3.0-tensor-weights — GRF and PDE spectral weights
 * ----------------------------------------------------------------
 * New weight constructors (no ctx required, pure spectral math):
 *   - `cheap_weights_laplacian_ev`               — flat Laplacian (_ev alias)
 *   - `cheap_weights_matern_ev/2d/3d`            — Matérn covariance (GRF)
 *   - `cheap_weights_anisotropic_matern_2d/3d`   — anisotropic Matérn
 *   - `cheap_weights_heat_propagator_ev/2d/3d`   — heat equation propagator
 *   - `cheap_weights_biharmonic_ev/2d/3d`        — biharmonic inverse (SIMD)
 *   - `cheap_weights_poisson_ev/2d/3d`           — Poisson inverse (SIMD)
 *   - `cheap_weights_higher_order_tikhonov_deconv_ev` — HOT deconvolution
 *
 * SIMD: biharmonic_ev, poisson_ev and the second pass of their
 * _2d/_3d variants are AVX2/NEON vectorized. Matern, heat, and
 * HOT-deconv functions are scalar-only (pow/exp in hot path).
 * ----------------------------------------------------------------
 */

#define CHEAP_VERSION_MAJOR 0
#define CHEAP_VERSION_MINOR 3
#define CHEAP_VERSION_PATCH 0
#define CHEAP_VERSION "0.3.0-tensor-weights"

#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================
 * Error codes
 * ============================================================ */

#define CHEAP_OK           0
#define CHEAP_EINVAL      -1
#define CHEAP_ENOMEM      -2
#define CHEAP_ENOCONV     -3
#define CHEAP_EDOM        -4
#define CHEAP_EUNINIT     -5

/* ============================================================
 * Numerical constants
 * ============================================================ */

#define CHEAP_EPS_LOG     1e-12
#define CHEAP_EPS_DIV     1e-300
#define CHEAP_EPS_LAMBDA  1e-15

/* ============================================================
 * Alignment
 *
 * CHEAP hot buffers come from fftw_malloc. FFTW guarantees at
 * least SIMD_ALIGNMENT bytes — empirically 16 on AArch64 (Tegra
 * ARMv8 observed) and on non-AVX x86-64. This satisfies NEON
 * (`vld1q_f64` wants 16B). For AVX2 (`_mm256_load_pd` wants
 * 32B) we fall back to unaligned loads (`_mm256_loadu_pd`),
 * which modern Intel/AMD execute at the same rate as aligned
 * loads when the address happens to be 32-byte aligned.
 * AVX-512 is explicitly out of scope for v0.1.1.
 *
 * CHEAP_ASSUME_ALIGNED(p, a) — tell the compiler the pointer is
 * a-byte aligned so it can emit aligned loads/stores directly.
 * No-op on compilers that don't support the builtin.
 *
 * CHEAP_ALIGNMENT is the minimum guaranteed alignment for hot
 * buffers; used only for debug-contract assertions.
 * ============================================================ */

#define CHEAP_ALIGNMENT 16

#if defined(__GNUC__) || defined(__clang__)
#  define CHEAP_ASSUME_ALIGNED(p, a) ((__typeof__(p))__builtin_assume_aligned((p), (a)))
#else
#  define CHEAP_ASSUME_ALIGNED(p, a) (p)
#endif

#ifdef CHEAP_DEBUG_CONTRACTS
#  include <assert.h>
#  define CHEAP_ASSERT_ALIGNED(p) \
    assert(((uintptr_t)(p) & (CHEAP_ALIGNMENT - 1)) == 0)
#else
#  define CHEAP_ASSERT_ALIGNED(p) ((void)0)
#endif

/* ============================================================
 * Runtime contract monitors (CHEAP_DEBUG_CONTRACTS)
 *
 * Verifiable math invariants that fire in debug builds only.
 * Zero cost in release. Library code: never traps — returns
 * CHEAP_EDOM up the stack so callers can recover.
 * ============================================================ */
#ifdef CHEAP_DEBUG_CONTRACTS
#  define CHEAP_CONTRACT_FINITE_OR_EDOM(arr, len) \
    do { for (int _i = 0; _i < (len); ++_i) \
         if (!isfinite((arr)[_i])) return CHEAP_EDOM; } while (0)
#  define CHEAP_CONTRACT_STRICT_DEC_OR_EDOM(arr, len) \
    do { for (int _i = 1; _i < (len); ++_i) \
         if ((arr)[_i] >= (arr)[_i-1]) return CHEAP_EDOM; } while (0)
#  define CHEAP_CONTRACT_NONDEC_OR_EDOM(arr, len) \
    do { for (int _i = 1; _i < (len); ++_i) \
         if ((arr)[_i] < (arr)[_i-1]) return CHEAP_EDOM; } while (0)
#  define CHEAP_CONTRACT_NEAR_OR_EDOM(val, target, tol) \
    do { if (fabs((val) - (target)) > (tol)) return CHEAP_EDOM; } while (0)
#else
#  define CHEAP_CONTRACT_FINITE_OR_EDOM(arr, len)         ((void)0)
#  define CHEAP_CONTRACT_STRICT_DEC_OR_EDOM(arr, len)     ((void)0)
#  define CHEAP_CONTRACT_NONDEC_OR_EDOM(arr, len)         ((void)0)
#  define CHEAP_CONTRACT_NEAR_OR_EDOM(val, target, tol)   ((void)0)
#endif

/* ============================================================
 * SIMD dispatch (compile-time)
 *
 * We pick at most one SIMD ISA per build based on predefined
 * compiler macros. Callers can override by defining
 * CHEAP_SIMD_DISABLE to force the scalar path (useful in tests
 * that cross-check scalar vs vector results).
 * ============================================================ */
#if defined(CHEAP_SIMD_DISABLE)
#  define CHEAP_SIMD_SCALAR 1
#elif defined(__AVX2__)
#  define CHEAP_SIMD_AVX2 1
#  include <immintrin.h>
#elif defined(__ARM_NEON) && defined(__aarch64__)
#  define CHEAP_SIMD_NEON 1
#  include <arm_neon.h>
#else
#  define CHEAP_SIMD_SCALAR 1
#endif

/* ============================================================
 * Portable tick counter
 * ============================================================ */

static inline uint64_t cheap_rdtsc(void)
{
#if defined(__x86_64__) || defined(__i386__)
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#elif defined(__aarch64__)
    uint64_t v;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    return 0;
#endif
}

/* ============================================================
 * Core context and lifecycle
 * ============================================================ */

typedef struct {
    int n;
    int is_initialized;
    double* restrict lambda;
    double* restrict gibbs;
    double* restrict sqrt_lambda;
    double* restrict workspace;
    /* Preallocated scratch reused by Sinkhorn (and, in future, other
     * iterative solvers). All three are n-doubles, FFTW-aligned, and
     * lifecycle-managed by cheap_init / cheap_destroy. Never expose
     * these to users — they are overwritten on every call. */
    double* restrict scratch1;
    double* restrict scratch2;
    double* restrict prev_g;
    fftw_plan plan_fwd;
    fftw_plan plan_inv;
    double current_eps;
    double current_H;
} cheap_ctx;

static inline void cheap_destroy(cheap_ctx* ctx)
{
    if (!ctx) return;
    if (ctx->plan_fwd) { fftw_destroy_plan(ctx->plan_fwd); ctx->plan_fwd = NULL; }
    if (ctx->plan_inv) { fftw_destroy_plan(ctx->plan_inv); ctx->plan_inv = NULL; }
    if (ctx->lambda)     { fftw_free(ctx->lambda);     ctx->lambda = NULL; }
    if (ctx->gibbs)      { fftw_free(ctx->gibbs);      ctx->gibbs = NULL; }
    if (ctx->sqrt_lambda) { fftw_free(ctx->sqrt_lambda); ctx->sqrt_lambda = NULL; }
    if (ctx->workspace)  { fftw_free(ctx->workspace);  ctx->workspace = NULL; }
    if (ctx->scratch1)   { fftw_free(ctx->scratch1);   ctx->scratch1 = NULL; }
    if (ctx->scratch2)   { fftw_free(ctx->scratch2);   ctx->scratch2 = NULL; }
    if (ctx->prev_g)     { fftw_free(ctx->prev_g);     ctx->prev_g = NULL; }
    ctx->is_initialized = 0;
}

/*
 * cheap__alloc_ctx — private helper: allocate buffers and build FFTW plans.
 * Leaves lambda[] uninitialized; caller fills it. Returns CHEAP_OK or
 * CHEAP_ENOMEM (and calls cheap_destroy on failure).
 */
static inline int cheap__alloc_ctx(cheap_ctx* ctx, int n)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->n = n;
    ctx->current_eps = -1.0;
    ctx->current_H = -1.0;
    ctx->lambda      = (double*)fftw_malloc((size_t)n * sizeof(double));
    ctx->gibbs       = (double*)fftw_malloc((size_t)n * sizeof(double));
    ctx->sqrt_lambda = (double*)fftw_malloc((size_t)n * sizeof(double));
    ctx->workspace   = (double*)fftw_malloc((size_t)n * sizeof(double));
    ctx->scratch1    = (double*)fftw_malloc((size_t)n * sizeof(double));
    ctx->scratch2    = (double*)fftw_malloc((size_t)n * sizeof(double));
    ctx->prev_g      = (double*)fftw_malloc((size_t)n * sizeof(double));
    if (!ctx->lambda || !ctx->gibbs || !ctx->sqrt_lambda || !ctx->workspace ||
        !ctx->scratch1 || !ctx->scratch2 || !ctx->prev_g) {
        cheap_destroy(ctx);
        return CHEAP_ENOMEM;
    }
    CHEAP_ASSERT_ALIGNED(ctx->lambda);
    CHEAP_ASSERT_ALIGNED(ctx->gibbs);
    CHEAP_ASSERT_ALIGNED(ctx->sqrt_lambda);
    CHEAP_ASSERT_ALIGNED(ctx->workspace);
    CHEAP_ASSERT_ALIGNED(ctx->scratch1);
    CHEAP_ASSERT_ALIGNED(ctx->scratch2);
    CHEAP_ASSERT_ALIGNED(ctx->prev_g);
    ctx->plan_fwd = fftw_plan_r2r_1d(n, ctx->workspace, ctx->workspace,
                                      FFTW_REDFT10, FFTW_PATIENT);
    ctx->plan_inv = fftw_plan_r2r_1d(n, ctx->workspace, ctx->workspace,
                                      FFTW_REDFT01, FFTW_PATIENT);
    if (!ctx->plan_fwd || !ctx->plan_inv) {
        cheap_destroy(ctx);
        return CHEAP_ENOMEM;
    }
    return CHEAP_OK;
}

/*
 * cheap__finalize_sqrt_lambda — derive sqrt_lambda from lambda and flip the
 * is_initialized flag. Called at the end of every init entry point.
 *
 * Force-inlined (on GCC/Clang) so that callers of cheap_init can statically
 * observe that sqrt_lambda is fully written — avoids spurious
 * -Wmaybe-uninitialized warnings at -O3.
 */
static inline void cheap__finalize_sqrt_lambda(cheap_ctx* ctx)
{
    for (int k = 0; k < ctx->n; ++k)
        ctx->sqrt_lambda[k] = sqrt(fmax(ctx->lambda[k], CHEAP_EPS_LAMBDA));
    ctx->is_initialized = 1;
}

static inline int cheap_init(cheap_ctx* ctx, int n, double H)
{
    if (!ctx || n < 2 || H <= 0.0 || H >= 1.0) return CHEAP_EINVAL;
    int rc = cheap__alloc_ctx(ctx, n);
    if (rc != CHEAP_OK) return rc;
    ctx->current_H = H;
    const double pow_n_2H = pow((double)n, 2.0 * H);
    const double twoH_plus_1 = 2.0 * H + 1.0;
    /*
     * Flandrin spectrum: lambda_k = N^{2H} * sin(pi*k/(2N))^{-(2H+1)}.
     * The true lambda_0 diverges (DC singularity of fBm). We extrapolate
     * via the tail's power-law slope: in the large-k asymptotic
     * (Gupta-Joshi, n >> 1) lambda_k ~ k^{-(2H+1)}, so the ratio
     * lambda_{k-1}/lambda_k -> (k/(k-1))^{2H+1}. Setting
     * lambda_0 = lambda_1 * (lambda_1/lambda_2) preserves monotonicity
     * and matches the asymptotic log-slope exactly.
     */
    for (int k = 1; k < n; ++k) {
        double s = sin(M_PI * (double)k / (2.0 * (double)n));
        if (s < CHEAP_EPS_LOG) s = CHEAP_EPS_LOG;
        ctx->lambda[k] = pow_n_2H * pow(s, -twoH_plus_1);
    }
    if (n >= 3) {
        ctx->lambda[0] = ctx->lambda[1] * (ctx->lambda[1] / ctx->lambda[2]);
    } else {
        /* n == 2: no second tail point; fall back to doubling lambda[1]. */
        ctx->lambda[0] = 2.0 * ctx->lambda[1];
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(ctx->lambda, n);
    CHEAP_CONTRACT_STRICT_DEC_OR_EDOM(ctx->lambda, n);
    cheap__finalize_sqrt_lambda(ctx);
    return CHEAP_OK;
}

/*
 * cheap_init_from_toeplitz — initialize a context directly from the first
 * column of a symmetric Toeplitz autocovariance matrix. Computes the exact
 * DCT-II eigenvalues, bypassing the Flandrin asymptotic. Useful when the
 * covariance is measured or derived from a non-fBm model.
 *
 *   t:  first column of the n-by-n symmetric Toeplitz matrix (length n).
 *       All eigenvalues (entries of DCT-II(t)) must be strictly positive
 *       for the context to be valid for KRR/Sinkhorn/etc.
 *
 * On success, ctx->lambda holds the exact eigenvalues and ctx->current_H
 * remains -1.0 (sentinel indicating the Flandrin H is not applicable).
 */
static inline int cheap_init_from_toeplitz(cheap_ctx* ctx, int n,
                                             const double* t)
{
    if (!ctx || n < 2 || !t) return CHEAP_EINVAL;
    for (int i = 0; i < n; ++i) if (!isfinite(t[i])) return CHEAP_EDOM;
    int rc = cheap__alloc_ctx(ctx, n);
    if (rc != CHEAP_OK) return rc;
    memcpy(ctx->workspace, t, (size_t)n * sizeof(double));
    fftw_execute(ctx->plan_fwd);
    memcpy(ctx->lambda, ctx->workspace, (size_t)n * sizeof(double));
    CHEAP_CONTRACT_FINITE_OR_EDOM(ctx->lambda, n);
    cheap__finalize_sqrt_lambda(ctx);
    return CHEAP_OK;
}

/* ============================================================
 * 2D context — cheap_ctx_2d
 *
 * A 2D stationary covariance with separable structure diagonalizes under
 * the 2D DCT-II: lambda_{jk} = lambda_j^(x) * lambda_k^(y). We store the
 * nx*ny eigenvalue grid in row-major flat layout (index = j*ny + k), so
 * every pointwise kernel (cheap__mul_inplace, cheap__scale_copy, and every
 * _ev-family weight ctor) is dimension-agnostic and works as-is.
 *
 * Monotonicity of the flat lambda array is NOT guaranteed for tensor-product
 * spectra — CHEAP_DEBUG_CONTRACTS drops the strict-decreasing check here
 * and keeps only the finite sweep.
 * ============================================================ */

typedef struct {
    int nx, ny;
    int n;                          /* nx * ny, for convenience */
    int is_initialized;
    double* restrict lambda;        /* flat row-major nx*ny eigenvalues */
    double* restrict gibbs;         /* Sinkhorn Gibbs kernel (2D) */
    double* restrict sqrt_lambda;
    double* restrict workspace;     /* flat row-major nx*ny FFTW buffer */
    double* restrict scratch1;
    double* restrict scratch2;
    double* restrict prev_g;
    fftw_plan plan_fwd;             /* 2D DCT-II */
    fftw_plan plan_inv;             /* 2D iDCT-III */
    double current_eps;
    double current_Hx;              /* Flandrin H along x; -1.0 if from_toeplitz */
    double current_Hy;
} cheap_ctx_2d;

static inline void cheap_destroy_2d(cheap_ctx_2d* ctx)
{
    if (!ctx) return;
    if (ctx->plan_fwd) { fftw_destroy_plan(ctx->plan_fwd); ctx->plan_fwd = NULL; }
    if (ctx->plan_inv) { fftw_destroy_plan(ctx->plan_inv); ctx->plan_inv = NULL; }
    if (ctx->lambda)      { fftw_free(ctx->lambda);      ctx->lambda = NULL; }
    if (ctx->gibbs)       { fftw_free(ctx->gibbs);       ctx->gibbs = NULL; }
    if (ctx->sqrt_lambda) { fftw_free(ctx->sqrt_lambda); ctx->sqrt_lambda = NULL; }
    if (ctx->workspace)   { fftw_free(ctx->workspace);   ctx->workspace = NULL; }
    if (ctx->scratch1)    { fftw_free(ctx->scratch1);    ctx->scratch1 = NULL; }
    if (ctx->scratch2)    { fftw_free(ctx->scratch2);    ctx->scratch2 = NULL; }
    if (ctx->prev_g)      { fftw_free(ctx->prev_g);      ctx->prev_g = NULL; }
    ctx->is_initialized = 0;
}

/*
 * cheap__alloc_ctx_2d — private helper: allocate flat nx*ny buffers and
 * build 2D FFTW plans. Leaves lambda[] uninitialized.
 */
static inline int cheap__alloc_ctx_2d(cheap_ctx_2d* ctx, int nx, int ny)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->nx = nx;
    ctx->ny = ny;
    ctx->n  = nx * ny;
    ctx->current_eps = -1.0;
    ctx->current_Hx  = -1.0;
    ctx->current_Hy  = -1.0;
    const size_t bytes = (size_t)ctx->n * sizeof(double);
    ctx->lambda      = (double*)fftw_malloc(bytes);
    ctx->gibbs       = (double*)fftw_malloc(bytes);
    ctx->sqrt_lambda = (double*)fftw_malloc(bytes);
    ctx->workspace   = (double*)fftw_malloc(bytes);
    ctx->scratch1    = (double*)fftw_malloc(bytes);
    ctx->scratch2    = (double*)fftw_malloc(bytes);
    ctx->prev_g      = (double*)fftw_malloc(bytes);
    if (!ctx->lambda || !ctx->gibbs || !ctx->sqrt_lambda || !ctx->workspace ||
        !ctx->scratch1 || !ctx->scratch2 || !ctx->prev_g) {
        cheap_destroy_2d(ctx);
        return CHEAP_ENOMEM;
    }
    CHEAP_ASSERT_ALIGNED(ctx->lambda);
    CHEAP_ASSERT_ALIGNED(ctx->gibbs);
    CHEAP_ASSERT_ALIGNED(ctx->sqrt_lambda);
    CHEAP_ASSERT_ALIGNED(ctx->workspace);
    CHEAP_ASSERT_ALIGNED(ctx->scratch1);
    CHEAP_ASSERT_ALIGNED(ctx->scratch2);
    CHEAP_ASSERT_ALIGNED(ctx->prev_g);
    const fftw_r2r_kind fwd[2] = { FFTW_REDFT10, FFTW_REDFT10 };
    const fftw_r2r_kind inv[2] = { FFTW_REDFT01, FFTW_REDFT01 };
    ctx->plan_fwd = fftw_plan_r2r_2d(nx, ny, ctx->workspace, ctx->workspace,
                                      fwd[0], fwd[1], FFTW_PATIENT);
    ctx->plan_inv = fftw_plan_r2r_2d(nx, ny, ctx->workspace, ctx->workspace,
                                      inv[0], inv[1], FFTW_PATIENT);
    if (!ctx->plan_fwd || !ctx->plan_inv) {
        cheap_destroy_2d(ctx);
        return CHEAP_ENOMEM;
    }
    return CHEAP_OK;
}

/*
 * cheap__flandrin_1d_axis — fill an n-length Flandrin eigenvalue vector
 * using the same DC-extrapolation rule as cheap_init. Shared between
 * cheap_init and cheap_init_2d/3d tensor-product spectra.
 */
static inline void cheap__flandrin_1d_axis(double* restrict lam, int n, double H)
{
    const double pow_n_2H    = pow((double)n, 2.0 * H);
    const double twoH_plus_1 = 2.0 * H + 1.0;
    for (int k = 1; k < n; ++k) {
        double s = sin(M_PI * (double)k / (2.0 * (double)n));
        if (s < CHEAP_EPS_LOG) s = CHEAP_EPS_LOG;
        lam[k] = pow_n_2H * pow(s, -twoH_plus_1);
    }
    if (n >= 3)  lam[0] = lam[1] * (lam[1] / lam[2]);
    else         lam[0] = 2.0 * lam[1];
}

/*
 * cheap_init_2d — initialize a 2D context with a tensor-product Flandrin
 * spectrum: lambda_{jk} = lambda_j^(x, Hx) * lambda_k^(y, Hy).
 * nx, ny >= 2, Hx, Hy in (0, 1).
 */
static inline int cheap_init_2d(cheap_ctx_2d* ctx, int nx, int ny,
                                 double Hx, double Hy)
{
    if (!ctx || nx < 2 || ny < 2 ||
        Hx <= 0.0 || Hx >= 1.0 || Hy <= 0.0 || Hy >= 1.0) return CHEAP_EINVAL;
    int rc = cheap__alloc_ctx_2d(ctx, nx, ny);
    if (rc != CHEAP_OK) return rc;
    ctx->current_Hx = Hx;
    ctx->current_Hy = Hy;
    double* lx = (double*)fftw_malloc((size_t)nx * sizeof(double));
    double* ly = (double*)fftw_malloc((size_t)ny * sizeof(double));
    if (!lx || !ly) {
        if (lx) fftw_free(lx);
        if (ly) fftw_free(ly);
        cheap_destroy_2d(ctx);
        return CHEAP_ENOMEM;
    }
    cheap__flandrin_1d_axis(lx, nx, Hx);
    cheap__flandrin_1d_axis(ly, ny, Hy);
    for (int j = 0; j < nx; ++j)
        for (int k = 0; k < ny; ++k)
            ctx->lambda[j * ny + k] = lx[j] * ly[k];
    fftw_free(lx);
    fftw_free(ly);
    CHEAP_CONTRACT_FINITE_OR_EDOM(ctx->lambda, ctx->n);
    /* Monotonicity intentionally not checked: tensor-product spectra are
     * not sorted in row-major flat order. Each axis is individually
     * monotone, but the product grid is not. */
    for (int k = 0; k < ctx->n; ++k)
        ctx->sqrt_lambda[k] = sqrt(fmax(ctx->lambda[k], CHEAP_EPS_LAMBDA));
    ctx->is_initialized = 1;
    return CHEAP_OK;
}

/*
 * cheap_init_from_eigenvalues_2d — initialize a 2D context with a
 * user-supplied, flat row-major eigenvalue grid of length nx*ny.
 * All entries must be finite; positive entries are required for
 * KRR/Sinkhorn to remain well-posed (not enforced here — caller's
 * responsibility to ensure positivity when it matters).
 *
 * On success, current_Hx = current_Hy = -1.0 (sentinel).
 */
static inline int cheap_init_from_eigenvalues_2d(cheap_ctx_2d* ctx,
                                                  int nx, int ny,
                                                  const double* lambda_flat)
{
    if (!ctx || nx < 2 || ny < 2 || !lambda_flat) return CHEAP_EINVAL;
    const int n = nx * ny;
    for (int i = 0; i < n; ++i) if (!isfinite(lambda_flat[i])) return CHEAP_EDOM;
    int rc = cheap__alloc_ctx_2d(ctx, nx, ny);
    if (rc != CHEAP_OK) return rc;
    memcpy(ctx->lambda, lambda_flat, (size_t)n * sizeof(double));
    CHEAP_CONTRACT_FINITE_OR_EDOM(ctx->lambda, n);
    for (int k = 0; k < n; ++k)
        ctx->sqrt_lambda[k] = sqrt(fmax(ctx->lambda[k], CHEAP_EPS_LAMBDA));
    ctx->is_initialized = 1;
    return CHEAP_OK;
}

/*
 * cheap__dct2_1d_transient — run a one-shot DCT-II on an n-vector using
 * a locally-built FFTW_ESTIMATE plan. Used by cheap_init_from_toeplitz_2d
 * to compute per-axis eigenvalues without touching the 2D ctx's plans.
 * Destructive on input. Returns CHEAP_OK / CHEAP_ENOMEM.
 */
static inline int cheap__dct2_1d_transient(double* v, int n)
{
    fftw_plan p = fftw_plan_r2r_1d(n, v, v, FFTW_REDFT10, FFTW_ESTIMATE);
    if (!p) return CHEAP_ENOMEM;
    fftw_execute(p);
    fftw_destroy_plan(p);
    return CHEAP_OK;
}

/*
 * cheap_init_from_toeplitz_2d — initialize a 2D context from a separable
 * block-Toeplitz-with-Toeplitz-blocks (BTTB) covariance whose structure
 * factors as T_row ⊗ T_col. Takes two first-column vectors:
 *
 *   t_row: first column of the ny-by-ny row-axis Toeplitz block (length ny)
 *   t_col: first column of the nx-by-nx column-axis Toeplitz block (length nx)
 *
 * The 2D eigenvalues are the outer product of the axis eigenvalues:
 *   lambda_{jk} = DCT-II(t_col)[j] * DCT-II(t_row)[k].
 *
 * General (non-separable) BTTB is out of scope — use
 * cheap_init_from_eigenvalues_2d if you have the full grid.
 *
 * On success, current_Hx = current_Hy = -1.0.
 */
static inline int cheap_init_from_toeplitz_2d(cheap_ctx_2d* ctx,
                                                int nx, int ny,
                                                const double* t_col,
                                                const double* t_row)
{
    if (!ctx || nx < 2 || ny < 2 || !t_col || !t_row) return CHEAP_EINVAL;
    for (int i = 0; i < nx; ++i) if (!isfinite(t_col[i])) return CHEAP_EDOM;
    for (int i = 0; i < ny; ++i) if (!isfinite(t_row[i])) return CHEAP_EDOM;
    int rc = cheap__alloc_ctx_2d(ctx, nx, ny);
    if (rc != CHEAP_OK) return rc;

    double* lx = (double*)fftw_malloc((size_t)nx * sizeof(double));
    double* ly = (double*)fftw_malloc((size_t)ny * sizeof(double));
    if (!lx || !ly) {
        if (lx) fftw_free(lx);
        if (ly) fftw_free(ly);
        cheap_destroy_2d(ctx);
        return CHEAP_ENOMEM;
    }
    memcpy(lx, t_col, (size_t)nx * sizeof(double));
    memcpy(ly, t_row, (size_t)ny * sizeof(double));
    if (cheap__dct2_1d_transient(lx, nx) != CHEAP_OK ||
        cheap__dct2_1d_transient(ly, ny) != CHEAP_OK) {
        fftw_free(lx);
        fftw_free(ly);
        cheap_destroy_2d(ctx);
        return CHEAP_ENOMEM;
    }
    for (int j = 0; j < nx; ++j)
        for (int k = 0; k < ny; ++k)
            ctx->lambda[j * ny + k] = lx[j] * ly[k];
    fftw_free(lx);
    fftw_free(ly);
    CHEAP_CONTRACT_FINITE_OR_EDOM(ctx->lambda, ctx->n);
    for (int k = 0; k < ctx->n; ++k)
        ctx->sqrt_lambda[k] = sqrt(fmax(ctx->lambda[k], CHEAP_EPS_LAMBDA));
    ctx->is_initialized = 1;
    return CHEAP_OK;
}

/* ============================================================
 * 3D context — cheap_ctx_3d
 *
 * Mirrors cheap_ctx_2d for 3D grids. lambda is flat row-major of
 * length nx*ny*nz (index = (j*ny + k)*nz + l). FFTW_REDFT10/REDFT01
 * along all three axes. Normalization factor is 1/(8*nx*ny*nz).
 * ============================================================ */

typedef struct {
    int nx, ny, nz;
    int n;                          /* nx * ny * nz */
    int is_initialized;
    double* restrict lambda;
    double* restrict gibbs;
    double* restrict sqrt_lambda;
    double* restrict workspace;
    double* restrict scratch1;
    double* restrict scratch2;
    double* restrict prev_g;
    fftw_plan plan_fwd;             /* 3D DCT-II */
    fftw_plan plan_inv;             /* 3D iDCT-III */
    double current_eps;
    double current_Hx;              /* -1.0 if from_toeplitz/from_eigenvalues */
    double current_Hy;
    double current_Hz;
} cheap_ctx_3d;

static inline void cheap_destroy_3d(cheap_ctx_3d* ctx)
{
    if (!ctx) return;
    if (ctx->plan_fwd) { fftw_destroy_plan(ctx->plan_fwd); ctx->plan_fwd = NULL; }
    if (ctx->plan_inv) { fftw_destroy_plan(ctx->plan_inv); ctx->plan_inv = NULL; }
    if (ctx->lambda)      { fftw_free(ctx->lambda);      ctx->lambda = NULL; }
    if (ctx->gibbs)       { fftw_free(ctx->gibbs);       ctx->gibbs = NULL; }
    if (ctx->sqrt_lambda) { fftw_free(ctx->sqrt_lambda); ctx->sqrt_lambda = NULL; }
    if (ctx->workspace)   { fftw_free(ctx->workspace);   ctx->workspace = NULL; }
    if (ctx->scratch1)    { fftw_free(ctx->scratch1);    ctx->scratch1 = NULL; }
    if (ctx->scratch2)    { fftw_free(ctx->scratch2);    ctx->scratch2 = NULL; }
    if (ctx->prev_g)      { fftw_free(ctx->prev_g);      ctx->prev_g = NULL; }
    ctx->is_initialized = 0;
}

/*
 * cheap__alloc_ctx_3d — private helper: allocate flat nx*ny*nz buffers
 * and build 3D FFTW plans. Leaves lambda[] uninitialized.
 */
static inline int cheap__alloc_ctx_3d(cheap_ctx_3d* ctx, int nx, int ny, int nz)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->nx = nx;
    ctx->ny = ny;
    ctx->nz = nz;
    ctx->n  = nx * ny * nz;
    ctx->current_eps = -1.0;
    ctx->current_Hx  = -1.0;
    ctx->current_Hy  = -1.0;
    ctx->current_Hz  = -1.0;
    const size_t bytes = (size_t)ctx->n * sizeof(double);
    ctx->lambda      = (double*)fftw_malloc(bytes);
    ctx->gibbs       = (double*)fftw_malloc(bytes);
    ctx->sqrt_lambda = (double*)fftw_malloc(bytes);
    ctx->workspace   = (double*)fftw_malloc(bytes);
    ctx->scratch1    = (double*)fftw_malloc(bytes);
    ctx->scratch2    = (double*)fftw_malloc(bytes);
    ctx->prev_g      = (double*)fftw_malloc(bytes);
    if (!ctx->lambda || !ctx->gibbs || !ctx->sqrt_lambda || !ctx->workspace ||
        !ctx->scratch1 || !ctx->scratch2 || !ctx->prev_g) {
        cheap_destroy_3d(ctx);
        return CHEAP_ENOMEM;
    }
    CHEAP_ASSERT_ALIGNED(ctx->lambda);
    CHEAP_ASSERT_ALIGNED(ctx->gibbs);
    CHEAP_ASSERT_ALIGNED(ctx->sqrt_lambda);
    CHEAP_ASSERT_ALIGNED(ctx->workspace);
    CHEAP_ASSERT_ALIGNED(ctx->scratch1);
    CHEAP_ASSERT_ALIGNED(ctx->scratch2);
    CHEAP_ASSERT_ALIGNED(ctx->prev_g);
    ctx->plan_fwd = fftw_plan_r2r_3d(nx, ny, nz,
                                      ctx->workspace, ctx->workspace,
                                      FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10,
                                      FFTW_PATIENT);
    ctx->plan_inv = fftw_plan_r2r_3d(nx, ny, nz,
                                      ctx->workspace, ctx->workspace,
                                      FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01,
                                      FFTW_PATIENT);
    if (!ctx->plan_fwd || !ctx->plan_inv) {
        cheap_destroy_3d(ctx);
        return CHEAP_ENOMEM;
    }
    return CHEAP_OK;
}

/*
 * cheap_init_3d — initialize a 3D context with a tensor-product Flandrin
 * spectrum: lambda_{jkl} = lambda_j^(x,Hx) * lambda_k^(y,Hy) * lambda_l^(z,Hz).
 */
static inline int cheap_init_3d(cheap_ctx_3d* ctx, int nx, int ny, int nz,
                                  double Hx, double Hy, double Hz)
{
    if (!ctx || nx < 2 || ny < 2 || nz < 2 ||
        Hx <= 0.0 || Hx >= 1.0 ||
        Hy <= 0.0 || Hy >= 1.0 ||
        Hz <= 0.0 || Hz >= 1.0) return CHEAP_EINVAL;
    int rc = cheap__alloc_ctx_3d(ctx, nx, ny, nz);
    if (rc != CHEAP_OK) return rc;
    ctx->current_Hx = Hx;
    ctx->current_Hy = Hy;
    ctx->current_Hz = Hz;
    double* lx = (double*)fftw_malloc((size_t)nx * sizeof(double));
    double* ly = (double*)fftw_malloc((size_t)ny * sizeof(double));
    double* lz = (double*)fftw_malloc((size_t)nz * sizeof(double));
    if (!lx || !ly || !lz) {
        if (lx) fftw_free(lx);
        if (ly) fftw_free(ly);
        if (lz) fftw_free(lz);
        cheap_destroy_3d(ctx);
        return CHEAP_ENOMEM;
    }
    cheap__flandrin_1d_axis(lx, nx, Hx);
    cheap__flandrin_1d_axis(ly, ny, Hy);
    cheap__flandrin_1d_axis(lz, nz, Hz);
    for (int j = 0; j < nx; ++j) {
        for (int k = 0; k < ny; ++k) {
            double xy = lx[j] * ly[k];
            for (int l = 0; l < nz; ++l) {
                ctx->lambda[(j * ny + k) * nz + l] = xy * lz[l];
            }
        }
    }
    fftw_free(lx);
    fftw_free(ly);
    fftw_free(lz);
    CHEAP_CONTRACT_FINITE_OR_EDOM(ctx->lambda, ctx->n);
    for (int k = 0; k < ctx->n; ++k)
        ctx->sqrt_lambda[k] = sqrt(fmax(ctx->lambda[k], CHEAP_EPS_LAMBDA));
    ctx->is_initialized = 1;
    return CHEAP_OK;
}

/*
 * cheap_init_from_eigenvalues_3d — user-supplied flat row-major
 * eigenvalue grid of length nx*ny*nz. current_H{x,y,z} = -1.0.
 */
static inline int cheap_init_from_eigenvalues_3d(cheap_ctx_3d* ctx,
                                                  int nx, int ny, int nz,
                                                  const double* lambda_flat)
{
    if (!ctx || nx < 2 || ny < 2 || nz < 2 || !lambda_flat) return CHEAP_EINVAL;
    const int n = nx * ny * nz;
    for (int i = 0; i < n; ++i)
        if (!isfinite(lambda_flat[i])) return CHEAP_EDOM;
    int rc = cheap__alloc_ctx_3d(ctx, nx, ny, nz);
    if (rc != CHEAP_OK) return rc;
    memcpy(ctx->lambda, lambda_flat, (size_t)n * sizeof(double));
    CHEAP_CONTRACT_FINITE_OR_EDOM(ctx->lambda, ctx->n);
    for (int k = 0; k < ctx->n; ++k)
        ctx->sqrt_lambda[k] = sqrt(fmax(ctx->lambda[k], CHEAP_EPS_LAMBDA));
    ctx->is_initialized = 1;
    return CHEAP_OK;
}

/*
 * cheap_init_from_toeplitz_3d — separable BTTTB covariance specified
 * by three 1D Toeplitz first columns. Each axis is DCT'd with a
 * transient FFTW_ESTIMATE plan (see cheap__dct2_1d_transient), then
 * tensor-producted into the flat nx*ny*nz eigenvalue grid.
 */
static inline int cheap_init_from_toeplitz_3d(cheap_ctx_3d* ctx,
                                                int nx, int ny, int nz,
                                                const double* t_x,
                                                const double* t_y,
                                                const double* t_z)
{
    if (!ctx || nx < 2 || ny < 2 || nz < 2 || !t_x || !t_y || !t_z)
        return CHEAP_EINVAL;
    for (int i = 0; i < nx; ++i) if (!isfinite(t_x[i])) return CHEAP_EDOM;
    for (int i = 0; i < ny; ++i) if (!isfinite(t_y[i])) return CHEAP_EDOM;
    for (int i = 0; i < nz; ++i) if (!isfinite(t_z[i])) return CHEAP_EDOM;
    int rc = cheap__alloc_ctx_3d(ctx, nx, ny, nz);
    if (rc != CHEAP_OK) return rc;

    double* lx = (double*)fftw_malloc((size_t)nx * sizeof(double));
    double* ly = (double*)fftw_malloc((size_t)ny * sizeof(double));
    double* lz = (double*)fftw_malloc((size_t)nz * sizeof(double));
    if (!lx || !ly || !lz) {
        if (lx) fftw_free(lx);
        if (ly) fftw_free(ly);
        if (lz) fftw_free(lz);
        cheap_destroy_3d(ctx);
        return CHEAP_ENOMEM;
    }
    memcpy(lx, t_x, (size_t)nx * sizeof(double));
    memcpy(ly, t_y, (size_t)ny * sizeof(double));
    memcpy(lz, t_z, (size_t)nz * sizeof(double));
    if (cheap__dct2_1d_transient(lx, nx) != CHEAP_OK ||
        cheap__dct2_1d_transient(ly, ny) != CHEAP_OK ||
        cheap__dct2_1d_transient(lz, nz) != CHEAP_OK) {
        fftw_free(lx);
        fftw_free(ly);
        fftw_free(lz);
        cheap_destroy_3d(ctx);
        return CHEAP_ENOMEM;
    }
    for (int j = 0; j < nx; ++j) {
        for (int k = 0; k < ny; ++k) {
            double xy = lx[j] * ly[k];
            for (int l = 0; l < nz; ++l) {
                ctx->lambda[(j * ny + k) * nz + l] = xy * lz[l];
            }
        }
    }
    fftw_free(lx);
    fftw_free(ly);
    fftw_free(lz);
    CHEAP_CONTRACT_FINITE_OR_EDOM(ctx->lambda, ctx->n);
    for (int k = 0; k < ctx->n; ++k)
        ctx->sqrt_lambda[k] = sqrt(fmax(ctx->lambda[k], CHEAP_EPS_LAMBDA));
    ctx->is_initialized = 1;
    return CHEAP_OK;
}

/* ============================================================
 * Core spectral primitives: forward, inverse, apply
 * ============================================================ */

/*
 * cheap__mul_inplace — ws[k] *= w[k], vectorized where available.
 * ws is aligned (ctx->workspace); w may be unaligned (user buffer).
 */
static inline void cheap__mul_inplace(double* restrict ws,
                                        const double* restrict w, int n)
{
    int k = 0;
#if defined(CHEAP_SIMD_AVX2)
    for (; k + 4 <= n; k += 4) {
        __m256d vw = _mm256_loadu_pd(ws + k);
        __m256d vv = _mm256_loadu_pd(w  + k);
        _mm256_storeu_pd(ws + k, _mm256_mul_pd(vw, vv));
    }
#elif defined(CHEAP_SIMD_NEON)
    for (; k + 2 <= n; k += 2) {
        float64x2_t vw = vld1q_f64(ws + k);
        float64x2_t vv = vld1q_f64(w  + k);
        vst1q_f64(ws + k, vmulq_f64(vw, vv));
    }
#endif
    for (; k < n; ++k) ws[k] *= w[k];
}

/*
 * cheap__scale_copy — output[i] = ws[i] * norm, vectorized where
 * available. ws is aligned; output may be unaligned. Safe when
 * output == ws (in-place variant), so no restrict on the pointers.
 */
static inline void cheap__scale_copy(double* output,
                                       const double* ws,
                                       double norm, int n)
{
    int i = 0;
#if defined(CHEAP_SIMD_AVX2)
    __m256d vn = _mm256_set1_pd(norm);
    for (; i + 4 <= n; i += 4) {
        __m256d vw = _mm256_loadu_pd(ws + i);
        _mm256_storeu_pd(output + i, _mm256_mul_pd(vw, vn));
    }
#elif defined(CHEAP_SIMD_NEON)
    float64x2_t vn = vdupq_n_f64(norm);
    for (; i + 2 <= n; i += 2) {
        float64x2_t vw = vld1q_f64(ws + i);
        vst1q_f64(output + i, vmulq_f64(vw, vn));
    }
#endif
    for (; i < n; ++i) output[i] = ws[i] * norm;
}

/*
 * cheap_forward — DCT-II of input into ctx->workspace.
 * After this call, ctx->workspace contains the spectral coefficients.
 */
static inline int cheap_forward(cheap_ctx* restrict ctx,
                                  const double* restrict input)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    if (!input) return CHEAP_EINVAL;
    const int n = ctx->n;
    for (int i = 0; i < n; ++i) if (!isfinite(input[i])) return CHEAP_EDOM;
    memcpy(ctx->workspace, input, (size_t)n * sizeof(double));
    fftw_execute(ctx->plan_fwd);
    return CHEAP_OK;
}

/*
 * cheap_forward_inplace — DCT-II of ctx->workspace in place.
 * Caller is responsible for populating ctx->workspace with input
 * data first (e.g. via cheap_workspace(ctx)). Skips the input
 * memcpy and isfinite sweep — caller must guarantee finiteness.
 */
static inline int cheap_forward_inplace(cheap_ctx* restrict ctx)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    fftw_execute(ctx->plan_fwd);
    return CHEAP_OK;
}

/*
 * cheap_inverse — iDCT-III of ctx->workspace into output, with 1/(2N) normalization.
 * Assumes ctx->workspace already contains spectral data (e.g. from cheap_forward
 * followed by pointwise manipulation). SIMD-accelerated scale pass.
 */
static inline int cheap_inverse(cheap_ctx* restrict ctx,
                                  double* restrict output)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    if (!output) return CHEAP_EINVAL;
    const int n = ctx->n;
    const double norm = 1.0 / (2.0 * (double)n);
    fftw_execute(ctx->plan_inv);
    cheap__scale_copy(output, ctx->workspace, norm, n);
    return CHEAP_OK;
}

/*
 * cheap_inverse_inplace — iDCT-III of ctx->workspace in place with
 * normalization. Leaves the result in ctx->workspace; caller reads
 * via cheap_workspace(ctx). Saves one memcpy relative to cheap_inverse.
 */
static inline int cheap_inverse_inplace(cheap_ctx* restrict ctx)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    const int n = ctx->n;
    const double norm = 1.0 / (2.0 * (double)n);
    fftw_execute(ctx->plan_inv);
    cheap__scale_copy(ctx->workspace, ctx->workspace, norm, n);
    return CHEAP_OK;
}

/*
 * cheap_apply — Universal spectral operation:
 *   output = iDCT( DCT(input) ⊙ weights ) / (2N)
 *
 * This is the fundamental building block. Every spectral algorithm reduces to
 * choosing the right weight vector:
 *   - KRR solve:   weights[k] = 1 / (lambda[k] + lambda_reg)
 *   - Reparam:     weights[k] = sqrt(lambda[k])
 *   - LQR/MPC:     weights[k] = 1 / (lambda[k] + R)
 *   - Frac diff:   weights[k] = (2*sin(omega_k/2))^d
 * AVX2/NEON accelerated pointwise-multiply and scale passes.
 */
static inline int cheap_apply(cheap_ctx* restrict ctx,
                                const double* restrict input,
                                const double* restrict weights,
                                double* restrict output)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    if (!input || !weights || !output) return CHEAP_EINVAL;
    const int n = ctx->n;
    const double norm = 1.0 / (2.0 * (double)n);
    for (int i = 0; i < n; ++i) if (!isfinite(input[i])) return CHEAP_EDOM;
    memcpy(ctx->workspace, input, (size_t)n * sizeof(double));
    fftw_execute(ctx->plan_fwd);
    cheap__mul_inplace(ctx->workspace, weights, n);
    fftw_execute(ctx->plan_inv);
    cheap__scale_copy(output, ctx->workspace, norm, n);
    return CHEAP_OK;
}

/*
 * cheap_apply_inplace — in-place twin of cheap_apply.
 * Operates entirely on ctx->workspace: caller populates it first
 * (e.g. via cheap_workspace(ctx)), then reads the result from the
 * same buffer. Skips the input memcpy and isfinite sweep — caller
 * must guarantee finiteness. AVX2/NEON accelerated.
 */
static inline int cheap_apply_inplace(cheap_ctx* restrict ctx,
                                      const double* restrict weights)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    if (!weights) return CHEAP_EINVAL;
    const int n = ctx->n;
    const double norm = 1.0 / (2.0 * (double)n);
    fftw_execute(ctx->plan_fwd);
    cheap__mul_inplace(ctx->workspace, weights, n);
    fftw_execute(ctx->plan_inv);
    cheap__scale_copy(ctx->workspace, ctx->workspace, norm, n);
    return CHEAP_OK;
}

/* ============================================================
 * 2D spectral primitives: forward, inverse, apply
 *
 * Operate on the flat row-major nx*ny workspace of cheap_ctx_2d.
 * Normalization is 1/(4*nx*ny) — each DCT axis contributes a
 * factor of 2N_axis under the FFTW REDFT10/REDFT01 convention.
 * ============================================================ */

/*
 * cheap_forward_2d — 2D DCT-II of input into ctx->workspace.
 */
static inline int cheap_forward_2d(cheap_ctx_2d* restrict ctx,
                                     const double* restrict input)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    if (!input) return CHEAP_EINVAL;
    const int n = ctx->n;
    for (int i = 0; i < n; ++i) if (!isfinite(input[i])) return CHEAP_EDOM;
    memcpy(ctx->workspace, input, (size_t)n * sizeof(double));
    fftw_execute(ctx->plan_fwd);
    return CHEAP_OK;
}

/*
 * cheap_forward_inplace_2d — 2D DCT-II of ctx->workspace in place.
 * Caller must populate workspace beforehand (e.g. via cheap_workspace_2d).
 */
static inline int cheap_forward_inplace_2d(cheap_ctx_2d* restrict ctx)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    fftw_execute(ctx->plan_fwd);
    return CHEAP_OK;
}

/*
 * cheap_inverse_2d — 2D iDCT-III of ctx->workspace into output,
 * with 1/(4*nx*ny) normalization.
 */
static inline int cheap_inverse_2d(cheap_ctx_2d* restrict ctx,
                                     double* restrict output)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    if (!output) return CHEAP_EINVAL;
    const int n = ctx->n;
    const double norm = 1.0 / (4.0 * (double)ctx->nx * (double)ctx->ny);
    fftw_execute(ctx->plan_inv);
    cheap__scale_copy(output, ctx->workspace, norm, n);
    return CHEAP_OK;
}

/*
 * cheap_inverse_inplace_2d — iDCT-III of ctx->workspace in place
 * with 1/(4*nx*ny) normalization.
 */
static inline int cheap_inverse_inplace_2d(cheap_ctx_2d* restrict ctx)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    const int n = ctx->n;
    const double norm = 1.0 / (4.0 * (double)ctx->nx * (double)ctx->ny);
    fftw_execute(ctx->plan_inv);
    cheap__scale_copy(ctx->workspace, ctx->workspace, norm, n);
    return CHEAP_OK;
}

/*
 * cheap_apply_2d — 2D universal spectral operation:
 *   output = iDCT2( DCT2(input) ⊙ weights ) / (4*nx*ny)
 *
 * weights is a flat nx*ny array in row-major order matching
 * ctx->lambda. Reuses the SIMD-accelerated cheap__mul_inplace and
 * cheap__scale_copy helpers on the flat workspace.
 */
static inline int cheap_apply_2d(cheap_ctx_2d* restrict ctx,
                                   const double* restrict input,
                                   const double* restrict weights,
                                   double* restrict output)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    if (!input || !weights || !output) return CHEAP_EINVAL;
    const int n = ctx->n;
    const double norm = 1.0 / (4.0 * (double)ctx->nx * (double)ctx->ny);
    for (int i = 0; i < n; ++i) if (!isfinite(input[i])) return CHEAP_EDOM;
    memcpy(ctx->workspace, input, (size_t)n * sizeof(double));
    fftw_execute(ctx->plan_fwd);
    cheap__mul_inplace(ctx->workspace, weights, n);
    fftw_execute(ctx->plan_inv);
    cheap__scale_copy(output, ctx->workspace, norm, n);
    return CHEAP_OK;
}

/*
 * cheap_apply_inplace_2d — in-place twin of cheap_apply_2d.
 * Operates entirely on ctx->workspace: caller populates it first
 * (e.g. via cheap_workspace_2d), then reads the result from the
 * same buffer. Skips the input memcpy and isfinite sweep.
 */
static inline int cheap_apply_inplace_2d(cheap_ctx_2d* restrict ctx,
                                           const double* restrict weights)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    if (!weights) return CHEAP_EINVAL;
    const int n = ctx->n;
    const double norm = 1.0 / (4.0 * (double)ctx->nx * (double)ctx->ny);
    fftw_execute(ctx->plan_fwd);
    cheap__mul_inplace(ctx->workspace, weights, n);
    fftw_execute(ctx->plan_inv);
    cheap__scale_copy(ctx->workspace, ctx->workspace, norm, n);
    return CHEAP_OK;
}

/*
 * cheap_workspace_2d — accessor to the flat nx*ny workspace buffer.
 * Mirrors cheap_workspace for the 2D context.
 */
static inline double* cheap_workspace_2d(cheap_ctx_2d* ctx)
{
    return (ctx && ctx->is_initialized) ? ctx->workspace : NULL;
}

/* ============================================================
 * 3D spectral primitives: forward, inverse, apply
 *
 * Operate on the flat row-major nx*ny*nz workspace of cheap_ctx_3d.
 * Normalization is 1/(8*nx*ny*nz) — each DCT axis contributes a
 * factor of 2N_axis under the FFTW REDFT10/REDFT01 convention.
 * ============================================================ */

static inline int cheap_forward_3d(cheap_ctx_3d* restrict ctx,
                                     const double* restrict input)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    if (!input) return CHEAP_EINVAL;
    const int n = ctx->n;
    for (int i = 0; i < n; ++i) if (!isfinite(input[i])) return CHEAP_EDOM;
    memcpy(ctx->workspace, input, (size_t)n * sizeof(double));
    fftw_execute(ctx->plan_fwd);
    return CHEAP_OK;
}

static inline int cheap_forward_inplace_3d(cheap_ctx_3d* restrict ctx)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    fftw_execute(ctx->plan_fwd);
    return CHEAP_OK;
}

static inline int cheap_inverse_3d(cheap_ctx_3d* restrict ctx,
                                     double* restrict output)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    if (!output) return CHEAP_EINVAL;
    const int n = ctx->n;
    const double norm = 1.0 / (8.0 * (double)ctx->nx * (double)ctx->ny * (double)ctx->nz);
    fftw_execute(ctx->plan_inv);
    cheap__scale_copy(output, ctx->workspace, norm, n);
    return CHEAP_OK;
}

static inline int cheap_inverse_inplace_3d(cheap_ctx_3d* restrict ctx)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    const int n = ctx->n;
    const double norm = 1.0 / (8.0 * (double)ctx->nx * (double)ctx->ny * (double)ctx->nz);
    fftw_execute(ctx->plan_inv);
    cheap__scale_copy(ctx->workspace, ctx->workspace, norm, n);
    return CHEAP_OK;
}

static inline int cheap_apply_3d(cheap_ctx_3d* restrict ctx,
                                   const double* restrict input,
                                   const double* restrict weights,
                                   double* restrict output)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    if (!input || !weights || !output) return CHEAP_EINVAL;
    const int n = ctx->n;
    const double norm = 1.0 / (8.0 * (double)ctx->nx * (double)ctx->ny * (double)ctx->nz);
    for (int i = 0; i < n; ++i) if (!isfinite(input[i])) return CHEAP_EDOM;
    memcpy(ctx->workspace, input, (size_t)n * sizeof(double));
    fftw_execute(ctx->plan_fwd);
    cheap__mul_inplace(ctx->workspace, weights, n);
    fftw_execute(ctx->plan_inv);
    cheap__scale_copy(output, ctx->workspace, norm, n);
    return CHEAP_OK;
}

static inline int cheap_apply_inplace_3d(cheap_ctx_3d* restrict ctx,
                                           const double* restrict weights)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    if (!weights) return CHEAP_EINVAL;
    const int n = ctx->n;
    const double norm = 1.0 / (8.0 * (double)ctx->nx * (double)ctx->ny * (double)ctx->nz);
    fftw_execute(ctx->plan_fwd);
    cheap__mul_inplace(ctx->workspace, weights, n);
    fftw_execute(ctx->plan_inv);
    cheap__scale_copy(ctx->workspace, ctx->workspace, norm, n);
    return CHEAP_OK;
}

static inline double* cheap_workspace_3d(cheap_ctx_3d* ctx)
{
    return (ctx && ctx->is_initialized) ? ctx->workspace : NULL;
}

/* ============================================================
 * Sinkhorn optimal transport (max-log stabilized)
 * ============================================================ */

static inline int cheap_recompute_gibbs(cheap_ctx* restrict ctx, double eps)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    if (eps <= 0.0) return CHEAP_EINVAL;
    if (fabs(eps - ctx->current_eps) > 1e-15 * fmax(fabs(eps), 1.0)) {
        const int n = ctx->n;
        for (int k = 0; k < n; ++k) {
            double arg = -ctx->lambda[k] / eps;
            ctx->gibbs[k] = (arg < -700.0) ? 0.0 : exp(arg);
        }
        ctx->current_eps = eps;
    }
    return CHEAP_OK;
}

static inline void cheap_apply_hybrid_log(cheap_ctx* restrict ctx,
                                            const double* restrict f,
                                            double* restrict out)
{
    if (!ctx || !ctx->is_initialized || !f || !out) return;
    const int n = ctx->n;
    const double norm = 1.0 / (2.0 * (double)n);
    double max_f = f[0];
    for (int i = 1; i < n; ++i) if (f[i] > max_f) max_f = f[i];
    for (int i = 0; i < n; ++i) ctx->workspace[i] = exp(f[i] - max_f);
    fftw_execute(ctx->plan_fwd);
    for (int k = 0; k < n; ++k) ctx->workspace[k] *= ctx->gibbs[k];
    fftw_execute(ctx->plan_inv);
    for (int i = 0; i < n; ++i) {
        double Kv_i = ctx->workspace[i] * norm;
        out[i] = log(Kv_i + CHEAP_EPS_DIV) + max_f;
    }
}

static inline int cheap_sinkhorn(cheap_ctx* restrict ctx,
                                   const double* restrict a,
                                   const double* restrict b,
                                   double eps, int max_iter, double tol,
                                   double* restrict f,
                                   double* restrict g)
{
    if (!ctx || !ctx->is_initialized || !a || !b || !f || !g) return CHEAP_EINVAL;
    if (eps <= 0.0) return CHEAP_EINVAL;
    int ret = cheap_recompute_gibbs(ctx, eps);
    if (ret != CHEAP_OK) return ret;
    double sum_a = 0.0, sum_b = 0.0;
    for (int i = 0; i < ctx->n; ++i) {
        if (!isfinite(a[i]) || !isfinite(b[i])) return CHEAP_EDOM;
        sum_a += a[i];
        sum_b += b[i];
    }
    if (fabs(sum_a - sum_b) > 1e-8 * fmax(fabs(sum_a), 1.0)) return CHEAP_EINVAL;
    const int n = ctx->n;
    for (int i = 0; i < n; ++i) f[i] = g[i] = 0.0;
    /* Reuse ctx-owned scratch buffers — zero heap traffic in the hot loop. */
    double* const temp   = ctx->scratch1;
    double* const prev_g = ctx->prev_g;
    int iter;
    for (iter = 0; iter < max_iter; ++iter) {
        memcpy(prev_g, g, (size_t)n * sizeof(double));
        cheap_apply_hybrid_log(ctx, g, temp);
        for (int i = 0; i < n; ++i) f[i] = log(a[i] + CHEAP_EPS_DIV) - temp[i];
        cheap_apply_hybrid_log(ctx, f, temp);
        for (int i = 0; i < n; ++i) g[i] = log(b[i] + CHEAP_EPS_DIV) - temp[i];
        double max_diff = 0.0;
        for (int i = 0; i < n; ++i) {
            double diff = fabs(g[i] - prev_g[i]);
            if (diff > max_diff) max_diff = diff;
        }
        if (max_diff < tol) break;
    }
    return (iter < max_iter) ? CHEAP_OK : CHEAP_ENOCONV;
}

/* ============================================================
 * Toeplitz utilities
 *
 * Generic Toeplitz matvec and solve using DCT diagonalization.
 * These reuse cheap_ctx for its FFTW plans and workspace but
 * ignore ctx->lambda — the caller supplies the first column of
 * the Toeplitz matrix or pre-computed eigenvalues.
 * ============================================================ */

static inline int cheap_toeplitz_eigenvalues(cheap_ctx* restrict ctx,
                                                const double* restrict t,
                                                double* restrict lambda_out)
{
    if (!ctx || !ctx->is_initialized || !t || !lambda_out) return CHEAP_EINVAL;
    const int n = ctx->n;
    for (int i = 0; i < n; ++i) if (!isfinite(t[i])) return CHEAP_EDOM;
    memcpy(ctx->workspace, t, (size_t)n * sizeof(double));
    fftw_execute(ctx->plan_fwd);
    memcpy(lambda_out, ctx->workspace, (size_t)n * sizeof(double));
    return CHEAP_OK;
}

static inline int cheap_toeplitz_solve_precomp(cheap_ctx* restrict ctx,
                                                  const double* restrict lambda_t,
                                                  const double* restrict y,
                                                  double lambda_reg,
                                                  double* restrict x)
{
    if (!ctx || !ctx->is_initialized || !lambda_t || !y || !x) return CHEAP_EINVAL;
    if (lambda_reg < 0.0) return CHEAP_EINVAL;
    const int n = ctx->n;
    const double norm = 1.0 / (2.0 * (double)n);
    for (int i = 0; i < n; ++i) if (!isfinite(y[i])) return CHEAP_EDOM;
    memcpy(ctx->workspace, y, (size_t)n * sizeof(double));
    fftw_execute(ctx->plan_fwd);
    for (int k = 0; k < n; ++k) {
        double denom = lambda_t[k] + lambda_reg;
        if (fabs(denom) < CHEAP_EPS_DIV)
            denom = (denom >= 0.0) ? CHEAP_EPS_DIV : -CHEAP_EPS_DIV;
        ctx->workspace[k] /= denom;
    }
    fftw_execute(ctx->plan_inv);
    for (int i = 0; i < n; ++i) x[i] = ctx->workspace[i] * norm;
    return CHEAP_OK;
}

/* ============================================================
 * Internal PRNG (64-bit LCG + Box-Muller with caching)
 *
 * Used only at init time for RFF frequency sampling.
 * Not cryptographically secure — adequate for Monte Carlo.
 * ============================================================ */

typedef struct {
    uint64_t state;
    double cached_normal;
    int has_cached;
} cheap__rng;

static inline void cheap__rng_init(cheap__rng* rng, uint64_t seed)
{
    rng->state = seed;
    rng->cached_normal = 0.0;
    rng->has_cached = 0;
}

static inline double cheap__lcg_uniform(cheap__rng* rng)
{
    rng->state = rng->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((double)(rng->state >> 11) + 0.5) / (double)(1ULL << 53);
}

static inline double cheap__lcg_normal(cheap__rng* rng)
{
    if (rng->has_cached) {
        rng->has_cached = 0;
        return rng->cached_normal;
    }
    double u1 = cheap__lcg_uniform(rng);
    double u2 = cheap__lcg_uniform(rng);
    if (u1 < 1e-300) u1 = 1e-300;
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    rng->cached_normal = r * sin(theta);
    rng->has_cached = 1;
    return r * cos(theta);
}

/* ============================================================
 * Spectral weight constructors
 *
 * These functions compute weight vectors for use with cheap_apply().
 * Two eigenvalue families appear in CHEAP:
 *
 *   Flandrin:   ctx->lambda[k] = N^{2H} * sin(pi*k/(2N))^{-(2H+1)}
 *               Decreasing in k. Used for fBm covariance approximation.
 *
 *   Laplacian:  lambda_k = 4 * sin^2(pi*k/(2N))
 *               Increasing in k. Zero at DC. Diagonalizes discrete
 *               Laplacian with Neumann BCs.
 *
 * Each function documents which family it uses or expects.
 * ============================================================ */

/*
 * cheap_weights_laplacian — Compute discrete Laplacian eigenvalues.
 *
 *   lambda_out[k] = 4 * sin^2(pi*k / (2*n)),   k = 0, ..., n-1
 *
 * These are the eigenvalues of the Neumann-BC discrete Laplacian.
 * lambda_out[0] = 0 exactly (DC component).
 * Does NOT require a cheap_ctx.
 */
static inline int cheap_weights_laplacian(int n, double* restrict lambda_out)
{
    if (n < 2 || !lambda_out) return CHEAP_EINVAL;
    lambda_out[0] = 0.0;
    for (int k = 1; k < n; ++k) {
        double s = sin(M_PI * (double)k / (2.0 * (double)n));
        lambda_out[k] = 4.0 * s * s;
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(lambda_out, n);
    CHEAP_CONTRACT_NONDEC_OR_EDOM(lambda_out, n);
    return CHEAP_OK;
}

/*
 * cheap_weights_fractional — Fractional integration/differentiation weights.
 *
 *   weights_out[k] = (2 * sin(pi*k/(2*n)))^d
 *
 * d > 0: fractional differentiation.  d < 0: fractional integration.
 * d = 0: identity (all weights = 1).
 * DC (k=0) sin argument is clamped to CHEAP_EPS_LOG before pow.
 * Does NOT require a cheap_ctx.
 */
static inline int cheap_weights_fractional(int n, double d,
                                             double* restrict weights_out)
{
    if (n < 2 || !weights_out || !isfinite(d)) return CHEAP_EINVAL;
    for (int k = 0; k < n; ++k) {
        double s = sin(M_PI * (double)k / (2.0 * (double)n));
        if (s < CHEAP_EPS_LOG) s = CHEAP_EPS_LOG;
        weights_out[k] = pow(2.0 * s, d);
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/*
 * cheap_weights_laplacian_2d — Isotropic 2D Laplacian eigenvalues.
 *
 *   w[j*ny + k] = 4*sin^2(pi*j/(2*nx)) + 4*sin^2(pi*k/(2*ny))
 *
 * Neumann-BC discrete 2D Laplacian on an nx*ny grid, row-major layout.
 * w[0] = 0 exactly (DC). Zero-frequency axes contribute zero — i.e. the
 * j=0 row equals 4*sin^2(pi*k/(2*ny)), and the k=0 column likewise.
 * Does NOT require a cheap_ctx_2d.
 */
static inline int cheap_weights_laplacian_2d(int nx, int ny,
                                               double* restrict weights_out)
{
    if (nx < 2 || ny < 2 || !weights_out) return CHEAP_EINVAL;
    /* Precompute per-axis 1D Laplacian spectra, then sum. */
    for (int j = 0; j < nx; ++j) {
        double sx = sin(M_PI * (double)j / (2.0 * (double)nx));
        double lx = 4.0 * sx * sx;
        for (int k = 0; k < ny; ++k) {
            double sy = sin(M_PI * (double)k / (2.0 * (double)ny));
            weights_out[j * ny + k] = lx + 4.0 * sy * sy;
        }
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, nx * ny);
    return CHEAP_OK;
}

/*
 * cheap_weights_fractional_laplacian_2d — (-Delta)^alpha on a 2D grid.
 *
 *   w[j*ny + k] = ( 4*sin^2(pi*j/(2*nx)) + 4*sin^2(pi*k/(2*ny)) )^alpha
 *
 * alpha > 0: fractional Laplacian (diffusive). alpha < 0: fractional
 * inverse Laplacian (smoothing/integration). alpha = 0: identity.
 * The DC cell (j=k=0) is floored to CHEAP_EPS_LOG before pow() to
 * avoid 0^negative = inf when alpha < 0.
 * Does NOT require a cheap_ctx_2d.
 */
static inline int cheap_weights_fractional_laplacian_2d(int nx, int ny,
                                                          double alpha,
                                                          double* restrict weights_out)
{
    if (nx < 2 || ny < 2 || !weights_out || !isfinite(alpha)) return CHEAP_EINVAL;
    for (int j = 0; j < nx; ++j) {
        double sx = sin(M_PI * (double)j / (2.0 * (double)nx));
        double lx = 4.0 * sx * sx;
        for (int k = 0; k < ny; ++k) {
            double sy = sin(M_PI * (double)k / (2.0 * (double)ny));
            double v  = lx + 4.0 * sy * sy;
            if (v < CHEAP_EPS_LOG) v = CHEAP_EPS_LOG;
            weights_out[j * ny + k] = pow(v, alpha);
        }
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, nx * ny);
    return CHEAP_OK;
}

/*
 * cheap_weights_laplacian_3d — Isotropic 3D Laplacian eigenvalues.
 *
 *   w[(j*ny+k)*nz + l] = 4*sin^2(pi*j/(2*nx)) + 4*sin^2(pi*k/(2*ny))
 *                        + 4*sin^2(pi*l/(2*nz))
 *
 * Neumann-BC discrete 3D Laplacian on an nx*ny*nz grid, row-major layout.
 * w[0] = 0 exactly (DC). Does NOT require a cheap_ctx_3d.
 */
static inline int cheap_weights_laplacian_3d(int nx, int ny, int nz,
                                               double* restrict weights_out)
{
    if (nx < 2 || ny < 2 || nz < 2 || !weights_out) return CHEAP_EINVAL;
    for (int j = 0; j < nx; ++j) {
        double sx = sin(M_PI * (double)j / (2.0 * (double)nx));
        double lx = 4.0 * sx * sx;
        for (int k = 0; k < ny; ++k) {
            double sy = sin(M_PI * (double)k / (2.0 * (double)ny));
            double ly = 4.0 * sy * sy;
            for (int l = 0; l < nz; ++l) {
                double sz = sin(M_PI * (double)l / (2.0 * (double)nz));
                double lz = 4.0 * sz * sz;
                weights_out[(j * ny + k) * nz + l] = lx + ly + lz;
            }
        }
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, nx * ny * nz);
    return CHEAP_OK;
}

/*
 * cheap_weights_fractional_laplacian_3d — (-Delta)^alpha on a 3D grid.
 *
 *   w[(j*ny+k)*nz+l] = ( 4*sin^2(pi*j/(2*nx)) + 4*sin^2(pi*k/(2*ny))
 *                        + 4*sin^2(pi*l/(2*nz)) )^alpha
 *
 * alpha > 0: fractional Laplacian. alpha < 0: fractional inverse.
 * alpha = 0: identity. DC floored to CHEAP_EPS_LOG before pow().
 * Does NOT require a cheap_ctx_3d.
 */
static inline int cheap_weights_fractional_laplacian_3d(int nx, int ny, int nz,
                                                          double alpha,
                                                          double* restrict weights_out)
{
    if (nx < 2 || ny < 2 || nz < 2 || !weights_out || !isfinite(alpha)) return CHEAP_EINVAL;
    for (int j = 0; j < nx; ++j) {
        double sx = sin(M_PI * (double)j / (2.0 * (double)nx));
        double lx = 4.0 * sx * sx;
        for (int k = 0; k < ny; ++k) {
            double sy = sin(M_PI * (double)k / (2.0 * (double)ny));
            double ly = 4.0 * sy * sy;
            for (int l = 0; l < nz; ++l) {
                double sz = sin(M_PI * (double)l / (2.0 * (double)nz));
                double v  = lx + ly + 4.0 * sz * sz;
                if (v < CHEAP_EPS_LOG) v = CHEAP_EPS_LOG;
                weights_out[(j * ny + k) * nz + l] = pow(v, alpha);
            }
        }
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, nx * ny * nz);
    return CHEAP_OK;
}

/*
 * cheap_weights_kpca_hard — Hard spectral truncation (Kernel PCA).
 *
 *   weights_out[k] = (k < K) ? 1.0 : 0.0
 *
 * Retains the first K spectral components exactly.
 * K=0 zeroes everything; K=n is the identity.
 */
static inline int cheap_weights_kpca_hard(int n, int K,
                                            double* restrict weights_out)
{
    if (n < 2 || K < 0 || K > n || !weights_out) return CHEAP_EINVAL;
    for (int k = 0; k < K; ++k) weights_out[k] = 1.0;
    for (int k = K; k < n; ++k) weights_out[k] = 0.0;
    return CHEAP_OK;
}

/*
 * cheap_weights_kpca_soft — Soft spectral threshold (Kernel PCA).
 *
 *   weights_out[k] = max(0, 1 - lambda[K] / lambda[k])
 *
 * Uses Flandrin eigenvalues from ctx->lambda (decreasing in k).
 * Components with lambda[k] >> lambda[K] get weight ~1;
 * components with lambda[k] ~ lambda[K] get weight ~0.
 * Retains low-frequency (high-variance) components, consistent with PCA.
 */
static inline int cheap_weights_kpca_soft(const cheap_ctx* restrict ctx, int K,
                                            double* restrict weights_out)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    if (K < 0 || K >= ctx->n || !weights_out) return CHEAP_EINVAL;
    const int n = ctx->n;
    const double threshold = ctx->lambda[K];
    for (int k = 0; k < n; ++k) {
        if (ctx->lambda[k] < CHEAP_EPS_LAMBDA)
            weights_out[k] = 0.0;
        else
            weights_out[k] = fmax(0.0, 1.0 - threshold / ctx->lambda[k]);
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/*
 * cheap_weights_wiener — Wiener filter weights (Laplacian eigenvalues).
 *
 *   weights_out[k] = lambda_k / (lambda_k + sigma_sq)
 *
 * where lambda_k = 4*sin^2(pi*k/(2*n)) are Laplacian eigenvalues.
 * DC (k=0): weight = 0 (zero signal power at DC).
 * All weights in [0, 1), monotonically non-decreasing in k.
 */
static inline int cheap_weights_wiener(int n, double sigma_sq,
                                         double* restrict weights_out)
{
    if (n < 2 || sigma_sq <= 0.0 || !weights_out) return CHEAP_EINVAL;
    weights_out[0] = 0.0;
    for (int k = 1; k < n; ++k) {
        double s = sin(M_PI * (double)k / (2.0 * (double)n));
        double lk = 4.0 * s * s;
        weights_out[k] = lk / (lk + sigma_sq);
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    CHEAP_CONTRACT_NONDEC_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/*
 * cheap_weights_wiener_ev — Wiener filter with user-provided eigenvalues.
 *
 *   weights_out[k] = lambda[k] / (lambda[k] + sigma_sq)
 *
 * Caller provides eigenvalues (Flandrin, Laplacian, or empirical).
 */
static inline int cheap_weights_wiener_ev(int n,
                                            const double* restrict lambda,
                                            double sigma_sq,
                                            double* restrict weights_out)
{
    if (n < 2 || !lambda || sigma_sq <= 0.0 || !weights_out) return CHEAP_EINVAL;
    for (int k = 0; k < n; ++k)
        if (!isfinite(lambda[k])) return CHEAP_EDOM;
    int k = 0;
#if defined(CHEAP_SIMD_AVX2)
    {
        __m256d vss = _mm256_set1_pd(sigma_sq);
        __m256d vz  = _mm256_setzero_pd();
        for (; k + 4 <= n; k += 4) {
            __m256d vl = _mm256_loadu_pd(lambda + k);
            vl = _mm256_max_pd(vl, vz);
            __m256d denom = _mm256_add_pd(vl, vss);
            _mm256_storeu_pd(weights_out + k, _mm256_div_pd(vl, denom));
        }
    }
#elif defined(CHEAP_SIMD_NEON)
    {
        float64x2_t vss = vdupq_n_f64(sigma_sq);
        float64x2_t vz  = vdupq_n_f64(0.0);
        for (; k + 2 <= n; k += 2) {
            float64x2_t vl = vld1q_f64(lambda + k);
            vl = vmaxq_f64(vl, vz);
            float64x2_t denom = vaddq_f64(vl, vss);
            vst1q_f64(weights_out + k, vdivq_f64(vl, denom));
        }
    }
#endif
    for (; k < n; ++k) {
        double lk = fmax(lambda[k], 0.0);
        weights_out[k] = lk / (lk + sigma_sq);
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/*
 * cheap_weights_specnorm — Spectral normalization (Laplacian eigenvalues).
 *
 *   weights_out[k] = 1 / sqrt(lambda_k + eps)
 *
 * where lambda_k = 4*sin^2(pi*k/(2*n)). Implements covariance whitening
 * for operators diagonalized by the DCT. eps > 0 prevents singularity.
 * DC (k=0): weight = 1/sqrt(eps).
 */
static inline int cheap_weights_specnorm(int n, double eps,
                                           double* restrict weights_out)
{
    if (n < 2 || eps <= 0.0 || !weights_out) return CHEAP_EINVAL;
    weights_out[0] = 1.0 / sqrt(eps);
    for (int k = 1; k < n; ++k) {
        double s = sin(M_PI * (double)k / (2.0 * (double)n));
        double lk = 4.0 * s * s;
        weights_out[k] = 1.0 / sqrt(lk + eps);
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/*
 * cheap_weights_specnorm_ev — Spectral normalization with user eigenvalues.
 *
 *   weights_out[k] = 1 / sqrt(lambda[k] + eps)
 */
static inline int cheap_weights_specnorm_ev(int n,
                                              const double* restrict lambda,
                                              double eps,
                                              double* restrict weights_out)
{
    if (n < 2 || !lambda || eps <= 0.0 || !weights_out) return CHEAP_EINVAL;
    for (int k = 0; k < n; ++k)
        if (!isfinite(lambda[k])) return CHEAP_EDOM;
    int k = 0;
#if defined(CHEAP_SIMD_AVX2)
    {
        __m256d veps = _mm256_set1_pd(eps);
        __m256d vone = _mm256_set1_pd(1.0);
        __m256d vz   = _mm256_setzero_pd();
        for (; k + 4 <= n; k += 4) {
            __m256d vl = _mm256_loadu_pd(lambda + k);
            vl = _mm256_max_pd(vl, vz);
            __m256d s  = _mm256_sqrt_pd(_mm256_add_pd(vl, veps));
            _mm256_storeu_pd(weights_out + k, _mm256_div_pd(vone, s));
        }
    }
#elif defined(CHEAP_SIMD_NEON)
    {
        float64x2_t veps = vdupq_n_f64(eps);
        float64x2_t vone = vdupq_n_f64(1.0);
        float64x2_t vz   = vdupq_n_f64(0.0);
        for (; k + 2 <= n; k += 2) {
            float64x2_t vl = vld1q_f64(lambda + k);
            vl = vmaxq_f64(vl, vz);
            float64x2_t s  = vsqrtq_f64(vaddq_f64(vl, veps));
            vst1q_f64(weights_out + k, vdivq_f64(vone, s));
        }
    }
#endif
    for (; k < n; ++k) {
        double lk = fmax(lambda[k], 0.0);
        weights_out[k] = 1.0 / sqrt(lk + eps);
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/* ============================================================
 * Mandelbrot spectral weight — complex Gamma ratio
 *
 * Private helper: Lanczos approximation for complex log-Gamma.
 * Uses g=7, N=9 Godfrey coefficients for ~15-digit accuracy.
 * ============================================================ */

static inline void cheap__cmul(double ar, double ai, double br, double bi,
                                double* restrict cr,
                                double* restrict ci)
{
    *cr = ar * br - ai * bi;
    *ci = ar * bi + ai * br;
}

static inline void cheap__cdiv(double ar, double ai, double br, double bi,
                                 double* restrict cr, double* restrict ci)
{
    double d = br * br + bi * bi;
    if (d < CHEAP_EPS_DIV) d = CHEAP_EPS_DIV;
    *cr = (ar * br + ai * bi) / d;
    *ci = (ai * br - ar * bi) / d;
}

static inline void cheap__clog(double ar, double ai,
                                 double* restrict cr, double* restrict ci)
{
    double mag = sqrt(ar * ar + ai * ai);
    if (mag < CHEAP_EPS_DIV) mag = CHEAP_EPS_DIV;
    *cr = log(mag);
    *ci = atan2(ai, ar);
}

/*
 * cheap__clgamma — Complex log-Gamma via Lanczos approximation.
 *
 * Computes ln(Gamma(re + i*im)) = (*out_re) + i*(*out_im).
 * For re < 0.5, uses the reflection formula:
 *   lnGamma(z) = ln(pi / sin(pi*z)) - lnGamma(1 - z)
 */
static inline void cheap__clgamma(double re, double im,
                                    double* restrict out_re,
                                    double* restrict out_im)
{
    static const double g = 7.0;
    static const double c[9] = {
        0.99999999999980993,
        676.5203681218851,
       -1259.1392167224028,
        771.32342877765313,
       -176.61502916214059,
        12.507343278686905,
       -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    };

    if (re < 0.5) {
        /* Reflection: lnGamma(z) = ln(pi/sin(pi*z)) - lnGamma(1-z) */
        double sin_re, sin_im;
        sin_re = sin(M_PI * re) * cosh(M_PI * im);
        sin_im = cos(M_PI * re) * sinh(M_PI * im);

        double log_sin_r, log_sin_i;
        cheap__clog(sin_re, sin_im, &log_sin_r, &log_sin_i);

        double lg1r, lg1i;
        cheap__clgamma(1.0 - re, -im, &lg1r, &lg1i);

        *out_re = log(M_PI) - log_sin_r - lg1r;
        *out_im =            - log_sin_i - lg1i;
        return;
    }

    /* Lanczos series for Re(z) >= 0.5 */
    re -= 1.0;

    /* x = c[0] + sum_{i=1}^{8} c[i] / (z + i) */
    double xr = c[0], xi = 0.0;
    for (int i = 1; i <= 8; ++i) {
        double dr = re + (double)i;
        double di = im;
        double qr, qi;
        cheap__cdiv(c[i], 0.0, dr, di, &qr, &qi);
        xr += qr;
        xi += qi;
    }

    /* t = z + g + 0.5 */
    double tr = re + g + 0.5;
    double ti = im;

    /* lnGamma = 0.5*ln(2*pi) + (z + 0.5)*ln(t) - t + ln(x) */
    double log_t_r, log_t_i;
    cheap__clog(tr, ti, &log_t_r, &log_t_i);

    /* (z + 0.5) * ln(t) */
    double zph_r = re + 0.5, zph_i = im;
    double term1_r, term1_i;
    cheap__cmul(zph_r, zph_i, log_t_r, log_t_i, &term1_r, &term1_i);

    /* ln(x) */
    double log_x_r, log_x_i;
    cheap__clog(xr, xi, &log_x_r, &log_x_i);

    *out_re = 0.5 * log(2.0 * M_PI) + term1_r - tr + log_x_r;
    *out_im =                           term1_i - ti + log_x_i;
}

/*
 * cheap__clgamma_re_noreflect — batched real-part log-Gamma.
 * Computes Re(lnGamma(re + i*im[lane])) for LANES lanes in
 * parallel via Lanczos. Requires re >= 0.5 (no reflection).
 * Real part only: Mandelbrot needs Re(ln Γ_num - ln Γ_den).
 *
 * Used by cheap_weights_mandelbrot for whichever of H, 1-H is
 * >= 0.5. The other side falls back to scalar cheap__clgamma.
 */
#if defined(CHEAP_SIMD_AVX2)
static inline __m256d cheap__clgamma_re_noreflect_avx2(double re, __m256d vim)
{
    static const double c[9] = {
        0.99999999999980993,  676.5203681218851,   -1259.1392167224028,
        771.32342877765313,  -176.61502916214059,   12.507343278686905,
       -0.13857109526572012,  9.9843695780195716e-6, 1.5056327351493116e-7
    };
    const double g = 7.0;
    const double zm1 = re - 1.0;
    __m256d xr = _mm256_set1_pd(c[0]);
    __m256d xi = _mm256_setzero_pd();
    __m256d veps = _mm256_set1_pd(CHEAP_EPS_DIV);
    for (int i = 1; i <= 8; ++i) {
        double dr = zm1 + (double)i;
        __m256d vdr = _mm256_set1_pd(dr);
        __m256d vdi = vim;
        __m256d denom = _mm256_add_pd(_mm256_mul_pd(vdr, vdr),
                                       _mm256_mul_pd(vdi, vdi));
        denom = _mm256_max_pd(denom, veps);
        __m256d ci = _mm256_set1_pd(c[i]);
        __m256d qr = _mm256_div_pd(_mm256_mul_pd(ci, vdr), denom);
        __m256d qi = _mm256_div_pd(_mm256_sub_pd(_mm256_setzero_pd(),
                                                  _mm256_mul_pd(ci, vdi)), denom);
        xr = _mm256_add_pd(xr, qr);
        xi = _mm256_add_pd(xi, qi);
    }
    __m256d vtr = _mm256_set1_pd(zm1 + g + 0.5);
    __m256d vti = vim;
    /* log|t| = 0.5 * log(tr^2 + ti^2) */
    __m256d mag_t2 = _mm256_add_pd(_mm256_mul_pd(vtr, vtr),
                                    _mm256_mul_pd(vti, vti));
    mag_t2 = _mm256_max_pd(mag_t2, veps);
    /* log|x| = 0.5 * log(xr^2 + xi^2) */
    __m256d mag_x2 = _mm256_add_pd(_mm256_mul_pd(xr, xr),
                                    _mm256_mul_pd(xi, xi));
    mag_x2 = _mm256_max_pd(mag_x2, veps);
    double halfln2pi = 0.5 * log(2.0 * M_PI);
    double zph_r = zm1 + 0.5;
    /* Re((z+0.5)*ln(t)) = zph_r * 0.5*log(|t|^2) - zph_i * atan2(ti, tr)
       zph_i = vim (lane-varying); atan2 is scalar-only — extract. */
    double tmp_t[4], tmp_x[4], tmp_im[4];
    _mm256_storeu_pd(tmp_t, mag_t2);
    _mm256_storeu_pd(tmp_x, mag_x2);
    _mm256_storeu_pd(tmp_im, vim);
    double tr_scalar = zm1 + g + 0.5;
    double out[4];
    for (int l = 0; l < 4; ++l) {
        double log_t_r = 0.5 * log(tmp_t[l]);
        double log_t_i = atan2(tmp_im[l], tr_scalar);
        double log_x_r = 0.5 * log(tmp_x[l]);
        double term1_r = zph_r * log_t_r - tmp_im[l] * log_t_i;
        out[l] = halfln2pi + term1_r - tr_scalar + log_x_r;
    }
    return _mm256_loadu_pd(out);
}
#endif

#if defined(CHEAP_SIMD_NEON)
static inline float64x2_t cheap__clgamma_re_noreflect_neon(double re, float64x2_t vim)
{
    static const double c[9] = {
        0.99999999999980993,  676.5203681218851,   -1259.1392167224028,
        771.32342877765313,  -176.61502916214059,   12.507343278686905,
       -0.13857109526572012,  9.9843695780195716e-6, 1.5056327351493116e-7
    };
    const double g = 7.0;
    const double zm1 = re - 1.0;
    float64x2_t xr = vdupq_n_f64(c[0]);
    float64x2_t xi = vdupq_n_f64(0.0);
    float64x2_t veps = vdupq_n_f64(CHEAP_EPS_DIV);
    for (int i = 1; i <= 8; ++i) {
        double dr = zm1 + (double)i;
        float64x2_t vdr = vdupq_n_f64(dr);
        float64x2_t vdi = vim;
        float64x2_t denom = vaddq_f64(vmulq_f64(vdr, vdr),
                                        vmulq_f64(vdi, vdi));
        denom = vmaxq_f64(denom, veps);
        float64x2_t ci = vdupq_n_f64(c[i]);
        float64x2_t qr = vdivq_f64(vmulq_f64(ci, vdr), denom);
        float64x2_t qi = vdivq_f64(vnegq_f64(vmulq_f64(ci, vdi)), denom);
        xr = vaddq_f64(xr, qr);
        xi = vaddq_f64(xi, qi);
    }
    float64x2_t mag_t2 = vaddq_f64(vdupq_n_f64((zm1 + g + 0.5) * (zm1 + g + 0.5)),
                                     vmulq_f64(vim, vim));
    mag_t2 = vmaxq_f64(mag_t2, veps);
    float64x2_t mag_x2 = vaddq_f64(vmulq_f64(xr, xr), vmulq_f64(xi, xi));
    mag_x2 = vmaxq_f64(mag_x2, veps);
    double halfln2pi = 0.5 * log(2.0 * M_PI);
    double zph_r = zm1 + 0.5;
    double tr_scalar = zm1 + g + 0.5;
    double tmp_t[2], tmp_x[2], tmp_im[2];
    vst1q_f64(tmp_t, mag_t2);
    vst1q_f64(tmp_x, mag_x2);
    vst1q_f64(tmp_im, vim);
    double out[2];
    for (int l = 0; l < 2; ++l) {
        double log_t_r = 0.5 * log(tmp_t[l]);
        double log_t_i = atan2(tmp_im[l], tr_scalar);
        double log_x_r = 0.5 * log(tmp_x[l]);
        double term1_r = zph_r * log_t_r - tmp_im[l] * log_t_i;
        out[l] = halfln2pi + term1_r - tr_scalar + log_x_r;
    }
    return vld1q_f64(out);
}
#endif

/*
 * cheap_weights_mandelbrot — Mandelbrot multifractal spectral weights.
 *
 *   weights_out[k] = |Gamma(H + i*tau_k) / Gamma(1-H + i*tau_k)|
 *
 * where tau_k = pi*k/n. H must be in (0, 1).
 * At H = 0.5, all weights equal 1.0 (symmetry).
 * Computed in log-space via Lanczos complex log-Gamma.
 * AVX2/NEON accelerated when H != 0.5: the Lanczos sum runs
 * 2 (NEON) or 4 (AVX2) k-values in parallel on the side of
 * the ratio with Re(z) >= 0.5; the reflection side stays scalar.
 */
static inline int cheap_weights_mandelbrot(int n, double H,
                                             double* restrict weights_out)
{
    if (n < 2 || H <= 0.0 || H >= 1.0 || !weights_out) return CHEAP_EINVAL;

    /* k = 0: real Gamma ratio */
    weights_out[0] = exp(lgamma(H) - lgamma(1.0 - H));

    const double inv_n = 1.0 / (double)n;
    int k = 1;
#if defined(CHEAP_SIMD_AVX2) || defined(CHEAP_SIMD_NEON)
    /*
     * Identify which side (H or 1-H) is the non-reflection side
     * (Re(z) >= 0.5). Vectorize that side via batched Lanczos;
     * the other side stays scalar (reflection branch is branchy).
     */
    const double re_noref  = (H >= 0.5) ? H : 1.0 - H;
    const double re_ref    = (H >= 0.5) ? 1.0 - H : H;
    const int noref_is_num = (H >= 0.5);
#endif
#if defined(CHEAP_SIMD_AVX2)
    for (; k + 4 <= n; k += 4) {
        double taus_arr[4] = {
            M_PI * (double)(k    ) * inv_n,
            M_PI * (double)(k + 1) * inv_n,
            M_PI * (double)(k + 2) * inv_n,
            M_PI * (double)(k + 3) * inv_n
        };
        __m256d vim = _mm256_loadu_pd(taus_arr);
        __m256d vre_noref = cheap__clgamma_re_noreflect_avx2(re_noref, vim);
        double noref[4];
        _mm256_storeu_pd(noref, vre_noref);
        for (int l = 0; l < 4; ++l) {
            double lg_ref_re, lg_ref_im;
            cheap__clgamma(re_ref, taus_arr[l], &lg_ref_re, &lg_ref_im);
            double diff = noref_is_num ? (noref[l] - lg_ref_re)
                                        : (lg_ref_re - noref[l]);
            weights_out[k + l] = exp(diff);
        }
    }
#elif defined(CHEAP_SIMD_NEON)
    for (; k + 2 <= n; k += 2) {
        double taus_arr[2] = {
            M_PI * (double)(k    ) * inv_n,
            M_PI * (double)(k + 1) * inv_n
        };
        float64x2_t vim = vld1q_f64(taus_arr);
        float64x2_t vre_noref = cheap__clgamma_re_noreflect_neon(re_noref, vim);
        double noref[2];
        vst1q_f64(noref, vre_noref);
        for (int l = 0; l < 2; ++l) {
            double lg_ref_re, lg_ref_im;
            cheap__clgamma(re_ref, taus_arr[l], &lg_ref_re, &lg_ref_im);
            double diff = noref_is_num ? (noref[l] - lg_ref_re)
                                        : (lg_ref_re - noref[l]);
            weights_out[k + l] = exp(diff);
        }
    }
#endif
    for (; k < n; ++k) {
        double tau = M_PI * (double)k * inv_n;
        double lg_num_re, lg_num_im;
        double lg_den_re, lg_den_im;
        cheap__clgamma(H, tau, &lg_num_re, &lg_num_im);
        cheap__clgamma(1.0 - H, tau, &lg_den_re, &lg_den_im);
        weights_out[k] = exp(lg_num_re - lg_den_re);
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    /*
     * H == 0.5 identity: |Gamma(0.5+it)/Gamma(0.5+it)| = 1 exactly.
     * Any drift here is a numerical bug in the Lanczos path.
     */
#ifdef CHEAP_DEBUG_CONTRACTS
    if (fabs(H - 0.5) < 1e-15) {
        for (int k = 0; k < n; ++k)
            if (fabs(weights_out[k] - 1.0) > 1e-12) return CHEAP_EDOM;
    }
#endif
    return CHEAP_OK;
}

/* ============================================================
 * RMT Denoising — Marchenko-Pastur thresholding
 * ============================================================ */

/*
 * cheap_weights_rmt_hard — Hard Marchenko-Pastur thresholding.
 *
 *   lambda_plus = sigma_sq * (1 + sqrt(c))^2
 *   weights_out[k] = (lambda[k] > lambda_plus) ? lambda[k] : 0
 *
 * Accepts user-provided eigenvalues (any family).
 * c = N/p is the aspect ratio (number of samples / dimension).
 */
static inline int cheap_weights_rmt_hard(const double* restrict lambda, int n,
                                           double sigma_sq, double c,
                                           double* restrict weights_out)
{
    if (n < 2 || !lambda || sigma_sq <= 0.0 || c <= 0.0 || !weights_out)
        return CHEAP_EINVAL;
    for (int k = 0; k < n; ++k)
        if (!isfinite(lambda[k])) return CHEAP_EDOM;
    const double sc = sqrt(c);
    const double lambda_plus = sigma_sq * (1.0 + sc) * (1.0 + sc);
    int k = 0;
#if defined(CHEAP_SIMD_AVX2)
    __m256d vth = _mm256_set1_pd(lambda_plus);
    __m256d vz  = _mm256_setzero_pd();
    for (; k + 4 <= n; k += 4) {
        __m256d vl = _mm256_loadu_pd(lambda + k);
        /* mask = lambda > lambda_plus */
        __m256d mask = _mm256_cmp_pd(vl, vth, _CMP_GT_OQ);
        _mm256_storeu_pd(weights_out + k, _mm256_blendv_pd(vz, vl, mask));
    }
#elif defined(CHEAP_SIMD_NEON)
    float64x2_t vth = vdupq_n_f64(lambda_plus);
    float64x2_t vz  = vdupq_n_f64(0.0);
    for (; k + 2 <= n; k += 2) {
        float64x2_t vl = vld1q_f64(lambda + k);
        uint64x2_t mask = vcgtq_f64(vl, vth);
        vst1q_f64(weights_out + k, vbslq_f64(mask, vl, vz));
    }
#endif
    for (; k < n; ++k)
        weights_out[k] = (lambda[k] > lambda_plus) ? lambda[k] : 0.0;
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/*
 * cheap_weights_rmt_shrink — Optimal nonlinear shrinkage (Donoho-Gavish).
 *
 * For lambda[k] <= lambda_plus: weights_out[k] = 0.
 * For lambda[k] >  lambda_plus:
 *   l = lambda[k] / sigma_sq
 *   weights_out[k] = lambda[k] * sqrt((l - lp)*(l - lm)) / l
 *
 * where lp = (1+sqrt(c))^2, lm = (1-sqrt(c))^2.
 * This is the asymptotically optimal Frobenius-norm shrinkage.
 */
static inline int cheap_weights_rmt_shrink(const double* restrict lambda, int n,
                                             double sigma_sq, double c,
                                             double* restrict weights_out)
{
    if (n < 2 || !lambda || sigma_sq <= 0.0 || c <= 0.0 || !weights_out)
        return CHEAP_EINVAL;
    for (int k = 0; k < n; ++k)
        if (!isfinite(lambda[k])) return CHEAP_EDOM;
    const double sc = sqrt(c);
    const double lp = (1.0 + sc) * (1.0 + sc);
    const double lm = (1.0 - sc) * (1.0 - sc);
    const double lambda_plus = sigma_sq * lp;
    const double inv_sigma_sq = 1.0 / sigma_sq;
    int k = 0;
#if defined(CHEAP_SIMD_AVX2)
    {
        __m256d vth  = _mm256_set1_pd(lambda_plus);
        __m256d vz   = _mm256_setzero_pd();
        __m256d visg = _mm256_set1_pd(inv_sigma_sq);
        __m256d vlp  = _mm256_set1_pd(lp);
        __m256d vlm  = _mm256_set1_pd(lm);
        for (; k + 4 <= n; k += 4) {
            __m256d vl   = _mm256_loadu_pd(lambda + k);
            __m256d mask = _mm256_cmp_pd(vl, vth, _CMP_GT_OQ);
            __m256d vx   = _mm256_mul_pd(vl, visg);            /* l = lambda/sigma_sq */
            __m256d a    = _mm256_sub_pd(vx, vlp);
            __m256d b    = _mm256_sub_pd(vx, vlm);
            __m256d prod = _mm256_mul_pd(a, b);
            prod = _mm256_max_pd(prod, vz);                     /* fmax(0, ...) */
            __m256d fac  = _mm256_sqrt_pd(prod);
            __m256d res  = _mm256_div_pd(_mm256_mul_pd(vl, fac), vx); /* lambda*fac/l */
            _mm256_storeu_pd(weights_out + k, _mm256_blendv_pd(vz, res, mask));
        }
    }
#elif defined(CHEAP_SIMD_NEON)
    {
        float64x2_t vth  = vdupq_n_f64(lambda_plus);
        float64x2_t vz   = vdupq_n_f64(0.0);
        float64x2_t visg = vdupq_n_f64(inv_sigma_sq);
        float64x2_t vlp  = vdupq_n_f64(lp);
        float64x2_t vlm  = vdupq_n_f64(lm);
        for (; k + 2 <= n; k += 2) {
            float64x2_t vl   = vld1q_f64(lambda + k);
            uint64x2_t  mask = vcgtq_f64(vl, vth);
            float64x2_t vx   = vmulq_f64(vl, visg);
            float64x2_t a    = vsubq_f64(vx, vlp);
            float64x2_t b    = vsubq_f64(vx, vlm);
            float64x2_t prod = vmulq_f64(a, b);
            prod = vmaxq_f64(prod, vz);
            float64x2_t fac  = vsqrtq_f64(prod);
            float64x2_t res  = vdivq_f64(vmulq_f64(vl, fac), vx);
            vst1q_f64(weights_out + k, vbslq_f64(mask, res, vz));
        }
    }
#endif
    for (; k < n; ++k) {
        if (lambda[k] <= lambda_plus) {
            weights_out[k] = 0.0;
        } else {
            double l = lambda[k] * inv_sigma_sq;
            double factor = sqrt(fmax(0.0, (l - lp) * (l - lm)));
            weights_out[k] = lambda[k] * factor / l;
        }
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/*
 * cheap_weights_wiener_deconv_ev — Wiener deconvolution weights for circulant PSF.
 *
 *   weights_out[k] = λ_psf[k] / (λ_psf[k]² + noise_power)
 *
 * Frequency-domain Wiener filter for y = PSF * x + noise where the blur
 * operator is diagonalized by the same DCT-II basis as the covariance.
 * λ_psf[] = DCT-II(first column of the symmetric circulant Toeplitz PSF).
 * Obtain via cheap_toeplitz_eigenvalues() or cheap_init_from_toeplitz().
 *
 * noise_power = σ_noise² / σ_signal² (or just σ_noise² if you normalize
 * signal power to 1). Bounded |w_k| ≤ 1/(2√noise_power). Perfectly stable.
 *
 * Vectorized AVX2 (4×f64) / NEON (2×f64). Zero UB. Full contracts under
 * CHEAP_DEBUG_CONTRACTS. Scalar tail is identical to vector path.
 *
 * Usage (1D example):
 *   double psf_eig[N];
 *   cheap_toeplitz_eigenvalues(&ctx, psf_first_col, psf_eig);
 *   double weights[N];
 *   cheap_weights_wiener_deconv_ev(N, psf_eig, noise_power, weights);
 *   cheap_apply(&ctx, blurred, weights, restored);
 *
 * Same call works for 2D/3D — just pass n = nx*ny (or nx*ny*nz) and the
 * flat row-major eigenvalue grid of the PSF.
 */
static inline int cheap_weights_wiener_deconv_ev(int n,
    const double* restrict psf_eigenvalues,
    double noise_power,
    double* restrict weights_out)
{
    if (n < 2 || !psf_eigenvalues || noise_power < 0.0 || !weights_out)
        return CHEAP_EINVAL;

    /* Input contract */
    for (int i = 0; i < n; ++i)
        if (!isfinite(psf_eigenvalues[i])) return CHEAP_EDOM;

    int k = 0;
#if defined(CHEAP_SIMD_AVX2)
    {
        __m256d vnp  = _mm256_set1_pd(noise_power);
        __m256d veps = _mm256_set1_pd(CHEAP_EPS_DIV);
        for (; k + 4 <= n; k += 4) {
            __m256d lam  = _mm256_loadu_pd(psf_eigenvalues + k);
            __m256d lam2 = _mm256_mul_pd(lam, lam);
            __m256d den  = _mm256_add_pd(lam2, vnp);
            den = _mm256_max_pd(den, veps);               /* prevent div0/NaN */
            __m256d w    = _mm256_div_pd(lam, den);
            _mm256_storeu_pd(weights_out + k, w);
        }
    }
#elif defined(CHEAP_SIMD_NEON)
    {
        float64x2_t vnp  = vdupq_n_f64(noise_power);
        float64x2_t veps = vdupq_n_f64(CHEAP_EPS_DIV);
        for (; k + 2 <= n; k += 2) {
            float64x2_t lam  = vld1q_f64(psf_eigenvalues + k);
            float64x2_t lam2 = vmulq_f64(lam, lam);
            float64x2_t den  = vaddq_f64(lam2, vnp);
            den = vmaxq_f64(den, veps);
            float64x2_t w    = vdivq_f64(lam, den);
            vst1q_f64(weights_out + k, w);
        }
    }
#endif
    /* Scalar tail — identical arithmetic to vector path */
    for (; k < n; ++k) {
        double lam = psf_eigenvalues[k];
        double den = lam * lam + noise_power;
        den = fmax(den, CHEAP_EPS_DIV);
        weights_out[k] = lam / den;
    }

    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/* ============================================================
 * v0.3.0-tensor-weights — GRF, Heat, Biharmonic, Poisson,
 * Higher-Order Tikhonov weight constructors
 * ============================================================ */

/*
 * cheap__build_laplacian_flat — fill out[k] = 4*sin²(π*k/(2*n)), out[0]=0.
 * Private helper. Caller must validate n >= 2 and out != NULL.
 */
static inline void cheap__build_laplacian_flat(int n, double* restrict out)
{
    out[0] = 0.0;
    for (int k = 1; k < n; ++k) {
        double s = sin(M_PI * (double)k / (2.0 * (double)n));
        out[k] = 4.0 * s * s;
    }
}

/*
 * cheap_weights_laplacian_ev — flat Laplacian eigenvalues (_ev convention).
 *
 *   weights_out[k] = 4 * sin²(π*k / (2*n)),   k = 0..n-1
 *
 * Identical formula to cheap_weights_laplacian. This alias exists so the
 * output can be passed directly to matern_ev, poisson_ev, biharmonic_ev,
 * etc. without the caller needing to know it came from the Laplacian.
 *
 * Works with any flat buffer of length n — pass n = nx*ny for 2D grids
 * if you want the 1D Laplacian spectrum tiled flat (unusual; prefer
 * cheap_weights_laplacian_2d for proper tensor-product 2D spectra).
 *
 * Usage:
 *   double mu[N];
 *   cheap_weights_laplacian_ev(N, mu);
 *   double w[N];
 *   cheap_weights_matern_ev(N, mu, 1.0, 1.5, w);  // Matérn-1.5 GRF weights
 */
static inline int cheap_weights_laplacian_ev(int n, double* restrict weights_out)
{
    if (n < 2 || !weights_out) return CHEAP_EINVAL;
    cheap__build_laplacian_flat(n, weights_out);
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    CHEAP_CONTRACT_NONDEC_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/*
 * cheap_weights_matern_ev — Matérn covariance spectral weights.
 *
 *   weights_out[k] = (κ² + μ[k])^(-ν),   k = 0..n-1
 *
 * Spectral density of the Matérn-ν covariance kernel when μ[] are the
 * spatial-frequency squared values (e.g. Laplacian eigenvalues). Sampling
 * white noise z, applying cheap_apply with these weights, and normalizing
 * gives a Matérn-ν Gaussian random field (SPDE formulation, Lindgren et al.
 * 2011). κ > 0 is the inverse correlation length; ν > 0 is the smoothness.
 *
 * DC (μ[0] = 0 from Laplacian): base = κ² > 0, so w[0] = κ^(-2ν). Finite.
 *
 * Works with both Flandrin and Laplacian eigenvalue bases.
 *
 * Usage:
 *   double mu[N], w[N];
 *   cheap_weights_laplacian_ev(N, mu);
 *   cheap_weights_matern_ev(N, mu, 1.0, 1.5, w);  // Matérn-1.5
 *   cheap_apply(&ctx, white_noise, w, grf_sample);
 */
static inline int cheap_weights_matern_ev(int n,
                                           const double* restrict mu,
                                           double kappa, double nu,
                                           double* restrict weights_out)
{
    if (n < 2 || !mu || kappa <= 0.0 || nu <= 0.0 || !weights_out)
        return CHEAP_EINVAL;
    for (int i = 0; i < n; ++i)
        if (!isfinite(mu[i])) return CHEAP_EDOM;
    double kk = kappa * kappa;
    for (int k = 0; k < n; ++k) {
        double base = kk + mu[k];
        if (base < CHEAP_EPS_LOG) base = CHEAP_EPS_LOG;
        weights_out[k] = pow(base, -nu);
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/*
 * cheap_weights_matern_2d — isotropic 2D Matérn covariance weights.
 *
 *   weights_out[j*ny + k] = (κ² + lx[j] + ly[k])^(-ν)
 *
 * where lx[j] = 4*sin²(πj/(2*nx)), ly[k] = 4*sin²(πk/(2*ny)).
 * Equivalent to calling cheap_weights_laplacian_2d then matern_ev on the
 * result. DC: weights_out[0] = κ^(-2ν) — finite, positive.
 *
 * Row-major: weights_out[j*ny + k], j in [0,nx), k in [0,ny).
 *
 * Usage:
 *   double w[NX*NY];
 *   cheap_weights_matern_2d(NX, NY, 1.0, 1.5, w);
 *   cheap_apply_2d(&ctx2d, noise, w, grf_sample);
 */
static inline int cheap_weights_matern_2d(int nx, int ny,
                                           double kappa, double nu,
                                           double* restrict weights_out)
{
    if (nx < 2 || ny < 2 || kappa <= 0.0 || nu <= 0.0 || !weights_out)
        return CHEAP_EINVAL;
    double kk = kappa * kappa;
    for (int j = 0; j < nx; ++j) {
        double sx = sin(M_PI * (double)j / (2.0 * (double)nx));
        double lx = 4.0 * sx * sx;
        for (int k = 0; k < ny; ++k) {
            double sy = sin(M_PI * (double)k / (2.0 * (double)ny));
            double base = kk + lx + 4.0 * sy * sy;
            if (base < CHEAP_EPS_LOG) base = CHEAP_EPS_LOG;
            weights_out[j * ny + k] = pow(base, -nu);
        }
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, nx * ny);
    return CHEAP_OK;
}

/*
 * cheap_weights_matern_3d — isotropic 3D Matérn covariance weights.
 *
 *   weights_out[(j*ny+k)*nz + l] = (κ² + lx[j] + ly[k] + lz[l])^(-ν)
 *
 * Row-major: (j*ny+k)*nz+l. DC: κ^(-2ν) — finite, positive.
 *
 * Usage:
 *   double w[NX*NY*NZ];
 *   cheap_weights_matern_3d(NX, NY, NZ, 1.0, 2.5, w);
 */
static inline int cheap_weights_matern_3d(int nx, int ny, int nz,
                                           double kappa, double nu,
                                           double* restrict weights_out)
{
    if (nx < 2 || ny < 2 || nz < 2 || kappa <= 0.0 || nu <= 0.0 || !weights_out)
        return CHEAP_EINVAL;
    double kk = kappa * kappa;
    for (int j = 0; j < nx; ++j) {
        double sx = sin(M_PI * (double)j / (2.0 * (double)nx));
        double lx = 4.0 * sx * sx;
        for (int k = 0; k < ny; ++k) {
            double sy = sin(M_PI * (double)k / (2.0 * (double)ny));
            double ly = 4.0 * sy * sy;
            for (int l = 0; l < nz; ++l) {
                double sz = sin(M_PI * (double)l / (2.0 * (double)nz));
                double base = kk + lx + ly + 4.0 * sz * sz;
                if (base < CHEAP_EPS_LOG) base = CHEAP_EPS_LOG;
                weights_out[(j * ny + k) * nz + l] = pow(base, -nu);
            }
        }
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, nx * ny * nz);
    return CHEAP_OK;
}

/*
 * cheap_weights_anisotropic_matern_2d — anisotropic 2D Matérn weights.
 *
 *   weights_out[j*ny + k] = (κ_x²·lx[j] + κ_y²·ly[k] + CHEAP_EPS_LOG)^(-ν)
 *
 * Per-axis inverse length scales κ_x, κ_y allow directional anisotropy.
 * CHEAP_EPS_LOG (1e-12) regularizes the DC bin: w[0] ≈ (1e-12)^(-ν), which
 * is large but finite. Zero it after the call if you want DC suppressed.
 *
 * Row-major: weights_out[j*ny + k].
 *
 * Usage:
 *   double w[NX*NY];
 *   cheap_weights_anisotropic_matern_2d(NX, NY, 2.0, 0.5, 1.5, w);
 *   w[0] = 0.0;  // suppress DC if desired
 */
static inline int cheap_weights_anisotropic_matern_2d(int nx, int ny,
                                                        double kappa_x,
                                                        double kappa_y,
                                                        double nu,
                                                        double* restrict weights_out)
{
    if (nx < 2 || ny < 2 || kappa_x <= 0.0 || kappa_y <= 0.0 || nu <= 0.0
        || !weights_out)
        return CHEAP_EINVAL;
    double kkx = kappa_x * kappa_x;
    double kky = kappa_y * kappa_y;
    for (int j = 0; j < nx; ++j) {
        double sx = sin(M_PI * (double)j / (2.0 * (double)nx));
        double lx = 4.0 * sx * sx;
        for (int k = 0; k < ny; ++k) {
            double sy = sin(M_PI * (double)k / (2.0 * (double)ny));
            double base = kkx * lx + kky * 4.0 * sy * sy + CHEAP_EPS_LOG;
            weights_out[j * ny + k] = pow(base, -nu);
        }
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, nx * ny);
    return CHEAP_OK;
}

/*
 * cheap_weights_anisotropic_matern_3d — anisotropic 3D Matérn weights.
 *
 *   weights_out[(j*ny+k)*nz+l] =
 *       (κ_x²·lx[j] + κ_y²·ly[k] + κ_z²·lz[l] + CHEAP_EPS_LOG)^(-ν)
 *
 * DC: w[0] ≈ (CHEAP_EPS_LOG)^(-ν) — large but finite. Zero if desired.
 * Row-major: (j*ny+k)*nz+l.
 */
static inline int cheap_weights_anisotropic_matern_3d(int nx, int ny, int nz,
                                                        double kappa_x,
                                                        double kappa_y,
                                                        double kappa_z,
                                                        double nu,
                                                        double* restrict weights_out)
{
    if (nx < 2 || ny < 2 || nz < 2 || kappa_x <= 0.0 || kappa_y <= 0.0
        || kappa_z <= 0.0 || nu <= 0.0 || !weights_out)
        return CHEAP_EINVAL;
    double kkx = kappa_x * kappa_x;
    double kky = kappa_y * kappa_y;
    double kkz = kappa_z * kappa_z;
    for (int j = 0; j < nx; ++j) {
        double sx = sin(M_PI * (double)j / (2.0 * (double)nx));
        double lx = 4.0 * sx * sx;
        for (int k = 0; k < ny; ++k) {
            double sy = sin(M_PI * (double)k / (2.0 * (double)ny));
            double ly = 4.0 * sy * sy;
            for (int l = 0; l < nz; ++l) {
                double sz = sin(M_PI * (double)l / (2.0 * (double)nz));
                double base = kkx * lx + kky * ly + kkz * 4.0 * sz * sz
                              + CHEAP_EPS_LOG;
                weights_out[(j * ny + k) * nz + l] = pow(base, -nu);
            }
        }
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, nx * ny * nz);
    return CHEAP_OK;
}

/*
 * cheap_weights_heat_propagator_ev — heat equation propagator weights.
 *
 *   weights_out[k] = exp(-t · μ[k]),   k = 0..n-1
 *
 * Spectral representation of the heat kernel e^(t·Δ). At t=0 this is the
 * identity; as t→∞ all high-frequency components decay. The DC component
 * (μ[0]=0) satisfies exp(0)=1.0 — DC is always preserved. Attenuate it
 * manually after the call if the application requires mean removal.
 *
 * Connection to Sinkhorn: the Gibbs kernel exp(-λ_k/ε) is a heat propagator
 * with t = 1/ε on the Flandrin spectrum (μ = ctx->lambda).
 *
 * Works with both Flandrin and Laplacian bases. t > 0 required.
 *
 * Usage:
 *   double mu[N], w[N];
 *   cheap_weights_laplacian_ev(N, mu);
 *   cheap_weights_heat_propagator_ev(N, mu, 0.1, w);  // t=0.1
 *   cheap_apply(&ctx, field, w, smoothed);
 */
static inline int cheap_weights_heat_propagator_ev(int n,
                                                    const double* restrict mu,
                                                    double t,
                                                    double* restrict weights_out)
{
    if (n < 2 || !mu || t <= 0.0 || !weights_out) return CHEAP_EINVAL;
    for (int i = 0; i < n; ++i)
        if (!isfinite(mu[i])) return CHEAP_EDOM;
    for (int k = 0; k < n; ++k)
        weights_out[k] = exp(-t * mu[k]);
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/*
 * cheap_weights_heat_propagator_2d — 2D heat propagator weights.
 *
 *   weights_out[j*ny + k] = exp(-t · (lx[j] + ly[k]))
 *
 * Builds 2D Laplacian in-place (pass 1), then applies exp(-t··) (pass 2).
 * DC: exp(0) = 1.0. Row-major: weights_out[j*ny + k].
 *
 * Usage:
 *   double w[NX*NY];
 *   cheap_weights_heat_propagator_2d(NX, NY, 0.05, w);
 */
static inline int cheap_weights_heat_propagator_2d(int nx, int ny,
                                                    double t,
                                                    double* restrict weights_out)
{
    if (nx < 2 || ny < 2 || t <= 0.0 || !weights_out) return CHEAP_EINVAL;
    for (int j = 0; j < nx; ++j) {
        double sx = sin(M_PI * (double)j / (2.0 * (double)nx));
        double lx = 4.0 * sx * sx;
        for (int k = 0; k < ny; ++k) {
            double sy = sin(M_PI * (double)k / (2.0 * (double)ny));
            weights_out[j * ny + k] = lx + 4.0 * sy * sy;
        }
    }
    const int N = nx * ny;
    for (int i = 0; i < N; ++i)
        weights_out[i] = exp(-t * weights_out[i]);
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, N);
    return CHEAP_OK;
}

/*
 * cheap_weights_heat_propagator_3d — 3D heat propagator weights.
 *
 *   weights_out[(j*ny+k)*nz+l] = exp(-t · (lx[j] + ly[k] + lz[l]))
 *
 * DC: 1.0. Row-major: (j*ny+k)*nz+l.
 */
static inline int cheap_weights_heat_propagator_3d(int nx, int ny, int nz,
                                                    double t,
                                                    double* restrict weights_out)
{
    if (nx < 2 || ny < 2 || nz < 2 || t <= 0.0 || !weights_out) return CHEAP_EINVAL;
    for (int j = 0; j < nx; ++j) {
        double sx = sin(M_PI * (double)j / (2.0 * (double)nx));
        double lx = 4.0 * sx * sx;
        for (int k = 0; k < ny; ++k) {
            double sy = sin(M_PI * (double)k / (2.0 * (double)ny));
            double ly = 4.0 * sy * sy;
            for (int l = 0; l < nz; ++l) {
                double sz = sin(M_PI * (double)l / (2.0 * (double)nz));
                weights_out[(j * ny + k) * nz + l] = lx + ly + 4.0 * sz * sz;
            }
        }
    }
    const int N = nx * ny * nz;
    for (int i = 0; i < N; ++i)
        weights_out[i] = exp(-t * weights_out[i]);
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, N);
    return CHEAP_OK;
}

/*
 * cheap_weights_biharmonic_ev — biharmonic inverse weights.
 *
 *   weights_out[k] = 1 / (μ[k]² + ε),   k = 0..n-1
 *
 * Inverts the biharmonic operator Δ² in the spectral domain. Appears in
 * thin-plate spline interpolation, Euler–Bernoulli beam equations, and 2D
 * Stokes flow. ε > 0 regularizes the DC bin: w[0] = 1/ε — bounded and
 * finite. Zero the DC after the call if mean-free output is required.
 *
 * Works with both Flandrin and Laplacian bases. Vectorized (AVX2/NEON).
 *
 * Usage:
 *   double mu[N], w[N];
 *   cheap_weights_laplacian_ev(N, mu);
 *   cheap_weights_biharmonic_ev(N, mu, 1e-4, w);
 *   cheap_apply(&ctx, rhs, w, solution);
 */
static inline int cheap_weights_biharmonic_ev(int n,
                                               const double* restrict mu,
                                               double eps,
                                               double* restrict weights_out)
{
    if (n < 2 || !mu || eps <= 0.0 || !weights_out) return CHEAP_EINVAL;
    for (int i = 0; i < n; ++i)
        if (!isfinite(mu[i])) return CHEAP_EDOM;
    int k = 0;
#if defined(CHEAP_SIMD_AVX2)
    {
        __m256d vone = _mm256_set1_pd(1.0);
        __m256d veps = _mm256_set1_pd(eps);
        for (; k + 4 <= n; k += 4) {
            __m256d vm  = _mm256_loadu_pd(mu + k);
            __m256d vm2 = _mm256_mul_pd(vm, vm);
            __m256d den = _mm256_add_pd(vm2, veps);
            _mm256_storeu_pd(weights_out + k, _mm256_div_pd(vone, den));
        }
    }
#elif defined(CHEAP_SIMD_NEON)
    {
        float64x2_t vone = vdupq_n_f64(1.0);
        float64x2_t veps = vdupq_n_f64(eps);
        for (; k + 2 <= n; k += 2) {
            float64x2_t vm  = vld1q_f64(mu + k);
            float64x2_t vm2 = vmulq_f64(vm, vm);
            float64x2_t den = vaddq_f64(vm2, veps);
            vst1q_f64(weights_out + k, vdivq_f64(vone, den));
        }
    }
#endif
    for (; k < n; ++k) {
        double m2 = mu[k] * mu[k];
        weights_out[k] = 1.0 / (m2 + eps);
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/*
 * cheap_weights_biharmonic_2d — 2D biharmonic inverse weights.
 *
 *   weights_out[j*ny + k] = 1 / ((lx[j]+ly[k])² + ε)
 *
 * Builds 2D Laplacian in-place (pass 1), then applies biharmonic formula
 * with SIMD (pass 2). DC: 1/ε. Row-major: weights_out[j*ny + k].
 *
 * Usage:
 *   double w[NX*NY];
 *   cheap_weights_biharmonic_2d(NX, NY, 1e-4, w);
 */
static inline int cheap_weights_biharmonic_2d(int nx, int ny,
                                               double eps,
                                               double* restrict weights_out)
{
    if (nx < 2 || ny < 2 || eps <= 0.0 || !weights_out) return CHEAP_EINVAL;
    for (int j = 0; j < nx; ++j) {
        double sx = sin(M_PI * (double)j / (2.0 * (double)nx));
        double lx = 4.0 * sx * sx;
        for (int k = 0; k < ny; ++k) {
            double sy = sin(M_PI * (double)k / (2.0 * (double)ny));
            weights_out[j * ny + k] = lx + 4.0 * sy * sy;
        }
    }
    const int N = nx * ny;
    int i = 0;
#if defined(CHEAP_SIMD_AVX2)
    {
        __m256d vone = _mm256_set1_pd(1.0);
        __m256d veps = _mm256_set1_pd(eps);
        for (; i + 4 <= N; i += 4) {
            __m256d vm  = _mm256_loadu_pd(weights_out + i);
            __m256d vm2 = _mm256_mul_pd(vm, vm);
            __m256d den = _mm256_add_pd(vm2, veps);
            _mm256_storeu_pd(weights_out + i, _mm256_div_pd(vone, den));
        }
    }
#elif defined(CHEAP_SIMD_NEON)
    {
        float64x2_t vone = vdupq_n_f64(1.0);
        float64x2_t veps = vdupq_n_f64(eps);
        for (; i + 2 <= N; i += 2) {
            float64x2_t vm  = vld1q_f64(weights_out + i);
            float64x2_t vm2 = vmulq_f64(vm, vm);
            float64x2_t den = vaddq_f64(vm2, veps);
            vst1q_f64(weights_out + i, vdivq_f64(vone, den));
        }
    }
#endif
    for (; i < N; ++i) {
        double m2 = weights_out[i] * weights_out[i];
        weights_out[i] = 1.0 / (m2 + eps);
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, N);
    return CHEAP_OK;
}

/*
 * cheap_weights_biharmonic_3d — 3D biharmonic inverse weights.
 *
 *   weights_out[(j*ny+k)*nz+l] = 1 / ((lx[j]+ly[k]+lz[l])² + ε)
 *
 * DC: 1/ε. Row-major: (j*ny+k)*nz+l.
 */
static inline int cheap_weights_biharmonic_3d(int nx, int ny, int nz,
                                               double eps,
                                               double* restrict weights_out)
{
    if (nx < 2 || ny < 2 || nz < 2 || eps <= 0.0 || !weights_out) return CHEAP_EINVAL;
    for (int j = 0; j < nx; ++j) {
        double sx = sin(M_PI * (double)j / (2.0 * (double)nx));
        double lx = 4.0 * sx * sx;
        for (int k = 0; k < ny; ++k) {
            double sy = sin(M_PI * (double)k / (2.0 * (double)ny));
            double ly = 4.0 * sy * sy;
            for (int l = 0; l < nz; ++l) {
                double sz = sin(M_PI * (double)l / (2.0 * (double)nz));
                weights_out[(j * ny + k) * nz + l] = lx + ly + 4.0 * sz * sz;
            }
        }
    }
    const int N = nx * ny * nz;
    int i = 0;
#if defined(CHEAP_SIMD_AVX2)
    {
        __m256d vone = _mm256_set1_pd(1.0);
        __m256d veps = _mm256_set1_pd(eps);
        for (; i + 4 <= N; i += 4) {
            __m256d vm  = _mm256_loadu_pd(weights_out + i);
            __m256d vm2 = _mm256_mul_pd(vm, vm);
            __m256d den = _mm256_add_pd(vm2, veps);
            _mm256_storeu_pd(weights_out + i, _mm256_div_pd(vone, den));
        }
    }
#elif defined(CHEAP_SIMD_NEON)
    {
        float64x2_t vone = vdupq_n_f64(1.0);
        float64x2_t veps = vdupq_n_f64(eps);
        for (; i + 2 <= N; i += 2) {
            float64x2_t vm  = vld1q_f64(weights_out + i);
            float64x2_t vm2 = vmulq_f64(vm, vm);
            float64x2_t den = vaddq_f64(vm2, veps);
            vst1q_f64(weights_out + i, vdivq_f64(vone, den));
        }
    }
#endif
    for (; i < N; ++i) {
        double m2 = weights_out[i] * weights_out[i];
        weights_out[i] = 1.0 / (m2 + eps);
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, N);
    return CHEAP_OK;
}

/*
 * cheap_weights_poisson_ev — Poisson inverse-Laplacian weights.
 *
 *   weights_out[0] = 0.0                        (DC: null space of Laplacian)
 *   weights_out[k] = 1 / (μ[k] + ε),   k = 1..n-1
 *
 * Inverts -Δ on the mean-free subspace. The DC projection (w[0]=0) enforces
 * the compatible solvability condition: the right-hand side must have zero
 * mean for -Δu = f to have a unique solution up to a constant.
 *
 * ε >= 0. With ε = 0 the caller must guarantee μ[k] > 0 for k > 0 (true
 * for Laplacian eigenvalues on any grid). ε > 0 adds Tikhonov-style DC
 * damping to the non-DC modes.
 *
 * Contrast with cheap_weights_specnorm_ev: that weights by 1/√(μ+ε)
 * (whitening), while this weights by 1/(μ+ε) (exact spectral inverse).
 *
 * Works with both Flandrin and Laplacian bases. Vectorized (AVX2/NEON).
 *
 * Usage:
 *   double mu[N], w[N], u[N];
 *   cheap_weights_laplacian_ev(N, mu);
 *   cheap_weights_poisson_ev(N, mu, 1e-8, w);
 *   cheap_apply(&ctx, rhs, w, u);  // solves -Δu ≈ rhs
 */
static inline int cheap_weights_poisson_ev(int n,
                                            const double* restrict mu,
                                            double eps,
                                            double* restrict weights_out)
{
    if (n < 2 || !mu || eps < 0.0 || !weights_out) return CHEAP_EINVAL;
    for (int i = 0; i < n; ++i)
        if (!isfinite(mu[i])) return CHEAP_EDOM;
    weights_out[0] = 0.0;
    int k = 1;
#if defined(CHEAP_SIMD_AVX2)
    {
        __m256d vone = _mm256_set1_pd(1.0);
        __m256d veps = _mm256_set1_pd(eps);
        for (; k + 4 <= n; k += 4) {
            __m256d vm  = _mm256_loadu_pd(mu + k);
            __m256d den = _mm256_add_pd(vm, veps);
            _mm256_storeu_pd(weights_out + k, _mm256_div_pd(vone, den));
        }
    }
#elif defined(CHEAP_SIMD_NEON)
    {
        float64x2_t vone = vdupq_n_f64(1.0);
        float64x2_t veps = vdupq_n_f64(eps);
        for (; k + 2 <= n; k += 2) {
            float64x2_t vm  = vld1q_f64(mu + k);
            float64x2_t den = vaddq_f64(vm, veps);
            vst1q_f64(weights_out + k, vdivq_f64(vone, den));
        }
    }
#endif
    for (; k < n; ++k)
        weights_out[k] = 1.0 / (mu[k] + eps);
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/*
 * cheap_weights_poisson_2d — 2D Poisson inverse-Laplacian weights.
 *
 *   weights_out[0]      = 0.0   (DC projected out)
 *   weights_out[j*ny+k] = 1 / (lx[j] + ly[k] + ε),   otherwise
 *
 * Builds 2D Laplacian in-place (pass 1), then applies Poisson formula with
 * SIMD (pass 2, i=1..N-1), then enforces weights_out[0]=0.0.
 * Row-major: weights_out[j*ny + k].
 *
 * Usage:
 *   double w[NX*NY];
 *   cheap_weights_poisson_2d(NX, NY, 1e-8, w);
 */
static inline int cheap_weights_poisson_2d(int nx, int ny,
                                            double eps,
                                            double* restrict weights_out)
{
    if (nx < 2 || ny < 2 || eps < 0.0 || !weights_out) return CHEAP_EINVAL;
    for (int j = 0; j < nx; ++j) {
        double sx = sin(M_PI * (double)j / (2.0 * (double)nx));
        double lx = 4.0 * sx * sx;
        for (int k = 0; k < ny; ++k) {
            double sy = sin(M_PI * (double)k / (2.0 * (double)ny));
            weights_out[j * ny + k] = lx + 4.0 * sy * sy;
        }
    }
    const int N = nx * ny;
    int i = 1;
#if defined(CHEAP_SIMD_AVX2)
    {
        __m256d vone = _mm256_set1_pd(1.0);
        __m256d veps = _mm256_set1_pd(eps);
        for (; i + 4 <= N; i += 4) {
            __m256d vm  = _mm256_loadu_pd(weights_out + i);
            __m256d den = _mm256_add_pd(vm, veps);
            _mm256_storeu_pd(weights_out + i, _mm256_div_pd(vone, den));
        }
    }
#elif defined(CHEAP_SIMD_NEON)
    {
        float64x2_t vone = vdupq_n_f64(1.0);
        float64x2_t veps = vdupq_n_f64(eps);
        for (; i + 2 <= N; i += 2) {
            float64x2_t vm  = vld1q_f64(weights_out + i);
            float64x2_t den = vaddq_f64(vm, veps);
            vst1q_f64(weights_out + i, vdivq_f64(vone, den));
        }
    }
#endif
    for (; i < N; ++i)
        weights_out[i] = 1.0 / (weights_out[i] + eps);
    weights_out[0] = 0.0;
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, N);
    return CHEAP_OK;
}

/*
 * cheap_weights_poisson_3d — 3D Poisson inverse-Laplacian weights.
 *
 *   weights_out[0]            = 0.0   (DC projected out)
 *   weights_out[(j*ny+k)*nz+l] = 1 / (lx[j]+ly[k]+lz[l]+ε),   otherwise
 *
 * Row-major: (j*ny+k)*nz+l.
 */
static inline int cheap_weights_poisson_3d(int nx, int ny, int nz,
                                            double eps,
                                            double* restrict weights_out)
{
    if (nx < 2 || ny < 2 || nz < 2 || eps < 0.0 || !weights_out) return CHEAP_EINVAL;
    for (int j = 0; j < nx; ++j) {
        double sx = sin(M_PI * (double)j / (2.0 * (double)nx));
        double lx = 4.0 * sx * sx;
        for (int k = 0; k < ny; ++k) {
            double sy = sin(M_PI * (double)k / (2.0 * (double)ny));
            double ly = 4.0 * sy * sy;
            for (int l = 0; l < nz; ++l) {
                double sz = sin(M_PI * (double)l / (2.0 * (double)nz));
                weights_out[(j * ny + k) * nz + l] = lx + ly + 4.0 * sz * sz;
            }
        }
    }
    const int N = nx * ny * nz;
    int i = 1;
#if defined(CHEAP_SIMD_AVX2)
    {
        __m256d vone = _mm256_set1_pd(1.0);
        __m256d veps = _mm256_set1_pd(eps);
        for (; i + 4 <= N; i += 4) {
            __m256d vm  = _mm256_loadu_pd(weights_out + i);
            __m256d den = _mm256_add_pd(vm, veps);
            _mm256_storeu_pd(weights_out + i, _mm256_div_pd(vone, den));
        }
    }
#elif defined(CHEAP_SIMD_NEON)
    {
        float64x2_t vone = vdupq_n_f64(1.0);
        float64x2_t veps = vdupq_n_f64(eps);
        for (; i + 2 <= N; i += 2) {
            float64x2_t vm  = vld1q_f64(weights_out + i);
            float64x2_t den = vaddq_f64(vm, veps);
            vst1q_f64(weights_out + i, vdivq_f64(vone, den));
        }
    }
#endif
    for (; i < N; ++i)
        weights_out[i] = 1.0 / (weights_out[i] + eps);
    weights_out[0] = 0.0;
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, N);
    return CHEAP_OK;
}

/*
 * cheap_weights_higher_order_tikhonov_deconv_ev — higher-order Tikhonov
 * deconvolution weights.
 *
 *   weights_out[k] = ψ[k] / (ψ[k]² + α·μ[k]^p + ε)
 *
 * where ψ = psf_eigenvalues (DCT-II spectrum of PSF first column) and
 * μ = lap_eigenvalues (Laplacian, or 1D Laplacian computed internally if
 * lap_eigenvalues == NULL). α >= 0 is the regularization strength; p > 0
 * is the penalty order (p=1: gradient penalty, p=2: biharmonic penalty).
 * ε >= 0 prevents division by zero when both ψ[k] and μ[k] are near zero.
 *
 * Comparison to Wiener deconvolution (cheap_weights_wiener_deconv_ev):
 * that formula uses a flat noise floor η in the denominator; this formula
 * uses α·μ^p which penalizes rough solutions more than smooth ones.
 *
 * DC (μ[0]=0, pow(0,p)=0 by C99): w[0] = ψ[0]/(ψ[0]²+ε). The fmax
 * floor CHEAP_EPS_DIV guards against ψ[0]=0 with ε=0.
 *
 * Works with both Flandrin and Laplacian bases for lap_eigenvalues.
 * Scalar-only (pow in hot path).
 *
 * Usage:
 *   double psf_eig[N], w[N];
 *   cheap_toeplitz_eigenvalues(&ctx, psf_col, psf_eig);
 *   cheap_weights_higher_order_tikhonov_deconv_ev(
 *       N, psf_eig, NULL, 0.01, 2.0, 1e-8, w);  // biharmonic penalty
 *   cheap_apply(&ctx, blurred, w, restored);
 */
static inline int cheap_weights_higher_order_tikhonov_deconv_ev(
    int n,
    const double* restrict psf_eigenvalues,
    const double* restrict lap_eigenvalues,
    double alpha, double p, double eps,
    double* restrict weights_out)
{
    if (n < 2 || !psf_eigenvalues || alpha < 0.0 || p <= 0.0
        || eps < 0.0 || !weights_out)
        return CHEAP_EINVAL;
    for (int i = 0; i < n; ++i)
        if (!isfinite(psf_eigenvalues[i])) return CHEAP_EDOM;
    if (lap_eigenvalues)
        for (int i = 0; i < n; ++i)
            if (!isfinite(lap_eigenvalues[i])) return CHEAP_EDOM;

    double* lap_local = NULL;
    const double* lap = lap_eigenvalues;
    if (!lap) {
        lap_local = (double*)malloc((size_t)n * sizeof(double));
        if (!lap_local) return CHEAP_ENOMEM;
        cheap__build_laplacian_flat(n, lap_local);
        lap = lap_local;
    }
    for (int k = 0; k < n; ++k) {
        double psi = psf_eigenvalues[k];
        double lv  = lap[k];
        double pen = (lv > CHEAP_EPS_LOG) ? alpha * pow(lv, p) : 0.0;
        double den = psi * psi + pen + eps;
        if (den < CHEAP_EPS_DIV) den = CHEAP_EPS_DIV;
        weights_out[k] = psi / den;
    }
    free(lap_local);
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/* ============================================================
 * Random Fourier Features (RFF)
 *
 * Approximates a Gaussian kernel k(x,y) = exp(-||x-y||^2/(2*sigma^2))
 * via random Fourier features: k(x,y) ≈ z(x)^T z(y).
 * ============================================================ */

typedef struct {
    int D;               /* feature dimension (must be even, >= 2) */
    int d_in;            /* input dimension */
    double sigma;        /* kernel bandwidth */
    double* omega;       /* (D/2) * d_in frequencies, row-major */
    double* bias;        /* D/2 biases in [0, 2*pi) */
    double scale;        /* sqrt(2.0 / D) */
    int is_initialized;
} cheap_rff_ctx;

static inline int cheap_rff_init(cheap_rff_ctx* rctx,
                                    int D, int d_in, double sigma,
                                    uint64_t seed)
{
    if (!rctx || D < 2 || (D % 2) != 0 || d_in < 1 || sigma <= 0.0)
        return CHEAP_EINVAL;
    memset(rctx, 0, sizeof(*rctx));
    rctx->D = D;
    rctx->d_in = d_in;
    rctx->sigma = sigma;
    rctx->scale = sqrt(2.0 / (double)D);

    const int M = D / 2;
    rctx->omega = (double*)malloc((size_t)(M * d_in) * sizeof(double));
    rctx->bias  = (double*)malloc((size_t)M * sizeof(double));
    if (!rctx->omega || !rctx->bias) {
        free(rctx->omega); free(rctx->bias);
        rctx->omega = NULL; rctx->bias = NULL;
        return CHEAP_ENOMEM;
    }

    cheap__rng rng;
    cheap__rng_init(&rng, seed);
    const double inv_sigma = 1.0 / sigma;
    for (int i = 0; i < M * d_in; ++i)
        rctx->omega[i] = cheap__lcg_normal(&rng) * inv_sigma;
    for (int i = 0; i < M; ++i)
        rctx->bias[i] = cheap__lcg_uniform(&rng) * 2.0 * M_PI;

    rctx->is_initialized = 1;
    return CHEAP_OK;
}

static inline void cheap_rff_destroy(cheap_rff_ctx* rctx)
{
    if (!rctx) return;
    free(rctx->omega);
    free(rctx->bias);
    rctx->omega = NULL;
    rctx->bias = NULL;
    rctx->is_initialized = 0;
}

static inline int cheap_rff_map(const cheap_rff_ctx* restrict rctx,
                                   const double* restrict x_in,
                                   double* restrict z_out)
{
    if (!rctx || !rctx->is_initialized || !x_in || !z_out) return CHEAP_EINVAL;
    const int M = rctx->D / 2;
    const int d = rctx->d_in;
    const double s = rctx->scale;

    for (int i = 0; i < M; ++i) {
        double arg = rctx->bias[i];
        const double* wi = rctx->omega + i * d;
        for (int j = 0; j < d; ++j)
            arg += wi[j] * x_in[j];
        z_out[2 * i]     = s * cos(arg);
        z_out[2 * i + 1] = s * sin(arg);
    }
    return CHEAP_OK;
}

static inline int cheap_rff_map_batch(const cheap_rff_ctx* restrict rctx,
                                         const double* restrict X_in,
                                         int N,
                                         double* restrict Z_out)
{
    if (!rctx || !X_in || !Z_out || N < 1) return CHEAP_EINVAL;
    const int d = rctx->d_in;
    const int D = rctx->D;
    for (int n = 0; n < N; ++n) {
        int ret = cheap_rff_map(rctx, X_in + n * d, Z_out + n * D);
        if (ret != CHEAP_OK) return ret;
    }
    return CHEAP_OK;
}

#endif /* CHEAP_H */
