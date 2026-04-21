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
 */

#define CHEAP_VERSION_MAJOR 0
#define CHEAP_VERSION_MINOR 1
#define CHEAP_VERSION_PATCH 0
#define CHEAP_VERSION "0.1.0"

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
 * Core spectral primitives: forward, inverse, apply
 * ============================================================ */

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
 * cheap_inverse — iDCT-III of ctx->workspace into output, with 1/(2N) normalization.
 * Assumes ctx->workspace already contains spectral data (e.g. from cheap_forward
 * followed by pointwise manipulation).
 */
static inline int cheap_inverse(cheap_ctx* restrict ctx,
                                  double* restrict output)
{
    if (!ctx || !ctx->is_initialized) return CHEAP_EUNINIT;
    if (!output) return CHEAP_EINVAL;
    const int n = ctx->n;
    const double norm = 1.0 / (2.0 * (double)n);
    fftw_execute(ctx->plan_inv);
    for (int i = 0; i < n; ++i) output[i] = ctx->workspace[i] * norm;
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
    for (int k = 0; k < n; ++k) ctx->workspace[k] *= weights[k];
    fftw_execute(ctx->plan_inv);
    for (int i = 0; i < n; ++i) output[i] = ctx->workspace[i] * norm;
    return CHEAP_OK;
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
    for (int k = 0; k < n; ++k) {
        if (!isfinite(lambda[k])) return CHEAP_EDOM;
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
    for (int k = 0; k < n; ++k) {
        if (!isfinite(lambda[k])) return CHEAP_EDOM;
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
 * cheap_weights_mandelbrot — Mandelbrot multifractal spectral weights.
 *
 *   weights_out[k] = |Gamma(H + i*tau_k) / Gamma(1-H + i*tau_k)|
 *
 * where tau_k = pi*k/n. H must be in (0, 1).
 * At H = 0.5, all weights equal 1.0 (symmetry).
 * Computed in log-space via Lanczos complex log-Gamma.
 */
static inline int cheap_weights_mandelbrot(int n, double H,
                                             double* restrict weights_out)
{
    if (n < 2 || H <= 0.0 || H >= 1.0 || !weights_out) return CHEAP_EINVAL;

    /* k = 0: real Gamma ratio */
    weights_out[0] = exp(lgamma(H) - lgamma(1.0 - H));

    for (int k = 1; k < n; ++k) {
        double tau = M_PI * (double)k / (double)n;
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
    double sc = sqrt(c);
    double lambda_plus = sigma_sq * (1.0 + sc) * (1.0 + sc);
    for (int k = 0; k < n; ++k) {
        if (!isfinite(lambda[k])) return CHEAP_EDOM;
        weights_out[k] = (lambda[k] > lambda_plus) ? lambda[k] : 0.0;
    }
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
    double sc = sqrt(c);
    double lp = (1.0 + sc) * (1.0 + sc);
    double lm = (1.0 - sc) * (1.0 - sc);
    double lambda_plus = sigma_sq * lp;
    for (int k = 0; k < n; ++k) {
        if (!isfinite(lambda[k])) return CHEAP_EDOM;
        if (lambda[k] <= lambda_plus) {
            weights_out[k] = 0.0;
        } else {
            double l = lambda[k] / sigma_sq;
            double factor = sqrt(fmax(0.0, (l - lp) * (l - lm)));
            weights_out[k] = lambda[k] * factor / l;
        }
    }
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
