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
    ctx->is_initialized = 0;
}

static inline int cheap_init(cheap_ctx* ctx, int n, double H)
{
    if (!ctx || n < 2 || H <= 0.0 || H >= 1.0) return CHEAP_EINVAL;
    memset(ctx, 0, sizeof(*ctx));
    ctx->n = n;
    ctx->current_eps = -1.0;
    ctx->current_H = H;
    ctx->lambda     = (double*)fftw_malloc((size_t)n * sizeof(double));
    ctx->gibbs      = (double*)fftw_malloc((size_t)n * sizeof(double));
    ctx->sqrt_lambda = (double*)fftw_malloc((size_t)n * sizeof(double));
    ctx->workspace  = (double*)fftw_malloc((size_t)n * sizeof(double));
    if (!ctx->lambda || !ctx->gibbs || !ctx->sqrt_lambda || !ctx->workspace) {
        cheap_destroy(ctx);
        return CHEAP_ENOMEM;
    }
    ctx->plan_fwd = fftw_plan_r2r_1d(n, ctx->workspace, ctx->workspace,
                                      FFTW_REDFT10, FFTW_PATIENT);
    ctx->plan_inv = fftw_plan_r2r_1d(n, ctx->workspace, ctx->workspace,
                                      FFTW_REDFT01, FFTW_PATIENT);
    if (!ctx->plan_fwd || !ctx->plan_inv) {
        cheap_destroy(ctx);
        return CHEAP_ENOMEM;
    }
    const double pow_n_2H = pow((double)n, 2.0 * H);
    const double twoH_plus_1 = 2.0 * H + 1.0;
    ctx->lambda[0] = CHEAP_EPS_LAMBDA * pow_n_2H;
    for (int k = 1; k < n; ++k) {
        double s = sin(M_PI * (double)k / (2.0 * (double)n));
        if (s < CHEAP_EPS_LOG) s = CHEAP_EPS_LOG;
        ctx->lambda[k] = pow_n_2H * pow(s, -twoH_plus_1);
    }
    for (int k = 0; k < n; ++k)
        ctx->sqrt_lambda[k] = sqrt(fmax(ctx->lambda[k], CHEAP_EPS_LAMBDA));
    ctx->is_initialized = 1;
    return CHEAP_OK;
}

/* ============================================================
 * Core spectral primitives: forward, inverse, apply
 * ============================================================ */

/*
 * cheap_forward — DCT-II of input into ctx->workspace.
 * After this call, ctx->workspace contains the spectral coefficients.
 */
static inline int cheap_forward(cheap_ctx* ctx, const double* input)
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
static inline int cheap_inverse(cheap_ctx* ctx, double* output)
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
static inline int cheap_apply(cheap_ctx* ctx, const double* input,
                                const double* weights, double* output)
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

static inline int cheap_recompute_gibbs(cheap_ctx* ctx, double eps)
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

static inline int cheap_sinkhorn(cheap_ctx* ctx,
                                   const double* a, const double* b,
                                   double eps, int max_iter, double tol,
                                   double* f, double* g)
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
    for (int i = 0; i < ctx->n; ++i) f[i] = g[i] = 0.0;
    double* temp = (double*)malloc((size_t)ctx->n * sizeof(double));
    double* prev_g = (double*)malloc((size_t)ctx->n * sizeof(double));
    if (!temp || !prev_g) {
        free(temp); free(prev_g);
        return CHEAP_ENOMEM;
    }
    int iter;
    for (iter = 0; iter < max_iter; ++iter) {
        memcpy(prev_g, g, (size_t)ctx->n * sizeof(double));
        cheap_apply_hybrid_log(ctx, g, temp);
        for (int i = 0; i < ctx->n; ++i) f[i] = log(a[i] + CHEAP_EPS_DIV) - temp[i];
        cheap_apply_hybrid_log(ctx, f, temp);
        for (int i = 0; i < ctx->n; ++i) g[i] = log(b[i] + CHEAP_EPS_DIV) - temp[i];
        double max_diff = 0.0;
        for (int i = 0; i < ctx->n; ++i) {
            double diff = fabs(g[i] - prev_g[i]);
            if (diff > max_diff) max_diff = diff;
        }
        if (max_diff < tol) break;
    }
    free(temp);
    free(prev_g);
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

static inline int cheap_toeplitz_eigenvalues(cheap_ctx* ctx,
                                                const double* t,
                                                double* lambda_out)
{
    if (!ctx || !ctx->is_initialized || !t || !lambda_out) return CHEAP_EINVAL;
    const int n = ctx->n;
    for (int i = 0; i < n; ++i) if (!isfinite(t[i])) return CHEAP_EDOM;
    memcpy(ctx->workspace, t, (size_t)n * sizeof(double));
    fftw_execute(ctx->plan_fwd);
    memcpy(lambda_out, ctx->workspace, (size_t)n * sizeof(double));
    return CHEAP_OK;
}

static inline int cheap_toeplitz_solve_precomp(cheap_ctx* ctx,
                                                  const double* lambda_t,
                                                  const double* y,
                                                  double lambda_reg,
                                                  double* x)
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

static inline int cheap_rff_map(const cheap_rff_ctx* rctx,
                                   const double* x_in,
                                   double* z_out)
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

static inline int cheap_rff_map_batch(const cheap_rff_ctx* rctx,
                                         const double* X_in,
                                         int N,
                                         double* Z_out)
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
