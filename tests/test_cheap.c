/*
 * test_cheap.c — correctness tests for the CHEAP header-only library.
 * Core tests: context, apply/forward/inverse, Sinkhorn, Toeplitz, RFF
 *
 * Build (standalone):
 *   gcc -std=c99 -pedantic -Wall -Wextra -Werror -march=native -O1 \
 *       -fsanitize=address,undefined -D_POSIX_C_SOURCE=199309L \
 *       test_cheap.c -o test_cheap -lfftw3 -lm
 */

#include "cheap.h"

#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Minimal test framework
 * ========================================================================= */
static int g_tests_run    = 0;
static int g_tests_failed = 0;

#define ASSERT_TRUE(cond) do { \
    ++g_tests_run; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL  %s:%d  ASSERT_TRUE(%s)\n", \
                __FILE__, __LINE__, #cond); \
        ++g_tests_failed; \
    } \
} while (0)

#define ASSERT_EQ(a, b) do { \
    ++g_tests_run; \
    if ((a) != (b)) { \
        fprintf(stderr, "FAIL  %s:%d  ASSERT_EQ(%s, %s)  [%d != %d]\n", \
                __FILE__, __LINE__, #a, #b, (int)(a), (int)(b)); \
        ++g_tests_failed; \
    } \
} while (0)

#define ASSERT_NEAR(a, b, tol) do { \
    ++g_tests_run; \
    double _a = (double)(a); \
    double _b = (double)(b); \
    double _t = (double)(tol); \
    if (!isfinite(_a) || !isfinite(_b) || fabs(_a - _b) > _t) { \
        fprintf(stderr, \
                "FAIL  %s:%d  ASSERT_NEAR(%s, %s, %g)  [%.6e vs %.6e, diff=%.3e]\n", \
                __FILE__, __LINE__, #a, #b, _t, _a, _b, fabs(_a - _b)); \
        ++g_tests_failed; \
    } \
} while (0)

/* =========================================================================
 * Helper: apply K without touching ctx->workspace
 * ========================================================================= */
static void apply_K_test(cheap_ctx *ctx, const double *v, double *Kv)
{
    const int n = ctx->n;
    const double norm = 1.0 / (2.0 * (double)n);

    double *buf = (double *)fftw_malloc((size_t)n * sizeof(double));
    if (!buf) { fprintf(stderr, "apply_K_test: alloc failed\n"); return; }

    fftw_plan pfwd = fftw_plan_r2r_1d(n, buf, buf, FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_plan pinv = fftw_plan_r2r_1d(n, buf, buf, FFTW_REDFT01, FFTW_ESTIMATE);

    memcpy(buf, v, (size_t)n * sizeof(double));
    fftw_execute(pfwd);
    for (int k = 0; k < n; ++k) buf[k] *= ctx->lambda[k];
    fftw_execute(pinv);
    for (int i = 0; i < n; ++i) Kv[i] = buf[i] * norm;

    fftw_destroy_plan(pfwd);
    fftw_destroy_plan(pinv);
    fftw_free(buf);
}

/* =========================================================================
 * Section 1: Core lifecycle tests
 * ========================================================================= */

static void test_init_destroy(void)
{
    printf("  test_init_destroy\n");
    cheap_ctx ctx;
    int ret;

    ret = cheap_init(&ctx, 64, 0.7);
    ASSERT_EQ(ret, CHEAP_OK);
    ASSERT_TRUE(ctx.is_initialized == 1);
    ASSERT_TRUE(ctx.n == 64);

    for (int k = 0; k < ctx.n; ++k)
        ASSERT_TRUE(ctx.lambda[k] > 0.0);

    /* sqrt_lambda should be precomputed */
    for (int k = 0; k < ctx.n; ++k)
        ASSERT_NEAR(ctx.sqrt_lambda[k], sqrt(fmax(ctx.lambda[k], CHEAP_EPS_LAMBDA)), 1e-14);

    cheap_destroy(&ctx);
    ASSERT_TRUE(ctx.is_initialized == 0);

    /* Pointers should be nulled after destroy */
    ASSERT_TRUE(ctx.lambda == NULL);
    ASSERT_TRUE(ctx.gibbs == NULL);
    ASSERT_TRUE(ctx.sqrt_lambda == NULL);
    ASSERT_TRUE(ctx.workspace == NULL);
    ASSERT_TRUE(ctx.plan_fwd == NULL);
    ASSERT_TRUE(ctx.plan_inv == NULL);

    /* Double destroy should be safe */
    cheap_destroy(&ctx);
    cheap_destroy(NULL);
}

static void test_error_codes(void)
{
    printf("  test_error_codes\n");
    cheap_ctx ctx;
    double y[8] = {1,2,3,4,5,6,7,8};
    double w[8] = {1,1,1,1,1,1,1,1};
    double out[8];

    ASSERT_EQ(cheap_init(NULL, 8, 0.5), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init(&ctx, 1, 0.5), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init(&ctx, 8, 0.0), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init(&ctx, 8, 1.0), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init(&ctx, 8, -0.1), CHEAP_EINVAL);

    /* Uninitialized ctx */
    memset(&ctx, 0, sizeof(ctx));
    ASSERT_EQ(cheap_apply(&ctx, y, w, out), CHEAP_EUNINIT);
    ASSERT_EQ(cheap_forward(&ctx, y), CHEAP_EUNINIT);
    ASSERT_EQ(cheap_inverse(&ctx, out), CHEAP_EUNINIT);

    ASSERT_EQ(cheap_init(&ctx, 8, 0.5), CHEAP_OK);

    /* NaN input */
    double y_nan[8] = {1,2,3,4,5,6,7, (double)NAN};
    ASSERT_EQ(cheap_apply(&ctx, y_nan, w, out), CHEAP_EDOM);
    ASSERT_EQ(cheap_forward(&ctx, y_nan), CHEAP_EDOM);

    /* NULL ptrs */
    ASSERT_EQ(cheap_apply(&ctx, NULL, w, out), CHEAP_EINVAL);
    ASSERT_EQ(cheap_apply(&ctx, y, NULL, out), CHEAP_EINVAL);
    ASSERT_EQ(cheap_apply(&ctx, y, w, NULL), CHEAP_EINVAL);
    ASSERT_EQ(cheap_forward(&ctx, NULL), CHEAP_EINVAL);
    ASSERT_EQ(cheap_inverse(&ctx, NULL), CHEAP_EINVAL);

    /* Sinkhorn errors */
    double a8[8] = {0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125};
    double b8_bad[8] = {0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1};
    double f8[8], g8[8];
    ASSERT_EQ(cheap_sinkhorn(&ctx, a8, b8_bad, 0.5, 100, 1e-6, f8, g8),
              CHEAP_EINVAL);

    double b8[8] = {0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125};
    ASSERT_EQ(cheap_sinkhorn(&ctx, a8, b8, 0.5, 0, 1e-15, f8, g8),
              CHEAP_ENOCONV);

    cheap_destroy(&ctx);
}

/* =========================================================================
 * Section 2: Core spectral primitive tests
 * ========================================================================= */

static void test_apply_identity(void)
{
    printf("  test_apply_identity\n");
    const int n = 128;
    cheap_ctx ctx;
    ASSERT_EQ(cheap_init(&ctx, n, 0.5), CHEAP_OK);

    double *input = (double *)malloc((size_t)n * sizeof(double));
    double *output = (double *)malloc((size_t)n * sizeof(double));
    double *ones = (double *)malloc((size_t)n * sizeof(double));

    for (int i = 0; i < n; ++i) {
        input[i] = sin(2.0 * M_PI * (double)i / n);
        ones[i] = 1.0;
    }

    /* Apply with identity weights should return input */
    ASSERT_EQ(cheap_apply(&ctx, input, ones, output), CHEAP_OK);
    for (int i = 0; i < n; ++i)
        ASSERT_NEAR(output[i], input[i], 1e-12);

    free(input); free(output); free(ones);
    cheap_destroy(&ctx);
}

static void test_apply_krr(void)
{
    printf("  test_apply_krr\n");
    const int n = 128;
    cheap_ctx ctx;
    ASSERT_EQ(cheap_init(&ctx, n, 0.7), CHEAP_OK);

    double *y     = (double *)malloc((size_t)n * sizeof(double));
    double *alpha = (double *)malloc((size_t)n * sizeof(double));
    double *Ka    = (double *)malloc((size_t)n * sizeof(double));
    double *w_krr = (double *)malloc((size_t)n * sizeof(double));

    for (int i = 0; i < n; ++i)
        y[i] = sin(2.0 * M_PI * (double)i / (double)n) + 0.5;

    const double lambda_reg = 1e-3;
    for (int k = 0; k < n; ++k) {
        double denom = ctx.lambda[k] + lambda_reg;
        if (denom < CHEAP_EPS_DIV) denom = CHEAP_EPS_DIV;
        w_krr[k] = 1.0 / denom;
    }

    /* KRR solve via cheap_apply */
    ASSERT_EQ(cheap_apply(&ctx, y, w_krr, alpha), CHEAP_OK);

    /* Verify: K*alpha + lambda*alpha ≈ y */
    apply_K_test(&ctx, alpha, Ka);
    for (int i = 0; i < n; ++i) {
        double res = Ka[i] + lambda_reg * alpha[i] - y[i];
        ASSERT_NEAR(res, 0.0, 1e-6);
    }

    /* Large regularization: alpha ≈ y / lambda_reg */
    const double big_reg = 1e6;
    for (int k = 0; k < n; ++k) {
        double denom = ctx.lambda[k] + big_reg;
        w_krr[k] = 1.0 / denom;
    }
    ASSERT_EQ(cheap_apply(&ctx, y, w_krr, alpha), CHEAP_OK);
    for (int i = 0; i < n; ++i)
        ASSERT_NEAR(alpha[i], y[i] / big_reg, 1e-6);

    free(y); free(alpha); free(Ka); free(w_krr);
    cheap_destroy(&ctx);
}

static void test_apply_reparam(void)
{
    printf("  test_apply_reparam\n");
    const int n = 32;
    cheap_ctx ctx;
    ASSERT_EQ(cheap_init(&ctx, n, 0.7), CHEAP_OK);

    double *mu    = (double *)malloc((size_t)n * sizeof(double));
    double *eps   = (double *)malloc((size_t)n * sizeof(double));
    double *z_out = (double *)malloc((size_t)n * sizeof(double));

    for (int i = 0; i < n; ++i) mu[i] = (double)i;

    /* Zero noise: z = mu */
    memset(eps, 0, (size_t)n * sizeof(double));
    ASSERT_EQ(cheap_apply(&ctx, eps, ctx.sqrt_lambda, z_out), CHEAP_OK);
    for (int i = 0; i < n; ++i) z_out[i] += mu[i];
    for (int i = 0; i < n; ++i)
        ASSERT_NEAR(z_out[i], mu[i], 1e-14);

    /* Monte Carlo variance check */
    {
        const int M = 4096;
        uint32_t lcg = 98765u;
        double sum_sq = 0.0;
        memset(mu, 0, (size_t)n * sizeof(double));

        for (int m = 0; m < M; ++m) {
            for (int i = 0; i < n; i += 2) {
                lcg = lcg * 1664525u + 1013904223u;
                double u1 = ((double)(lcg >> 8) + 0.5) / (double)(1 << 24);
                lcg = lcg * 1664525u + 1013904223u;
                double u2 = ((double)(lcg >> 8) + 0.5) / (double)(1 << 24);
                double mag = sqrt(-2.0 * log(u1 + 1e-12));
                eps[i]     = mag * cos(2.0 * M_PI * u2);
                if (i + 1 < n)
                    eps[i + 1] = mag * sin(2.0 * M_PI * u2);
            }
            cheap_apply(&ctx, eps, ctx.sqrt_lambda, z_out);
            for (int i = 0; i < n; ++i) z_out[i] += mu[i];
            for (int i = 0; i < n; ++i)
                sum_sq += z_out[i] * z_out[i];
        }
        double emp_var_per_elem = sum_sq / ((double)M * (double)n);

        double theory = 0.0;
        for (int k = 0; k < n; ++k)
            theory += ctx.lambda[k];
        theory /= (double)n;

        ASSERT_NEAR(emp_var_per_elem, theory, 0.05 * theory);
    }

    free(mu); free(eps); free(z_out);
    cheap_destroy(&ctx);
}

static void test_forward_inverse_roundtrip(void)
{
    printf("  test_forward_inverse_roundtrip\n");
    const int n = 64;
    cheap_ctx ctx;
    ASSERT_EQ(cheap_init(&ctx, n, 0.5), CHEAP_OK);

    double *input = (double *)malloc((size_t)n * sizeof(double));
    double *output = (double *)malloc((size_t)n * sizeof(double));

    for (int i = 0; i < n; ++i)
        input[i] = cos(2.0 * M_PI * (double)i / n) + 0.5 * sin(6.0 * M_PI * (double)i / n);

    /* forward then inverse should recover input */
    ASSERT_EQ(cheap_forward(&ctx, input), CHEAP_OK);
    ASSERT_EQ(cheap_inverse(&ctx, output), CHEAP_OK);

    for (int i = 0; i < n; ++i)
        ASSERT_NEAR(output[i], input[i], 1e-12);

    free(input); free(output);
    cheap_destroy(&ctx);
}

static void test_forward_inverse_vs_apply(void)
{
    printf("  test_forward_inverse_vs_apply\n");
    const int n = 64;
    cheap_ctx ctx;
    ASSERT_EQ(cheap_init(&ctx, n, 0.7), CHEAP_OK);

    double *input = (double *)malloc((size_t)n * sizeof(double));
    double *out_apply = (double *)malloc((size_t)n * sizeof(double));
    double *out_manual = (double *)malloc((size_t)n * sizeof(double));

    for (int i = 0; i < n; ++i)
        input[i] = sin(2.0 * M_PI * (double)i / n);

    /* Using cheap_apply */
    ASSERT_EQ(cheap_apply(&ctx, input, ctx.lambda, out_apply), CHEAP_OK);

    /* Using forward + manual multiply + inverse */
    ASSERT_EQ(cheap_forward(&ctx, input), CHEAP_OK);
    for (int k = 0; k < n; ++k)
        ctx.workspace[k] *= ctx.lambda[k];
    ASSERT_EQ(cheap_inverse(&ctx, out_manual), CHEAP_OK);

    for (int i = 0; i < n; ++i)
        ASSERT_NEAR(out_apply[i], out_manual[i], 1e-14);

    free(input); free(out_apply); free(out_manual);
    cheap_destroy(&ctx);
}

static void test_apply_fractional_roundtrip(void)
{
    printf("  test_apply_fractional_roundtrip\n");
    const int n = 128;
    cheap_ctx ctx;
    ASSERT_EQ(cheap_init(&ctx, n, 0.7), CHEAP_OK);

    double *z      = (double *)malloc((size_t)n * sizeof(double));
    double *z_diff = (double *)malloc((size_t)n * sizeof(double));
    double *z_back = (double *)malloc((size_t)n * sizeof(double));
    double *w_diff = (double *)malloc((size_t)n * sizeof(double));
    double *w_int  = (double *)malloc((size_t)n * sizeof(double));

    for (int i = 0; i < n; ++i)
        z[i] = sin(2.0 * M_PI * (double)i / (double)n);

    const double d = 0.3;
    /* Compute fractional diff weights */
    for (int k = 0; k < n; ++k) {
        double omega = M_PI * (double)k / (double)n;
        double sin_half = sin(0.5 * omega);
        if (sin_half < CHEAP_EPS_LOG) sin_half = CHEAP_EPS_LOG;
        w_diff[k] = pow(2.0 * sin_half, d);
        w_int[k]  = pow(2.0 * sin_half, -d);
    }

    ASSERT_EQ(cheap_apply(&ctx, z, w_diff, z_diff), CHEAP_OK);
    ASSERT_EQ(cheap_apply(&ctx, z_diff, w_int, z_back), CHEAP_OK);

    for (int i = 1; i < n; ++i)
        ASSERT_NEAR(z_back[i], z[i], 1e-8);

    /* Identity test: d=0 weights are all 1 */
    double *w_id = (double *)malloc((size_t)n * sizeof(double));
    double *z_id = (double *)malloc((size_t)n * sizeof(double));
    for (int k = 0; k < n; ++k) w_id[k] = 1.0;
    ASSERT_EQ(cheap_apply(&ctx, z, w_id, z_id), CHEAP_OK);
    for (int i = 1; i < n; ++i)
        ASSERT_NEAR(z_id[i], z[i], 1e-12);

    free(z); free(z_diff); free(z_back); free(w_diff); free(w_int);
    free(w_id); free(z_id);
    cheap_destroy(&ctx);
}

/* =========================================================================
 * Section 3: Sinkhorn tests
 * ========================================================================= */

static void test_sinkhorn_convergence(void)
{
    printf("  test_sinkhorn_convergence\n");
    const int n = 32;
    cheap_ctx ctx;
    ASSERT_EQ(cheap_init(&ctx, n, 0.6), CHEAP_OK);

    double *a = (double *)malloc((size_t)n * sizeof(double));
    double *b = (double *)malloc((size_t)n * sizeof(double));
    double *f = (double *)malloc((size_t)n * sizeof(double));
    double *g = (double *)malloc((size_t)n * sizeof(double));

    for (int i = 0; i < n; ++i) a[i] = b[i] = 1.0 / (double)n;

    int ret = cheap_sinkhorn(&ctx, a, b, 0.5, 500, 1e-6, f, g);
    ASSERT_EQ(ret, CHEAP_OK);

    for (int i = 0; i < n; ++i) {
        ASSERT_TRUE(isfinite(f[i]));
        ASSERT_TRUE(isfinite(g[i]));
    }

    ASSERT_EQ(cheap_sinkhorn(&ctx, a, b, 0.5, 0, 1e-15, f, g),
              CHEAP_ENOCONV);

    free(a); free(b); free(f); free(g);
    cheap_destroy(&ctx);
}

/* =========================================================================
 * Section 4: Toeplitz tests
 * ========================================================================= */

static void test_toeplitz_eigenvalues(void)
{
    printf("  test_toeplitz_eigenvalues\n");
    const int n = 64;
    cheap_ctx ctx;
    ASSERT_EQ(cheap_init(&ctx, n, 0.5), CHEAP_OK);

    double *t = (double*)calloc((size_t)n, sizeof(double));
    double *lam = (double*)malloc((size_t)n * sizeof(double));
    t[0] = 2.0; t[1] = -1.0;

    ASSERT_EQ(cheap_toeplitz_eigenvalues(&ctx, t, lam), CHEAP_OK);

    for (int k = 0; k < n; ++k)
        ASSERT_TRUE(isfinite(lam[k]));

    free(t); free(lam);
    cheap_destroy(&ctx);
}

static void test_toeplitz_matvec_via_apply(void)
{
    printf("  test_toeplitz_matvec_via_apply\n");
    const int n = 64;
    cheap_ctx ctx;
    ASSERT_EQ(cheap_init(&ctx, n, 0.5), CHEAP_OK);

    double *t = (double*)calloc((size_t)n, sizeof(double));
    double *lam = (double*)malloc((size_t)n * sizeof(double));
    double *x = (double*)malloc((size_t)n * sizeof(double));
    double *Tx = (double*)malloc((size_t)n * sizeof(double));
    double *x_back = (double*)malloc((size_t)n * sizeof(double));

    t[0] = 4.0; t[1] = -1.0;
    for (int i = 0; i < n; ++i)
        x[i] = cos(2.0 * M_PI * (double)i / (double)n) + 1.0;

    /* Matvec via eigenvalues + apply */
    ASSERT_EQ(cheap_toeplitz_eigenvalues(&ctx, t, lam), CHEAP_OK);
    ASSERT_EQ(cheap_apply(&ctx, x, lam, Tx), CHEAP_OK);

    for (int i = 0; i < n; ++i)
        ASSERT_TRUE(isfinite(Tx[i]));

    /* Solve and check roundtrip */
    const double eps = 1e-6;
    ASSERT_EQ(cheap_toeplitz_solve_precomp(&ctx, lam, Tx, eps, x_back), CHEAP_OK);
    for (int i = 0; i < n; ++i)
        ASSERT_NEAR(x_back[i], x[i], 1e-3);

    free(t); free(lam); free(x); free(Tx); free(x_back);
    cheap_destroy(&ctx);
}

static void test_toeplitz_solve_residual(void)
{
    printf("  test_toeplitz_solve_residual\n");
    const int n = 64;
    cheap_ctx ctx;
    ASSERT_EQ(cheap_init(&ctx, n, 0.5), CHEAP_OK);

    double *t = (double*)calloc((size_t)n, sizeof(double));
    double *lam = (double*)malloc((size_t)n * sizeof(double));
    double *y = (double*)malloc((size_t)n * sizeof(double));
    double *x_sol = (double*)malloc((size_t)n * sizeof(double));
    double *Tx = (double*)malloc((size_t)n * sizeof(double));

    t[0] = 4.0; t[1] = -1.0;
    for (int i = 0; i < n; ++i)
        y[i] = cos(2.0 * M_PI * (double)i / (double)n) + 2.0;

    const double lambda_reg = 0.1;
    ASSERT_EQ(cheap_toeplitz_eigenvalues(&ctx, t, lam), CHEAP_OK);
    ASSERT_EQ(cheap_toeplitz_solve_precomp(&ctx, lam, y, lambda_reg, x_sol), CHEAP_OK);

    ASSERT_EQ(cheap_apply(&ctx, x_sol, lam, Tx), CHEAP_OK);
    for (int i = 0; i < n; ++i) {
        double res = Tx[i] + lambda_reg * x_sol[i] - y[i];
        ASSERT_NEAR(res, 0.0, 1e-4);
    }

    free(t); free(lam); free(y); free(x_sol); free(Tx);
    cheap_destroy(&ctx);
}

static void test_toeplitz_solve_vs_apply_krr(void)
{
    printf("  test_toeplitz_solve_vs_apply_krr\n");
    const int n = 64;
    cheap_ctx ctx;
    ASSERT_EQ(cheap_init(&ctx, n, 0.7), CHEAP_OK);

    double *y = (double*)malloc((size_t)n * sizeof(double));
    double *alpha_apply = (double*)malloc((size_t)n * sizeof(double));
    double *alpha_toep = (double*)malloc((size_t)n * sizeof(double));
    double *w_krr = (double*)malloc((size_t)n * sizeof(double));

    for (int i = 0; i < n; ++i)
        y[i] = sin(2.0 * M_PI * (double)i / (double)n) + 0.5;

    const double lambda_reg = 1e-2;

    /* Via cheap_apply with KRR weights */
    for (int k = 0; k < n; ++k) {
        double denom = ctx.lambda[k] + lambda_reg;
        if (denom < CHEAP_EPS_DIV) denom = CHEAP_EPS_DIV;
        w_krr[k] = 1.0 / denom;
    }
    ASSERT_EQ(cheap_apply(&ctx, y, w_krr, alpha_apply), CHEAP_OK);

    /* Via cheap_toeplitz_solve_precomp */
    ASSERT_EQ(cheap_toeplitz_solve_precomp(&ctx, ctx.lambda, y,
                                              lambda_reg, alpha_toep), CHEAP_OK);

    for (int i = 0; i < n; ++i)
        ASSERT_NEAR(alpha_apply[i], alpha_toep[i], 1e-14);

    free(y); free(alpha_apply); free(alpha_toep); free(w_krr);
    cheap_destroy(&ctx);
}

static void test_toeplitz_error_codes(void)
{
    printf("  test_toeplitz_error_codes\n");
    cheap_ctx ctx;
    ASSERT_EQ(cheap_init(&ctx, 8, 0.5), CHEAP_OK);
    double t[8] = {2,-1,0,0,0,0,0,0};
    double x[8] = {1,2,3,4,5,6,7,8};
    double y[8];

    ASSERT_EQ(cheap_toeplitz_eigenvalues(NULL, t, y), CHEAP_EINVAL);
    ASSERT_EQ(cheap_toeplitz_eigenvalues(&ctx, NULL, y), CHEAP_EINVAL);
    ASSERT_EQ(cheap_toeplitz_eigenvalues(&ctx, t, NULL), CHEAP_EINVAL);

    double t_nan[8] = {2,-1,0,0,0,0,0,(double)NAN};
    ASSERT_EQ(cheap_toeplitz_eigenvalues(&ctx, t_nan, y), CHEAP_EDOM);

    ASSERT_EQ(cheap_toeplitz_solve_precomp(NULL, t, x, 0.1, y), CHEAP_EINVAL);
    ASSERT_EQ(cheap_toeplitz_solve_precomp(&ctx, NULL, x, 0.1, y), CHEAP_EINVAL);
    ASSERT_EQ(cheap_toeplitz_solve_precomp(&ctx, t, NULL, 0.1, y), CHEAP_EINVAL);
    ASSERT_EQ(cheap_toeplitz_solve_precomp(&ctx, t, x, -1.0, y), CHEAP_EINVAL);
    ASSERT_EQ(cheap_toeplitz_solve_precomp(&ctx, t, x, 0.1, NULL), CHEAP_EINVAL);

    double x_nan[8] = {1,2,3,4,5,6,7,(double)NAN};
    ASSERT_EQ(cheap_toeplitz_solve_precomp(&ctx, t, x_nan, 0.1, y), CHEAP_EDOM);

    cheap_destroy(&ctx);
}

/* =========================================================================
 * Section 5: RFF tests
 * ========================================================================= */

static void test_rff_init_destroy(void)
{
    printf("  test_rff_init_destroy\n");
    cheap_rff_ctx rctx;

    ASSERT_EQ(cheap_rff_init(&rctx, 64, 1, 1.0, 42), CHEAP_OK);
    ASSERT_TRUE(rctx.is_initialized == 1);
    ASSERT_TRUE(rctx.D == 64);
    ASSERT_TRUE(rctx.d_in == 1);
    cheap_rff_destroy(&rctx);
    ASSERT_TRUE(rctx.is_initialized == 0);

    ASSERT_EQ(cheap_rff_init(NULL, 64, 1, 1.0, 42), CHEAP_EINVAL);
    ASSERT_EQ(cheap_rff_init(&rctx, 3, 1, 1.0, 42), CHEAP_EINVAL);
    ASSERT_EQ(cheap_rff_init(&rctx, 0, 1, 1.0, 42), CHEAP_EINVAL);
    ASSERT_EQ(cheap_rff_init(&rctx, 64, 0, 1.0, 42), CHEAP_EINVAL);
    ASSERT_EQ(cheap_rff_init(&rctx, 64, 1, 0.0, 42), CHEAP_EINVAL);

    cheap_rff_destroy(NULL);
}

static void test_rff_kernel_approx(void)
{
    printf("  test_rff_kernel_approx\n");
    const int D = 512;
    const double sigma = 1.0;
    cheap_rff_ctx rctx;
    ASSERT_EQ(cheap_rff_init(&rctx, D, 1, sigma, 12345), CHEAP_OK);

    double zx[512], zy[512];
    double test_pts[] = {0.0, 0.1, 0.5, 1.0, 2.0};
    const int npts = 5;

    for (int p = 0; p < npts; ++p) {
        double x_val = 0.0, y_val = test_pts[p];
        ASSERT_EQ(cheap_rff_map(&rctx, &x_val, zx), CHEAP_OK);
        ASSERT_EQ(cheap_rff_map(&rctx, &y_val, zy), CHEAP_OK);

        double approx = 0.0;
        for (int i = 0; i < D; ++i) approx += zx[i] * zy[i];

        double d2 = (x_val - y_val) * (x_val - y_val);
        double exact = exp(-d2 / (2.0 * sigma * sigma));

        ASSERT_NEAR(approx, exact, 0.1);
    }

    cheap_rff_destroy(&rctx);
}

static void test_rff_seed_determinism(void)
{
    printf("  test_rff_seed_determinism\n");
    cheap_rff_ctx r1, r2;
    ASSERT_EQ(cheap_rff_init(&r1, 32, 1, 2.0, 9999), CHEAP_OK);
    ASSERT_EQ(cheap_rff_init(&r2, 32, 1, 2.0, 9999), CHEAP_OK);

    for (int i = 0; i < 16; ++i) {
        ASSERT_NEAR(r1.omega[i], r2.omega[i], 0.0);
        ASSERT_NEAR(r1.bias[i], r2.bias[i], 0.0);
    }

    cheap_rff_destroy(&r1);
    cheap_rff_destroy(&r2);
}

static void test_rff_batch_consistency(void)
{
    printf("  test_rff_batch_consistency\n");
    const int D = 32, N = 16;
    cheap_rff_ctx rctx;
    ASSERT_EQ(cheap_rff_init(&rctx, D, 1, 1.0, 42), CHEAP_OK);

    double *X = (double*)malloc((size_t)N * sizeof(double));
    double *Z_batch = (double*)malloc((size_t)(N * D) * sizeof(double));
    double *z_single = (double*)malloc((size_t)D * sizeof(double));

    for (int i = 0; i < N; ++i) X[i] = (double)i * 0.1;

    ASSERT_EQ(cheap_rff_map_batch(&rctx, X, N, Z_batch), CHEAP_OK);

    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(cheap_rff_map(&rctx, &X[i], z_single), CHEAP_OK);
        for (int j = 0; j < D; ++j)
            ASSERT_NEAR(Z_batch[i * D + j], z_single[j], 0.0);
    }

    free(X); free(Z_batch); free(z_single);
    cheap_rff_destroy(&rctx);
}

/* =========================================================================
 * Section 6: Edge cases
 * ========================================================================= */

static void test_edge_cases(void)
{
    printf("  test_edge_cases\n");

    /* Minimum n=2 */
    cheap_ctx ctx;
    ASSERT_EQ(cheap_init(&ctx, 2, 0.5), CHEAP_OK);
    double y2[2] = {1.0, 2.0};
    double w2[2], a2[2];
    for (int k = 0; k < 2; ++k) {
        double denom = ctx.lambda[k] + 0.01;
        if (denom < CHEAP_EPS_DIV) denom = CHEAP_EPS_DIV;
        w2[k] = 1.0 / denom;
    }
    ASSERT_EQ(cheap_apply(&ctx, y2, w2, a2), CHEAP_OK);
    cheap_destroy(&ctx);

    /* NULL destroy is safe */
    cheap_destroy(NULL);
    ASSERT_TRUE(1);
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(void)
{
    printf("=== CHEAP %s tests ===\n", CHEAP_VERSION);

    /* Core lifecycle */
    test_init_destroy();
    test_error_codes();

    /* Spectral primitives */
    test_apply_identity();
    test_apply_krr();
    test_apply_reparam();
    test_forward_inverse_roundtrip();
    test_forward_inverse_vs_apply();
    test_apply_fractional_roundtrip();

    /* Sinkhorn */
    test_sinkhorn_convergence();

    /* Toeplitz */
    test_toeplitz_eigenvalues();
    test_toeplitz_matvec_via_apply();
    test_toeplitz_solve_residual();
    test_toeplitz_solve_vs_apply_krr();
    test_toeplitz_error_codes();

    /* RFF */
    test_rff_init_destroy();
    test_rff_kernel_approx();
    test_rff_seed_determinism();
    test_rff_batch_consistency();

    /* Edge cases */
    test_edge_cases();

    printf("\n=== %d tests run, %d failed ===\n", g_tests_run, g_tests_failed);
    return (g_tests_failed > 0) ? 1 : 0;
}
