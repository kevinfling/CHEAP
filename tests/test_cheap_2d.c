/*
 * test_cheap_2d.c — correctness tests for the CHEAP 2D API.
 *
 * Phase 1.1 scope: cheap_ctx_2d struct + cheap_init_2d + cheap_destroy_2d.
 * Later phases will append forward/inverse/apply + weight ctor tests.
 */

#include "cheap.h"

#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Minimal test framework (mirrors test_cheap.c) */
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
        fprintf(stderr, "FAIL  %s:%d  ASSERT_EQ(%s, %s)  [%lld != %lld]\n", \
                __FILE__, __LINE__, #a, #b, (long long)(a), (long long)(b)); \
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

/* Reference Flandrin 1D spectrum, mirrors cheap__flandrin_1d_axis but kept
 * independent so the test is a true cross-check and not a tautology. */
static void ref_flandrin_1d(double *lam, int n, double H)
{
    const double p = pow((double)n, 2.0 * H);
    const double e = -(2.0 * H + 1.0);
    for (int k = 1; k < n; ++k) {
        double s = sin(M_PI * (double)k / (2.0 * (double)n));
        if (s < 1e-12) s = 1e-12;
        lam[k] = p * pow(s, e);
    }
    if (n >= 3) lam[0] = lam[1] * (lam[1] / lam[2]);
    else        lam[0] = 2.0 * lam[1];
}

static void test_init_2d_invalid_args(void)
{
    cheap_ctx_2d ctx = {0};
    ASSERT_EQ(cheap_init_2d(NULL, 16, 16, 0.5, 0.5), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_2d(&ctx,   1, 16, 0.5, 0.5), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_2d(&ctx,  16,  1, 0.5, 0.5), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_2d(&ctx,  16, 16, 0.0, 0.5), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_2d(&ctx,  16, 16, 1.0, 0.5), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_2d(&ctx,  16, 16, 0.5, 0.0), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_2d(&ctx,  16, 16, 0.5, 1.0), CHEAP_EINVAL);
    /* Failed init must not flip is_initialized */
    ASSERT_EQ(ctx.is_initialized, 0);
}

static void test_init_2d_fields(void)
{
    cheap_ctx_2d ctx = {0};
    ASSERT_EQ(cheap_init_2d(&ctx, 16, 32, 0.3, 0.7), CHEAP_OK);
    ASSERT_EQ(ctx.nx, 16);
    ASSERT_EQ(ctx.ny, 32);
    ASSERT_EQ(ctx.n,  16 * 32);
    ASSERT_EQ(ctx.is_initialized, 1);
    ASSERT_NEAR(ctx.current_Hx, 0.3, 1e-15);
    ASSERT_NEAR(ctx.current_Hy, 0.7, 1e-15);
    ASSERT_TRUE(ctx.lambda      != NULL);
    ASSERT_TRUE(ctx.sqrt_lambda != NULL);
    ASSERT_TRUE(ctx.gibbs       != NULL);
    ASSERT_TRUE(ctx.workspace   != NULL);
    ASSERT_TRUE(ctx.scratch1    != NULL);
    ASSERT_TRUE(ctx.scratch2    != NULL);
    ASSERT_TRUE(ctx.prev_g      != NULL);
    ASSERT_TRUE(ctx.plan_fwd    != NULL);
    ASSERT_TRUE(ctx.plan_inv    != NULL);
    cheap_destroy_2d(&ctx);
    ASSERT_EQ(ctx.is_initialized, 0);
    ASSERT_TRUE(ctx.lambda    == NULL);
    ASSERT_TRUE(ctx.workspace == NULL);
    ASSERT_TRUE(ctx.plan_fwd  == NULL);
}

static void test_init_2d_destroy_idempotent(void)
{
    cheap_ctx_2d ctx = {0};
    ASSERT_EQ(cheap_init_2d(&ctx, 8, 8, 0.5, 0.5), CHEAP_OK);
    cheap_destroy_2d(&ctx);
    cheap_destroy_2d(&ctx);   /* double-destroy must be safe */
    cheap_destroy_2d(NULL);   /* null-destroy must be safe */
    ASSERT_EQ(ctx.is_initialized, 0);
}

static void check_tensor_product_grid(int nx, int ny, double Hx, double Hy)
{
    cheap_ctx_2d ctx = {0};
    ASSERT_EQ(cheap_init_2d(&ctx, nx, ny, Hx, Hy), CHEAP_OK);

    double *lx = (double *)malloc((size_t)nx * sizeof(double));
    double *ly = (double *)malloc((size_t)ny * sizeof(double));
    ASSERT_TRUE(lx != NULL && ly != NULL);
    ref_flandrin_1d(lx, nx, Hx);
    ref_flandrin_1d(ly, ny, Hy);

    /* Sweep the full grid: every entry finite, positive, matches tensor product. */
    for (int j = 0; j < nx; ++j) {
        for (int k = 0; k < ny; ++k) {
            double got      = ctx.lambda[j * ny + k];
            double expected = lx[j] * ly[k];
            ASSERT_TRUE(isfinite(got));
            ASSERT_TRUE(got > 0.0);
            /* Relative tolerance; entries span many orders of magnitude. */
            double tol = 1e-10 * fmax(fabs(expected), 1.0);
            ASSERT_NEAR(got, expected, tol);
        }
    }

    /* sqrt_lambda = sqrt(max(lambda, EPS)); check a handful of positions. */
    for (int j = 0; j < nx; j += (nx > 8 ? nx / 4 : 1)) {
        for (int k = 0; k < ny; k += (ny > 8 ? ny / 4 : 1)) {
            double lam = ctx.lambda[j * ny + k];
            double s   = ctx.sqrt_lambda[j * ny + k];
            ASSERT_NEAR(s, sqrt(lam), 1e-12 * fmax(sqrt(lam), 1.0));
        }
    }

    free(lx);
    free(ly);
    cheap_destroy_2d(&ctx);
}

static void test_init_2d_tensor_product(void)
{
    const int sizes[][2] = { {16, 16}, {17, 31}, {64, 128}, {2, 2} };
    const double Hs[]    = { 0.3, 0.7 };
    for (unsigned si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
        for (unsigned hx = 0; hx < 2; ++hx) {
            for (unsigned hy = 0; hy < 2; ++hy) {
                check_tensor_product_grid(sizes[si][0], sizes[si][1],
                                          Hs[hx], Hs[hy]);
            }
        }
    }
}

int main(void)
{
    printf("=== test_cheap_2d ===\n");
    test_init_2d_invalid_args();       printf("  test_init_2d_invalid_args\n");
    test_init_2d_fields();             printf("  test_init_2d_fields\n");
    test_init_2d_destroy_idempotent(); printf("  test_init_2d_destroy_idempotent\n");
    test_init_2d_tensor_product();     printf("  test_init_2d_tensor_product\n");

    printf("\n=== %d tests run, %d failed ===\n", g_tests_run, g_tests_failed);
    return g_tests_failed == 0 ? 0 : 1;
}
