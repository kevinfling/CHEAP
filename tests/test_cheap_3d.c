/*
 * test_cheap_3d.c — correctness tests for the CHEAP 3D API.
 *
 * Phase 2.1 scope: cheap_ctx_3d struct + cheap_init_3d / destroy_3d +
 * cheap_init_from_eigenvalues_3d + cheap_init_from_toeplitz_3d.
 * Later phases append forward/inverse/apply + 3D weight ctor tests.
 */

#include "cheap.h"

#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Minimal test framework (mirrors test_cheap.c / test_cheap_2d.c) */
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

static void test_init_3d_invalid_args(void)
{
    cheap_ctx_3d ctx = {0};
    ASSERT_EQ(cheap_init_3d(NULL, 8, 8, 8, 0.5, 0.5, 0.5), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_3d(&ctx,  1, 8, 8, 0.5, 0.5, 0.5), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_3d(&ctx,  8, 1, 8, 0.5, 0.5, 0.5), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_3d(&ctx,  8, 8, 1, 0.5, 0.5, 0.5), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_3d(&ctx,  8, 8, 8, 0.0, 0.5, 0.5), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_3d(&ctx,  8, 8, 8, 1.0, 0.5, 0.5), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_3d(&ctx,  8, 8, 8, 0.5, 0.0, 0.5), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_3d(&ctx,  8, 8, 8, 0.5, 1.0, 0.5), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_3d(&ctx,  8, 8, 8, 0.5, 0.5, 0.0), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_3d(&ctx,  8, 8, 8, 0.5, 0.5, 1.0), CHEAP_EINVAL);
    ASSERT_EQ(ctx.is_initialized, 0);
}

static void test_init_3d_fields(void)
{
    cheap_ctx_3d ctx = {0};
    ASSERT_EQ(cheap_init_3d(&ctx, 8, 16, 4, 0.4, 0.5, 0.6), CHEAP_OK);
    ASSERT_EQ(ctx.nx, 8);
    ASSERT_EQ(ctx.ny, 16);
    ASSERT_EQ(ctx.nz, 4);
    ASSERT_EQ(ctx.n,  8 * 16 * 4);
    ASSERT_EQ(ctx.is_initialized, 1);
    ASSERT_NEAR(ctx.current_Hx, 0.4, 1e-15);
    ASSERT_NEAR(ctx.current_Hy, 0.5, 1e-15);
    ASSERT_NEAR(ctx.current_Hz, 0.6, 1e-15);
    ASSERT_TRUE(ctx.lambda      != NULL);
    ASSERT_TRUE(ctx.gibbs       != NULL);
    ASSERT_TRUE(ctx.sqrt_lambda != NULL);
    ASSERT_TRUE(ctx.workspace   != NULL);
    ASSERT_TRUE(ctx.scratch1    != NULL);
    ASSERT_TRUE(ctx.scratch2    != NULL);
    ASSERT_TRUE(ctx.prev_g      != NULL);
    ASSERT_TRUE(ctx.plan_fwd    != NULL);
    ASSERT_TRUE(ctx.plan_inv    != NULL);

    cheap_destroy_3d(&ctx);
    ASSERT_EQ(ctx.is_initialized, 0);
    ASSERT_TRUE(ctx.lambda      == NULL);
    ASSERT_TRUE(ctx.gibbs       == NULL);
    ASSERT_TRUE(ctx.sqrt_lambda == NULL);
    ASSERT_TRUE(ctx.workspace   == NULL);
    ASSERT_TRUE(ctx.scratch1    == NULL);
    ASSERT_TRUE(ctx.scratch2    == NULL);
    ASSERT_TRUE(ctx.prev_g      == NULL);
    ASSERT_TRUE(ctx.plan_fwd    == NULL);
    ASSERT_TRUE(ctx.plan_inv    == NULL);
}

static void test_init_3d_destroy_idempotent(void)
{
    cheap_ctx_3d ctx = {0};
    cheap_destroy_3d(NULL);    /* null-safe */
    cheap_destroy_3d(&ctx);    /* zero-init-safe (no-op) */
    ASSERT_EQ(cheap_init_3d(&ctx, 4, 4, 4, 0.5, 0.5, 0.5), CHEAP_OK);
    cheap_destroy_3d(&ctx);
    cheap_destroy_3d(&ctx);    /* double-destroy must be safe */
    ASSERT_EQ(ctx.is_initialized, 0);
}

static void test_init_3d_tensor_product(void)
{
    typedef struct { int nx, ny, nz; double Hx, Hy, Hz; } case_t;
    const case_t cases[] = {
        {4, 4, 4,  0.3, 0.7, 0.5},
        {8, 16, 4, 0.4, 0.6, 0.5},
        {15, 17, 31, 0.5, 0.5, 0.5},
        {2, 2, 2,  0.3, 0.7, 0.4},
    };
    for (size_t ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ++ci) {
        const int nx = cases[ci].nx, ny = cases[ci].ny, nz = cases[ci].nz;
        const double Hx = cases[ci].Hx, Hy = cases[ci].Hy, Hz = cases[ci].Hz;

        cheap_ctx_3d ctx = {0};
        ASSERT_EQ(cheap_init_3d(&ctx, nx, ny, nz, Hx, Hy, Hz), CHEAP_OK);

        double *lx = (double *)malloc((size_t)nx * sizeof(double));
        double *ly = (double *)malloc((size_t)ny * sizeof(double));
        double *lz = (double *)malloc((size_t)nz * sizeof(double));
        ref_flandrin_1d(lx, nx, Hx);
        ref_flandrin_1d(ly, ny, Hy);
        ref_flandrin_1d(lz, nz, Hz);

        double max_rel = 0.0;
        for (int j = 0; j < nx; ++j) {
            for (int k = 0; k < ny; ++k) {
                for (int l = 0; l < nz; ++l) {
                    double expected = lx[j] * ly[k] * lz[l];
                    double got = ctx.lambda[(j * ny + k) * nz + l];
                    double rel = fabs(got - expected) / fmax(fabs(expected), 1.0);
                    if (rel > max_rel) max_rel = rel;
                    ASSERT_TRUE(isfinite(got));
                    ASSERT_TRUE(got > 0.0);
                }
            }
        }
        ASSERT_TRUE(max_rel < 1e-10);

        /* sqrt_lambda consistent */
        for (int i = 0; i < ctx.n; ++i) {
            double s = sqrt(fmax(ctx.lambda[i], 1e-15));
            ASSERT_NEAR(ctx.sqrt_lambda[i], s, 1e-14 * s);
        }

        cheap_destroy_3d(&ctx);
        free(lx); free(ly); free(lz);
    }
}

static void test_init_from_eigenvalues_3d(void)
{
    const int nx = 4, ny = 8, nz = 3;
    const int n  = nx * ny * nz;
    double *lam = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i) lam[i] = 1.5 + 0.1 * (double)i;

    cheap_ctx_3d ctx = {0};
    ASSERT_EQ(cheap_init_from_eigenvalues_3d(&ctx, nx, ny, nz, lam), CHEAP_OK);
    ASSERT_EQ(ctx.is_initialized, 1);
    ASSERT_NEAR(ctx.current_Hx, -1.0, 1e-15);
    ASSERT_NEAR(ctx.current_Hy, -1.0, 1e-15);
    ASSERT_NEAR(ctx.current_Hz, -1.0, 1e-15);
    for (int i = 0; i < n; ++i)
        ASSERT_NEAR(ctx.lambda[i], lam[i], 1e-15);
    cheap_destroy_3d(&ctx);

    /* Bad args */
    ASSERT_EQ(cheap_init_from_eigenvalues_3d(NULL, nx, ny, nz, lam), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_eigenvalues_3d(&ctx,   1, ny, nz, lam), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_eigenvalues_3d(&ctx,  nx,  1, nz, lam), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_eigenvalues_3d(&ctx,  nx, ny,  1, lam), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_eigenvalues_3d(&ctx,  nx, ny, nz, NULL), CHEAP_EINVAL);
    lam[5] = (double)NAN;
    ASSERT_EQ(cheap_init_from_eigenvalues_3d(&ctx, nx, ny, nz, lam), CHEAP_EDOM);
    free(lam);
}

/* Reconstruct a Toeplitz first-column whose DCT-II eigenvalues are the
 * Flandrin spectrum — iDCT-III(lambda) / (2N). Verifies _from_toeplitz_3d
 * reproduces cheap_init_3d's lambda grid. */
static void test_init_from_toeplitz_3d_matches_flandrin(void)
{
    const int nx = 8, ny = 6, nz = 4;
    const double Hx = 0.4, Hy = 0.55, Hz = 0.65;

    double *lx = (double *)fftw_malloc((size_t)nx * sizeof(double));
    double *ly = (double *)fftw_malloc((size_t)ny * sizeof(double));
    double *lz = (double *)fftw_malloc((size_t)nz * sizeof(double));
    ref_flandrin_1d(lx, nx, Hx);
    ref_flandrin_1d(ly, ny, Hy);
    ref_flandrin_1d(lz, nz, Hz);

    /* Build per-axis Toeplitz first columns by iDCT-III(lambda) / (2N). */
    double *tx = (double *)fftw_malloc((size_t)nx * sizeof(double));
    double *ty = (double *)fftw_malloc((size_t)ny * sizeof(double));
    double *tz = (double *)fftw_malloc((size_t)nz * sizeof(double));
    memcpy(tx, lx, (size_t)nx * sizeof(double));
    memcpy(ty, ly, (size_t)ny * sizeof(double));
    memcpy(tz, lz, (size_t)nz * sizeof(double));
    fftw_plan px = fftw_plan_r2r_1d(nx, tx, tx, FFTW_REDFT01, FFTW_ESTIMATE);
    fftw_plan py = fftw_plan_r2r_1d(ny, ty, ty, FFTW_REDFT01, FFTW_ESTIMATE);
    fftw_plan pz = fftw_plan_r2r_1d(nz, tz, tz, FFTW_REDFT01, FFTW_ESTIMATE);
    fftw_execute(px); fftw_execute(py); fftw_execute(pz);
    fftw_destroy_plan(px); fftw_destroy_plan(py); fftw_destroy_plan(pz);
    for (int i = 0; i < nx; ++i) tx[i] /= (2.0 * (double)nx);
    for (int i = 0; i < ny; ++i) ty[i] /= (2.0 * (double)ny);
    for (int i = 0; i < nz; ++i) tz[i] /= (2.0 * (double)nz);

    cheap_ctx_3d a = {0}, b = {0};
    ASSERT_EQ(cheap_init_3d(&a, nx, ny, nz, Hx, Hy, Hz), CHEAP_OK);
    ASSERT_EQ(cheap_init_from_toeplitz_3d(&b, nx, ny, nz, tx, ty, tz), CHEAP_OK);
    ASSERT_NEAR(b.current_Hx, -1.0, 1e-15);
    ASSERT_NEAR(b.current_Hy, -1.0, 1e-15);
    ASSERT_NEAR(b.current_Hz, -1.0, 1e-15);

    double max_rel = 0.0;
    for (int i = 0; i < nx * ny * nz; ++i) {
        double got = b.lambda[i], expected = a.lambda[i];
        double rel = fabs(got - expected) / fmax(fabs(expected), 1.0);
        if (rel > max_rel) max_rel = rel;
    }
    ASSERT_TRUE(max_rel < 1e-10);

    cheap_destroy_3d(&a);
    cheap_destroy_3d(&b);
    fftw_free(lx); fftw_free(ly); fftw_free(lz);
    fftw_free(tx); fftw_free(ty); fftw_free(tz);
}

static void test_init_from_toeplitz_3d_bad_args(void)
{
    const int nx = 4, ny = 4, nz = 4;
    double *tx = (double *)malloc((size_t)nx * sizeof(double));
    double *ty = (double *)malloc((size_t)ny * sizeof(double));
    double *tz = (double *)malloc((size_t)nz * sizeof(double));
    for (int i = 0; i < nx; ++i) tx[i] = 1.0 / (1.0 + i);
    for (int i = 0; i < ny; ++i) ty[i] = 1.0 / (1.0 + i);
    for (int i = 0; i < nz; ++i) tz[i] = 1.0 / (1.0 + i);
    cheap_ctx_3d ctx = {0};
    ASSERT_EQ(cheap_init_from_toeplitz_3d(NULL, nx, ny, nz, tx, ty, tz), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_toeplitz_3d(&ctx,  1, ny, nz, tx, ty, tz), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_toeplitz_3d(&ctx, nx,  1, nz, tx, ty, tz), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_toeplitz_3d(&ctx, nx, ny,  1, tx, ty, tz), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_toeplitz_3d(&ctx, nx, ny, nz, NULL, ty, tz), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_toeplitz_3d(&ctx, nx, ny, nz, tx, NULL, tz), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_toeplitz_3d(&ctx, nx, ny, nz, tx, ty, NULL), CHEAP_EINVAL);
    tx[0] = (double)NAN;
    ASSERT_EQ(cheap_init_from_toeplitz_3d(&ctx, nx, ny, nz, tx, ty, tz), CHEAP_EDOM);
    free(tx); free(ty); free(tz);
}

/* ---------- Phase 2.2: forward / inverse / apply 3D ---------- */

static double max_abs_diff(const double *a, const double *b, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

typedef struct { int nx, ny, nz; } size_triple;
static const size_triple k_sizes_3d[] = {
    {4, 4, 4}, {8, 16, 32}, {15, 17, 31}, {32, 32, 32}
};
static const int k_num_sizes_3d = (int)(sizeof(k_sizes_3d) / sizeof(k_sizes_3d[0]));

static void fill_signal_3d(double *x, int nx, int ny, int nz, unsigned seed)
{
    for (int j = 0; j < nx; ++j) {
        for (int k = 0; k < ny; ++k) {
            for (int l = 0; l < nz; ++l) {
                double s = sin(0.01 * (seed + j * 17 + k * 31 + l * 7));
                double c = cos(0.013 * (seed * 3 + j * 7 - k + l * 11));
                x[(j * ny + k) * nz + l] = s + 0.5 * c;
            }
        }
    }
}

static void test_apply_3d_identity(void)
{
    for (int s = 0; s < k_num_sizes_3d; ++s) {
        const int nx = k_sizes_3d[s].nx;
        const int ny = k_sizes_3d[s].ny;
        const int nz = k_sizes_3d[s].nz;
        const int n  = nx * ny * nz;

        cheap_ctx_3d ctx = {0};
        ASSERT_EQ(cheap_init_3d(&ctx, nx, ny, nz, 0.5, 0.5, 0.5), CHEAP_OK);

        double *x = (double *)malloc((size_t)n * sizeof(double));
        double *w = (double *)malloc((size_t)n * sizeof(double));
        double *y = (double *)malloc((size_t)n * sizeof(double));
        fill_signal_3d(x, nx, ny, nz, 7u);
        for (int i = 0; i < n; ++i) w[i] = 1.0;

        ASSERT_EQ(cheap_apply_3d(&ctx, x, w, y), CHEAP_OK);
        ASSERT_TRUE(max_abs_diff(x, y, n) < 1e-12);

        cheap_destroy_3d(&ctx);
        free(x); free(w); free(y);
    }
}

static void test_forward_inverse_3d_roundtrip(void)
{
    for (int s = 0; s < k_num_sizes_3d; ++s) {
        const int nx = k_sizes_3d[s].nx;
        const int ny = k_sizes_3d[s].ny;
        const int nz = k_sizes_3d[s].nz;
        const int n  = nx * ny * nz;

        cheap_ctx_3d ctx = {0};
        ASSERT_EQ(cheap_init_3d(&ctx, nx, ny, nz, 0.4, 0.6, 0.5), CHEAP_OK);

        double *x = (double *)malloc((size_t)n * sizeof(double));
        double *y = (double *)malloc((size_t)n * sizeof(double));
        fill_signal_3d(x, nx, ny, nz, 13u);

        ASSERT_EQ(cheap_forward_3d(&ctx, x), CHEAP_OK);
        ASSERT_EQ(cheap_inverse_3d(&ctx, y), CHEAP_OK);
        ASSERT_TRUE(max_abs_diff(x, y, n) < 1e-12);

        cheap_destroy_3d(&ctx);
        free(x); free(y);
    }
}

static void test_apply_inplace_3d_matches_apply(void)
{
    for (int s = 0; s < k_num_sizes_3d; ++s) {
        const int nx = k_sizes_3d[s].nx;
        const int ny = k_sizes_3d[s].ny;
        const int nz = k_sizes_3d[s].nz;
        const int n  = nx * ny * nz;

        cheap_ctx_3d ctx = {0};
        ASSERT_EQ(cheap_init_3d(&ctx, nx, ny, nz, 0.3, 0.7, 0.5), CHEAP_OK);

        double *x  = (double *)malloc((size_t)n * sizeof(double));
        double *w  = (double *)malloc((size_t)n * sizeof(double));
        double *y1 = (double *)malloc((size_t)n * sizeof(double));
        double *y2 = (double *)malloc((size_t)n * sizeof(double));
        fill_signal_3d(x, nx, ny, nz, 29u);
        for (int i = 0; i < n; ++i) w[i] = 1.0 / (ctx.lambda[i] + 1.0);

        ASSERT_EQ(cheap_apply_3d(&ctx, x, w, y1), CHEAP_OK);

        memcpy(cheap_workspace_3d(&ctx), x, (size_t)n * sizeof(double));
        ASSERT_EQ(cheap_apply_inplace_3d(&ctx, w), CHEAP_OK);
        memcpy(y2, cheap_workspace_3d(&ctx), (size_t)n * sizeof(double));

        ASSERT_TRUE(max_abs_diff(y1, y2, n) < 1e-12);

        cheap_destroy_3d(&ctx);
        free(x); free(w); free(y1); free(y2);
    }
}

static void test_forward_inverse_inplace_3d_equivalence(void)
{
    for (int s = 0; s < k_num_sizes_3d; ++s) {
        const int nx = k_sizes_3d[s].nx;
        const int ny = k_sizes_3d[s].ny;
        const int nz = k_sizes_3d[s].nz;
        const int n  = nx * ny * nz;

        cheap_ctx_3d ctx = {0};
        ASSERT_EQ(cheap_init_3d(&ctx, nx, ny, nz, 0.5, 0.5, 0.5), CHEAP_OK);

        double *x  = (double *)malloc((size_t)n * sizeof(double));
        double *yf = (double *)malloc((size_t)n * sizeof(double));
        double *yi = (double *)malloc((size_t)n * sizeof(double));
        fill_signal_3d(x, nx, ny, nz, 41u);

        ASSERT_EQ(cheap_forward_3d(&ctx, x), CHEAP_OK);
        memcpy(yf, cheap_workspace_3d(&ctx), (size_t)n * sizeof(double));

        memcpy(cheap_workspace_3d(&ctx), x, (size_t)n * sizeof(double));
        ASSERT_EQ(cheap_forward_inplace_3d(&ctx), CHEAP_OK);
        memcpy(yi, cheap_workspace_3d(&ctx), (size_t)n * sizeof(double));
        ASSERT_TRUE(max_abs_diff(yf, yi, n) < 1e-12);

        memcpy(cheap_workspace_3d(&ctx), yf, (size_t)n * sizeof(double));
        ASSERT_EQ(cheap_inverse_3d(&ctx, yi), CHEAP_OK);

        memcpy(cheap_workspace_3d(&ctx), yf, (size_t)n * sizeof(double));
        ASSERT_EQ(cheap_inverse_inplace_3d(&ctx), CHEAP_OK);
        ASSERT_TRUE(max_abs_diff(yi, cheap_workspace_3d(&ctx), n) < 1e-12);

        cheap_destroy_3d(&ctx);
        free(x); free(yf); free(yi);
    }
}

static void test_apply_3d_bad_args(void)
{
    const int nx = 8, ny = 8, nz = 8, n = nx * ny * nz;
    cheap_ctx_3d ctx = {0};
    ASSERT_EQ(cheap_init_3d(&ctx, nx, ny, nz, 0.5, 0.5, 0.5), CHEAP_OK);

    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *w = (double *)malloc((size_t)n * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    fill_signal_3d(x, nx, ny, nz, 3u);
    for (int i = 0; i < n; ++i) w[i] = 1.0;

    ASSERT_EQ(cheap_apply_3d(NULL, x, w, y), CHEAP_EUNINIT);
    cheap_ctx_3d zero = {0};
    ASSERT_EQ(cheap_apply_3d(&zero, x, w, y), CHEAP_EUNINIT);
    ASSERT_EQ(cheap_forward_3d(&zero, x), CHEAP_EUNINIT);
    ASSERT_EQ(cheap_inverse_3d(&zero, y), CHEAP_EUNINIT);
    ASSERT_EQ(cheap_apply_inplace_3d(&zero, w), CHEAP_EUNINIT);
    ASSERT_EQ(cheap_forward_inplace_3d(&zero), CHEAP_EUNINIT);
    ASSERT_EQ(cheap_inverse_inplace_3d(&zero), CHEAP_EUNINIT);

    ASSERT_EQ(cheap_apply_3d(&ctx, NULL, w, y), CHEAP_EINVAL);
    ASSERT_EQ(cheap_apply_3d(&ctx, x, NULL, y), CHEAP_EINVAL);
    ASSERT_EQ(cheap_apply_3d(&ctx, x, w, NULL), CHEAP_EINVAL);
    ASSERT_EQ(cheap_apply_inplace_3d(&ctx, NULL), CHEAP_EINVAL);
    ASSERT_EQ(cheap_forward_3d(&ctx, NULL), CHEAP_EINVAL);
    ASSERT_EQ(cheap_inverse_3d(&ctx, NULL), CHEAP_EINVAL);

    x[5] = (double)NAN;
    ASSERT_EQ(cheap_apply_3d(&ctx, x, w, y), CHEAP_EDOM);
    ASSERT_EQ(cheap_forward_3d(&ctx, x), CHEAP_EDOM);

    ASSERT_TRUE(cheap_workspace_3d(NULL) == NULL);
    ASSERT_TRUE(cheap_workspace_3d(&zero) == NULL);
    ASSERT_TRUE(cheap_workspace_3d(&ctx) == ctx.workspace);

    cheap_destroy_3d(&ctx);
    free(x); free(w); free(y);
}

/* ---------- Phase 2.3: 3D weight constructors ---------- */

static void test_weights_laplacian_3d(void)
{
    const int nx = 64, ny = 48, nz = 32;
    const int n  = nx * ny * nz;
    double *w = (double *)malloc((size_t)n * sizeof(double));
    ASSERT_EQ(cheap_weights_laplacian_3d(nx, ny, nz, w), CHEAP_OK);

    ASSERT_NEAR(w[0], 0.0, 0.0);
    int all_finite = 1, all_nonneg = 1;
    for (int i = 0; i < n; ++i) {
        if (!isfinite(w[i])) all_finite = 0;
        if (w[i] < 0.0) all_nonneg = 0;
    }
    ASSERT_TRUE(all_finite);
    ASSERT_TRUE(all_nonneg);

    const int probes[][3] = {{0,0,0}, {0,5,0}, {7,0,0}, {0,0,3}, {13,21,11}, {nx-1,ny-1,nz-1}};
    for (size_t p = 0; p < sizeof(probes)/sizeof(probes[0]); ++p) {
        int j = probes[p][0], k = probes[p][1], l = probes[p][2];
        double sx = sin(M_PI * (double)j / (2.0 * (double)nx));
        double sy = sin(M_PI * (double)k / (2.0 * (double)ny));
        double sz = sin(M_PI * (double)l / (2.0 * (double)nz));
        double expected = 4.0 * (sx * sx + sy * sy + sz * sz);
        ASSERT_NEAR(w[(j * ny + k) * nz + l], expected, 1e-14);
    }

    for (int j = 1; j < nx; ++j) {
        ASSERT_TRUE(w[j * ny * nz] >= w[(j - 1) * ny * nz] - 1e-15);
    }
    free(w);
}

static void test_weights_fractional_laplacian_3d_identity(void)
{
    const int nx = 16, ny = 16, nz = 16;
    const int n = nx * ny * nz;
    double *w0 = (double *)malloc((size_t)n * sizeof(double));
    ASSERT_EQ(cheap_weights_fractional_laplacian_3d(nx, ny, nz, 0.0, w0), CHEAP_OK);
    for (int i = 0; i < n; ++i) ASSERT_NEAR(w0[i], 1.0, 1e-15);
    free(w0);
}

static void test_weights_fractional_laplacian_3d_matches_laplacian(void)
{
    const int nx = 16, ny = 12, nz = 8;
    const int n = nx * ny * nz;
    double *wL = (double *)malloc((size_t)n * sizeof(double));
    double *wF = (double *)malloc((size_t)n * sizeof(double));
    ASSERT_EQ(cheap_weights_laplacian_3d(nx, ny, nz, wL), CHEAP_OK);
    ASSERT_EQ(cheap_weights_fractional_laplacian_3d(nx, ny, nz, 1.0, wF), CHEAP_OK);
    for (int i = 1; i < n; ++i) ASSERT_NEAR(wF[i], wL[i], 1e-12);
    free(wL); free(wF);
}

static void test_weights_fractional_laplacian_3d_power_identity(void)
{
    const int nx = 8, ny = 10, nz = 6;
    const int n = nx * ny * nz;
    const double alphas[] = {0.2, 0.5, 1.3, 2.0, -0.75};
    double *wL = (double *)malloc((size_t)n * sizeof(double));
    double *wF = (double *)malloc((size_t)n * sizeof(double));
    ASSERT_EQ(cheap_weights_laplacian_3d(nx, ny, nz, wL), CHEAP_OK);
    for (size_t ai = 0; ai < sizeof(alphas)/sizeof(alphas[0]); ++ai) {
        double alpha = alphas[ai];
        ASSERT_EQ(cheap_weights_fractional_laplacian_3d(nx, ny, nz, alpha, wF), CHEAP_OK);
        for (int i = 1; i < n; ++i) {
            double recovered = pow(wF[i], 1.0 / alpha);
            double rel = fabs(recovered - wL[i]) / fmax(fabs(wL[i]), 1.0);
            ASSERT_TRUE(rel < 1e-10);
        }
    }
    free(wL); free(wF);
}

static void test_weights_3d_bad_args(void)
{
    double w[64];
    ASSERT_EQ(cheap_weights_laplacian_3d(1, 4, 4, w), CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_laplacian_3d(4, 1, 4, w), CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_laplacian_3d(4, 4, 1, w), CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_laplacian_3d(4, 4, 4, NULL), CHEAP_EINVAL);

    ASSERT_EQ(cheap_weights_fractional_laplacian_3d(1, 4, 4, 0.5, w), CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_fractional_laplacian_3d(4, 4, 4, 0.5, NULL), CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_fractional_laplacian_3d(4, 4, 4, (double)NAN, w), CHEAP_EINVAL);
}

int main(void)
{
    printf("=== test_cheap_3d ===\n");
    test_init_3d_invalid_args();                  printf("  test_init_3d_invalid_args\n");
    test_init_3d_fields();                        printf("  test_init_3d_fields\n");
    test_init_3d_destroy_idempotent();            printf("  test_init_3d_destroy_idempotent\n");
    test_init_3d_tensor_product();                printf("  test_init_3d_tensor_product\n");
    test_init_from_eigenvalues_3d();              printf("  test_init_from_eigenvalues_3d\n");
    test_init_from_toeplitz_3d_matches_flandrin();printf("  test_init_from_toeplitz_3d_matches_flandrin\n");
    test_init_from_toeplitz_3d_bad_args();        printf("  test_init_from_toeplitz_3d_bad_args\n");
    test_apply_3d_identity();                     printf("  test_apply_3d_identity\n");
    test_forward_inverse_3d_roundtrip();          printf("  test_forward_inverse_3d_roundtrip\n");
    test_apply_inplace_3d_matches_apply();        printf("  test_apply_inplace_3d_matches_apply\n");
    test_forward_inverse_inplace_3d_equivalence();printf("  test_forward_inverse_inplace_3d_equivalence\n");
    test_apply_3d_bad_args();                     printf("  test_apply_3d_bad_args\n");
    test_weights_laplacian_3d();                  printf("  test_weights_laplacian_3d\n");
    test_weights_fractional_laplacian_3d_identity();printf("  test_weights_fractional_laplacian_3d_identity\n");
    test_weights_fractional_laplacian_3d_matches_laplacian();printf("  test_weights_fractional_laplacian_3d_matches_laplacian\n");
    test_weights_fractional_laplacian_3d_power_identity();printf("  test_weights_fractional_laplacian_3d_power_identity\n");
    test_weights_3d_bad_args();                   printf("  test_weights_3d_bad_args\n");

    printf("\n=== %d tests run, %d failed ===\n", g_tests_run, g_tests_failed);
    return g_tests_failed == 0 ? 0 : 1;
}
