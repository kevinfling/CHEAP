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

/* ---------- Phase 1.2 ---------- */

static void test_init_from_eigenvalues_2d(void)
{
    const int nx = 17, ny = 13;
    const int n  = nx * ny;
    double *lam = (double *)malloc((size_t)n * sizeof(double));
    ASSERT_TRUE(lam != NULL);
    for (int i = 0; i < n; ++i) lam[i] = 1.0 + 0.5 * sin(0.37 * i);

    cheap_ctx_2d ctx = {0};
    ASSERT_EQ(cheap_init_from_eigenvalues_2d(&ctx, nx, ny, lam), CHEAP_OK);
    ASSERT_EQ(ctx.nx, nx);
    ASSERT_EQ(ctx.ny, ny);
    ASSERT_NEAR(ctx.current_Hx, -1.0, 1e-15);
    ASSERT_NEAR(ctx.current_Hy, -1.0, 1e-15);
    for (int i = 0; i < n; ++i) ASSERT_NEAR(ctx.lambda[i], lam[i], 1e-15);
    for (int i = 0; i < n; ++i)
        ASSERT_NEAR(ctx.sqrt_lambda[i], sqrt(fmax(lam[i], 1e-15)), 1e-14);
    cheap_destroy_2d(&ctx);

    /* Bad args */
    cheap_ctx_2d bad = {0};
    ASSERT_EQ(cheap_init_from_eigenvalues_2d(NULL, nx, ny, lam), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_eigenvalues_2d(&bad,   1, ny, lam), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_eigenvalues_2d(&bad,  nx,  1, lam), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_eigenvalues_2d(&bad,  nx, ny, NULL), CHEAP_EINVAL);
    lam[5] = (double)INFINITY;
    ASSERT_EQ(cheap_init_from_eigenvalues_2d(&bad,  nx, ny, lam), CHEAP_EDOM);

    free(lam);
}

static void test_init_from_toeplitz_2d_matches_flandrin(void)
{
    /* Strategy: build a Flandrin 1D spectrum along each axis, transform it
     * back to a Toeplitz first column via iDCT-III, then feed those columns
     * into cheap_init_from_toeplitz_2d. The resulting 2D eigenvalues must
     * match cheap_init_2d (tensor-product Flandrin) to near float64 epsilon. */
    const int nx = 32, ny = 48;
    const double Hx = 0.4, Hy = 0.7;

    double *lx = (double *)fftw_malloc((size_t)nx * sizeof(double));
    double *ly = (double *)fftw_malloc((size_t)ny * sizeof(double));
    double *tx = (double *)fftw_malloc((size_t)nx * sizeof(double));
    double *ty = (double *)fftw_malloc((size_t)ny * sizeof(double));
    ASSERT_TRUE(lx && ly && tx && ty);

    ref_flandrin_1d(lx, nx, Hx);
    ref_flandrin_1d(ly, ny, Hy);

    /* First column t = iDCT-III(lambda) / (2N): DCT-II is its own inverse
     * up to this normalization, so running the inverse plan on lambda
     * recovers the Toeplitz first column whose DCT-II eigenvalues are
     * lambda itself. */
    memcpy(tx, lx, (size_t)nx * sizeof(double));
    memcpy(ty, ly, (size_t)ny * sizeof(double));
    fftw_plan pix = fftw_plan_r2r_1d(nx, tx, tx, FFTW_REDFT01, FFTW_ESTIMATE);
    fftw_plan piy = fftw_plan_r2r_1d(ny, ty, ty, FFTW_REDFT01, FFTW_ESTIMATE);
    fftw_execute(pix); fftw_execute(piy);
    fftw_destroy_plan(pix); fftw_destroy_plan(piy);
    for (int i = 0; i < nx; ++i) tx[i] /= (2.0 * (double)nx);
    for (int i = 0; i < ny; ++i) ty[i] /= (2.0 * (double)ny);

    cheap_ctx_2d a = {0}, b = {0};
    ASSERT_EQ(cheap_init_2d(&a, nx, ny, Hx, Hy), CHEAP_OK);
    ASSERT_EQ(cheap_init_from_toeplitz_2d(&b, nx, ny, tx, ty), CHEAP_OK);
    ASSERT_NEAR(b.current_Hx, -1.0, 1e-15);
    ASSERT_NEAR(b.current_Hy, -1.0, 1e-15);

    double max_rel = 0.0;
    for (int i = 0; i < nx * ny; ++i) {
        double got = b.lambda[i], expected = a.lambda[i];
        double rel = fabs(got - expected) / fmax(fabs(expected), 1.0);
        if (rel > max_rel) max_rel = rel;
    }
    /* 1e-10 relative is the contract written into the plan — the DCT
     * round-trip accumulates O(N*eps) error, so this is comfortable. */
    ASSERT_TRUE(max_rel < 1e-10);

    cheap_destroy_2d(&a);
    cheap_destroy_2d(&b);
    fftw_free(lx); fftw_free(ly); fftw_free(tx); fftw_free(ty);
}

/* ---------- Phase 1.3: forward / inverse / apply 2D ---------- */

static double max_abs_diff(const double *a, const double *b, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

typedef struct { int nx, ny; } size_pair;
static const size_pair k_sizes_2d[] = {
    {7, 5}, {16, 16}, {17, 64}, {63, 65}, {128, 256}
};
static const int k_num_sizes_2d = (int)(sizeof(k_sizes_2d) / sizeof(k_sizes_2d[0]));

/* Fill an nx*ny buffer with reproducible "signal" values. */
static void fill_signal(double *x, int nx, int ny, unsigned seed)
{
    for (int j = 0; j < nx; ++j) {
        for (int k = 0; k < ny; ++k) {
            double s = sin(0.01 * (seed + j * 17 + k * 31));
            double c = cos(0.013 * (seed * 3 + j * 7 - k));
            x[j * ny + k] = s + 0.5 * c;
        }
    }
}

static void test_apply_2d_identity(void)
{
    for (int s = 0; s < k_num_sizes_2d; ++s) {
        const int nx = k_sizes_2d[s].nx;
        const int ny = k_sizes_2d[s].ny;
        const int n  = nx * ny;

        cheap_ctx_2d ctx = {0};
        ASSERT_EQ(cheap_init_2d(&ctx, nx, ny, 0.5, 0.5), CHEAP_OK);

        double *x  = (double *)malloc((size_t)n * sizeof(double));
        double *w  = (double *)malloc((size_t)n * sizeof(double));
        double *y  = (double *)malloc((size_t)n * sizeof(double));
        fill_signal(x, nx, ny, 7u);
        for (int i = 0; i < n; ++i) w[i] = 1.0;

        ASSERT_EQ(cheap_apply_2d(&ctx, x, w, y), CHEAP_OK);
        ASSERT_TRUE(max_abs_diff(x, y, n) < 1e-12);

        cheap_destroy_2d(&ctx);
        free(x); free(w); free(y);
    }
}

static void test_forward_inverse_2d_roundtrip(void)
{
    for (int s = 0; s < k_num_sizes_2d; ++s) {
        const int nx = k_sizes_2d[s].nx;
        const int ny = k_sizes_2d[s].ny;
        const int n  = nx * ny;

        cheap_ctx_2d ctx = {0};
        ASSERT_EQ(cheap_init_2d(&ctx, nx, ny, 0.4, 0.6), CHEAP_OK);

        double *x = (double *)malloc((size_t)n * sizeof(double));
        double *y = (double *)malloc((size_t)n * sizeof(double));
        fill_signal(x, nx, ny, 13u);

        ASSERT_EQ(cheap_forward_2d(&ctx, x), CHEAP_OK);
        ASSERT_EQ(cheap_inverse_2d(&ctx, y), CHEAP_OK);
        ASSERT_TRUE(max_abs_diff(x, y, n) < 1e-12);

        cheap_destroy_2d(&ctx);
        free(x); free(y);
    }
}

static void test_apply_inplace_2d_matches_apply(void)
{
    for (int s = 0; s < k_num_sizes_2d; ++s) {
        const int nx = k_sizes_2d[s].nx;
        const int ny = k_sizes_2d[s].ny;
        const int n  = nx * ny;

        cheap_ctx_2d ctx = {0};
        ASSERT_EQ(cheap_init_2d(&ctx, nx, ny, 0.3, 0.7), CHEAP_OK);

        double *x  = (double *)malloc((size_t)n * sizeof(double));
        double *w  = (double *)malloc((size_t)n * sizeof(double));
        double *y1 = (double *)malloc((size_t)n * sizeof(double));
        double *y2 = (double *)malloc((size_t)n * sizeof(double));
        fill_signal(x, nx, ny, 29u);
        /* Use ctx->lambda as a non-trivial weight vector. */
        for (int i = 0; i < n; ++i) w[i] = 1.0 / (ctx.lambda[i] + 1.0);

        ASSERT_EQ(cheap_apply_2d(&ctx, x, w, y1), CHEAP_OK);

        memcpy(cheap_workspace_2d(&ctx), x, (size_t)n * sizeof(double));
        ASSERT_EQ(cheap_apply_inplace_2d(&ctx, w), CHEAP_OK);
        memcpy(y2, cheap_workspace_2d(&ctx), (size_t)n * sizeof(double));

        ASSERT_TRUE(max_abs_diff(y1, y2, n) < 1e-12);

        cheap_destroy_2d(&ctx);
        free(x); free(w); free(y1); free(y2);
    }
}

static void test_forward_inverse_inplace_2d_equivalence(void)
{
    for (int s = 0; s < k_num_sizes_2d; ++s) {
        const int nx = k_sizes_2d[s].nx;
        const int ny = k_sizes_2d[s].ny;
        const int n  = nx * ny;

        cheap_ctx_2d ctx = {0};
        ASSERT_EQ(cheap_init_2d(&ctx, nx, ny, 0.5, 0.5), CHEAP_OK);

        double *x  = (double *)malloc((size_t)n * sizeof(double));
        double *yf = (double *)malloc((size_t)n * sizeof(double));
        double *yi = (double *)malloc((size_t)n * sizeof(double));
        fill_signal(x, nx, ny, 41u);

        /* Forward equivalence. */
        ASSERT_EQ(cheap_forward_2d(&ctx, x), CHEAP_OK);
        memcpy(yf, cheap_workspace_2d(&ctx), (size_t)n * sizeof(double));

        memcpy(cheap_workspace_2d(&ctx), x, (size_t)n * sizeof(double));
        ASSERT_EQ(cheap_forward_inplace_2d(&ctx), CHEAP_OK);
        memcpy(yi, cheap_workspace_2d(&ctx), (size_t)n * sizeof(double));
        ASSERT_TRUE(max_abs_diff(yf, yi, n) < 1e-12);

        /* Inverse equivalence: from the same spectral state in the
         * workspace, cheap_inverse_2d and cheap_inverse_inplace_2d
         * must produce the same signal. */
        memcpy(cheap_workspace_2d(&ctx), yf, (size_t)n * sizeof(double));
        ASSERT_EQ(cheap_inverse_2d(&ctx, yi), CHEAP_OK);

        memcpy(cheap_workspace_2d(&ctx), yf, (size_t)n * sizeof(double));
        ASSERT_EQ(cheap_inverse_inplace_2d(&ctx), CHEAP_OK);
        ASSERT_TRUE(max_abs_diff(yi, cheap_workspace_2d(&ctx), n) < 1e-12);

        cheap_destroy_2d(&ctx);
        free(x); free(yf); free(yi);
    }
}

static void test_apply_2d_bad_args(void)
{
    const int nx = 16, ny = 16, n = nx * ny;
    cheap_ctx_2d ctx = {0};
    ASSERT_EQ(cheap_init_2d(&ctx, nx, ny, 0.5, 0.5), CHEAP_OK);

    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *w = (double *)malloc((size_t)n * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    fill_signal(x, nx, ny, 3u);
    for (int i = 0; i < n; ++i) w[i] = 1.0;

    /* EUNINIT: NULL ctx / uninitialized ctx */
    ASSERT_EQ(cheap_apply_2d(NULL, x, w, y), CHEAP_EUNINIT);
    cheap_ctx_2d zero = {0};
    ASSERT_EQ(cheap_apply_2d(&zero, x, w, y), CHEAP_EUNINIT);
    ASSERT_EQ(cheap_forward_2d(&zero, x), CHEAP_EUNINIT);
    ASSERT_EQ(cheap_inverse_2d(&zero, y), CHEAP_EUNINIT);
    ASSERT_EQ(cheap_apply_inplace_2d(&zero, w), CHEAP_EUNINIT);
    ASSERT_EQ(cheap_forward_inplace_2d(&zero), CHEAP_EUNINIT);
    ASSERT_EQ(cheap_inverse_inplace_2d(&zero), CHEAP_EUNINIT);

    /* EINVAL: NULL buffers */
    ASSERT_EQ(cheap_apply_2d(&ctx, NULL, w, y), CHEAP_EINVAL);
    ASSERT_EQ(cheap_apply_2d(&ctx, x, NULL, y), CHEAP_EINVAL);
    ASSERT_EQ(cheap_apply_2d(&ctx, x, w, NULL), CHEAP_EINVAL);
    ASSERT_EQ(cheap_apply_inplace_2d(&ctx, NULL), CHEAP_EINVAL);
    ASSERT_EQ(cheap_forward_2d(&ctx, NULL), CHEAP_EINVAL);
    ASSERT_EQ(cheap_inverse_2d(&ctx, NULL), CHEAP_EINVAL);

    /* EDOM: non-finite input to the finiteness-checked variants */
    x[5] = (double)NAN;
    ASSERT_EQ(cheap_apply_2d(&ctx, x, w, y), CHEAP_EDOM);
    ASSERT_EQ(cheap_forward_2d(&ctx, x), CHEAP_EDOM);

    /* workspace accessor round-trips */
    ASSERT_TRUE(cheap_workspace_2d(NULL) == NULL);
    ASSERT_TRUE(cheap_workspace_2d(&zero) == NULL);
    ASSERT_TRUE(cheap_workspace_2d(&ctx) == ctx.workspace);

    cheap_destroy_2d(&ctx);
    free(x); free(w); free(y);
}

/* ---------- Phase 1.4: 2D weight constructors ---------- */

static void test_weights_laplacian_2d(void)
{
    const int nx = 128, ny = 96;
    const int n  = nx * ny;
    double *w = (double *)malloc((size_t)n * sizeof(double));
    ASSERT_EQ(cheap_weights_laplacian_2d(nx, ny, w), CHEAP_OK);

    /* DC is exactly zero. */
    ASSERT_NEAR(w[0], 0.0, 0.0);
    /* All entries finite and non-negative. */
    int all_finite = 1, all_nonneg = 1;
    for (int i = 0; i < n; ++i) {
        if (!isfinite(w[i])) all_finite = 0;
        if (w[i] < 0.0) all_nonneg = 0;
    }
    ASSERT_TRUE(all_finite);
    ASSERT_TRUE(all_nonneg);

    /* Spot-check the axis-sum formula at a few cells. */
    const int probes[][2] = {{0, 0}, {0, 5}, {7, 0}, {13, 21}, {nx-1, ny-1}};
    for (size_t p = 0; p < sizeof(probes)/sizeof(probes[0]); ++p) {
        int j = probes[p][0], k = probes[p][1];
        double sx = sin(M_PI * (double)j / (2.0 * (double)nx));
        double sy = sin(M_PI * (double)k / (2.0 * (double)ny));
        double expected = 4.0 * sx * sx + 4.0 * sy * sy;
        ASSERT_NEAR(w[j * ny + k], expected, 1e-14);
    }

    /* Monotonic along each axis from the origin: w[j,0] nondecreasing in j,
     * and w[0,k] nondecreasing in k. (Row-major layout is not globally
     * sorted; only the axis projections are.) */
    for (int j = 1; j < nx; ++j) {
        ASSERT_TRUE(w[j * ny] >= w[(j - 1) * ny] - 1e-15);
    }
    for (int k = 1; k < ny; ++k) {
        ASSERT_TRUE(w[k] >= w[k - 1] - 1e-15);
    }

    free(w);
}

static void test_weights_fractional_laplacian_2d_identity(void)
{
    const int nx = 64, ny = 64;
    const int n  = nx * ny;
    double *w0 = (double *)malloc((size_t)n * sizeof(double));
    ASSERT_EQ(cheap_weights_fractional_laplacian_2d(nx, ny, 0.0, w0), CHEAP_OK);
    /* alpha = 0 means identity: every entry = 1.0, including the DC floor
     * (CHEAP_EPS_LOG)^0 = 1.0. */
    for (int i = 0; i < n; ++i) ASSERT_NEAR(w0[i], 1.0, 1e-15);
    free(w0);
}

static void test_weights_fractional_laplacian_2d_matches_laplacian(void)
{
    const int nx = 64, ny = 48;
    const int n  = nx * ny;
    double *wL   = (double *)malloc((size_t)n * sizeof(double));
    double *wF   = (double *)malloc((size_t)n * sizeof(double));
    ASSERT_EQ(cheap_weights_laplacian_2d(nx, ny, wL), CHEAP_OK);
    ASSERT_EQ(cheap_weights_fractional_laplacian_2d(nx, ny, 1.0, wF), CHEAP_OK);
    /* alpha=1 must match the Laplacian spectrum at every nonzero cell. DC
     * is the exception — Laplacian is exactly 0, but the fractional path
     * floors to CHEAP_EPS_LOG to stay safe under alpha<0. */
    for (int i = 1; i < n; ++i) ASSERT_NEAR(wF[i], wL[i], 1e-12);
    free(wL); free(wF);
}

static void test_weights_fractional_laplacian_2d_power_identity(void)
{
    /* For any nonzero alpha, (w_frac)^(1/alpha) should reproduce the
     * Laplacian spectrum at cells where the DC floor did not engage. */
    const int nx = 32, ny = 40;
    const int n  = nx * ny;
    const double alphas[] = {0.2, 0.5, 1.3, 2.0, -0.75};
    double *wL = (double *)malloc((size_t)n * sizeof(double));
    double *wF = (double *)malloc((size_t)n * sizeof(double));
    ASSERT_EQ(cheap_weights_laplacian_2d(nx, ny, wL), CHEAP_OK);
    for (size_t ai = 0; ai < sizeof(alphas)/sizeof(alphas[0]); ++ai) {
        double alpha = alphas[ai];
        ASSERT_EQ(cheap_weights_fractional_laplacian_2d(nx, ny, alpha, wF), CHEAP_OK);
        for (int i = 1; i < n; ++i) {
            /* wF = wL^alpha ⇒ wF^(1/alpha) = wL when wL > floor */
            double recovered = pow(wF[i], 1.0 / alpha);
            double rel = fabs(recovered - wL[i]) / fmax(fabs(wL[i]), 1.0);
            ASSERT_TRUE(rel < 1e-10);
        }
    }
    free(wL); free(wF);
}

static void test_weights_2d_bad_args(void)
{
    double w[16];
    ASSERT_EQ(cheap_weights_laplacian_2d(1, 4, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_laplacian_2d(4, 1, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_laplacian_2d(4, 4, NULL), CHEAP_EINVAL);

    ASSERT_EQ(cheap_weights_fractional_laplacian_2d(1, 4, 0.5, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_fractional_laplacian_2d(4, 1, 0.5, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_fractional_laplacian_2d(4, 4, 0.5, NULL), CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_fractional_laplacian_2d(4, 4, (double)NAN, w), CHEAP_EINVAL);
}

static void test_init_from_toeplitz_2d_bad_args(void)
{
    const int nx = 8, ny = 6;
    double *tx = (double *)malloc((size_t)nx * sizeof(double));
    double *ty = (double *)malloc((size_t)ny * sizeof(double));
    for (int i = 0; i < nx; ++i) tx[i] = 1.0 / (1.0 + i);
    for (int i = 0; i < ny; ++i) ty[i] = 1.0 / (1.0 + i);
    cheap_ctx_2d ctx = {0};
    ASSERT_EQ(cheap_init_from_toeplitz_2d(NULL, nx, ny, tx, ty), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_toeplitz_2d(&ctx,  1, ny, tx, ty), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_toeplitz_2d(&ctx, nx,  1, tx, ty), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_toeplitz_2d(&ctx, nx, ny, NULL, ty), CHEAP_EINVAL);
    ASSERT_EQ(cheap_init_from_toeplitz_2d(&ctx, nx, ny, tx, NULL), CHEAP_EINVAL);
    tx[0] = (double)NAN;
    ASSERT_EQ(cheap_init_from_toeplitz_2d(&ctx, nx, ny, tx, ty), CHEAP_EDOM);
    free(tx); free(ty);
}

int main(void)
{
    printf("=== test_cheap_2d ===\n");
    test_init_2d_invalid_args();                  printf("  test_init_2d_invalid_args\n");
    test_init_2d_fields();                        printf("  test_init_2d_fields\n");
    test_init_2d_destroy_idempotent();            printf("  test_init_2d_destroy_idempotent\n");
    test_init_2d_tensor_product();                printf("  test_init_2d_tensor_product\n");
    test_init_from_eigenvalues_2d();              printf("  test_init_from_eigenvalues_2d\n");
    test_init_from_toeplitz_2d_matches_flandrin();printf("  test_init_from_toeplitz_2d_matches_flandrin\n");
    test_init_from_toeplitz_2d_bad_args();        printf("  test_init_from_toeplitz_2d_bad_args\n");
    test_apply_2d_identity();                     printf("  test_apply_2d_identity\n");
    test_forward_inverse_2d_roundtrip();          printf("  test_forward_inverse_2d_roundtrip\n");
    test_apply_inplace_2d_matches_apply();        printf("  test_apply_inplace_2d_matches_apply\n");
    test_forward_inverse_inplace_2d_equivalence();printf("  test_forward_inverse_inplace_2d_equivalence\n");
    test_apply_2d_bad_args();                     printf("  test_apply_2d_bad_args\n");
    test_weights_laplacian_2d();                  printf("  test_weights_laplacian_2d\n");
    test_weights_fractional_laplacian_2d_identity();
                                                  printf("  test_weights_fractional_laplacian_2d_identity\n");
    test_weights_fractional_laplacian_2d_matches_laplacian();
                                                  printf("  test_weights_fractional_laplacian_2d_matches_laplacian\n");
    test_weights_fractional_laplacian_2d_power_identity();
                                                  printf("  test_weights_fractional_laplacian_2d_power_identity\n");
    test_weights_2d_bad_args();                   printf("  test_weights_2d_bad_args\n");

    printf("\n=== %d tests run, %d failed ===\n", g_tests_run, g_tests_failed);
    return g_tests_failed == 0 ? 0 : 1;
}
