/*
 * test_cheap_weights.c — correctness tests for v0.3.0 weight constructors.
 *
 * Build (standalone):
 *   gcc -std=c99 -pedantic -Wall -Wextra -Werror -march=native -O1 \
 *       -fsanitize=address,undefined \
 *       tests/test_cheap_weights.c -Iinclude -o test_cheap_weights -lfftw3 -lm
 *
 * Scalar (SIMD-disabled) cross-check:
 *   gcc -std=c99 -O1 -DCHEAP_SIMD_DISABLE \
 *       tests/test_cheap_weights.c -Iinclude -o test_cheap_weights_scalar -lfftw3 -lm
 *
 * Debug-contracts build:
 *   gcc -std=c99 -O0 -g -DCHEAP_DEBUG_CONTRACTS \
 *       tests/test_cheap_weights.c -Iinclude -o test_cheap_weights_dc -lfftw3 -lm
 */

#include "cheap.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Minimal test framework (matches test_cheap.c)
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

static int all_finite(const double *v, int n)
{
    for (int i = 0; i < n; ++i)
        if (!isfinite(v[i])) return 0;
    return 1;
}

/* =========================================================================
 * cheap_weights_laplacian_ev
 * ========================================================================= */
static void test_weights_laplacian_ev(void)
{
    printf("  test_weights_laplacian_ev\n");

    const int N = 64;
    double w[64], ref[64];

    ASSERT_EQ(cheap_weights_laplacian_ev(N, w), CHEAP_OK);
    ASSERT_TRUE(all_finite(w, N));

    /* DC must be exactly zero */
    ASSERT_NEAR(w[0], 0.0, 0.0);

    /* Formula check */
    for (int k = 1; k < 4; ++k) {
        double s = sin(M_PI * (double)k / (2.0 * (double)N));
        ASSERT_NEAR(w[k], 4.0 * s * s, 1e-15);
    }

    /* Must match cheap_weights_laplacian exactly */
    ASSERT_EQ(cheap_weights_laplacian(N, ref), CHEAP_OK);
    for (int k = 0; k < N; ++k)
        ASSERT_NEAR(w[k], ref[k], 0.0);

    /* Monotone non-decreasing */
    for (int k = 1; k < N; ++k)
        ASSERT_TRUE(w[k] >= w[k - 1]);

    /* Error codes */
    ASSERT_EQ(cheap_weights_laplacian_ev(1, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_laplacian_ev(N, NULL), CHEAP_EINVAL);
}

/* =========================================================================
 * cheap_weights_matern_ev
 * ========================================================================= */
static void test_weights_matern_ev(void)
{
    printf("  test_weights_matern_ev\n");

    const int N = 64;
    double mu[64], w[64];
    ASSERT_EQ(cheap_weights_laplacian_ev(N, mu), CHEAP_OK);

    /* kappa=2, nu=1.5: DC = pow(4, -1.5) = 0.125 */
    ASSERT_EQ(cheap_weights_matern_ev(N, mu, 2.0, 1.5, w), CHEAP_OK);
    ASSERT_TRUE(all_finite(w, N));
    ASSERT_NEAR(w[0], pow(4.0, -1.5), 1e-14);

    /* Monotone decreasing: (kappa² + mu[k])^(-nu) decreases as mu increases */
    for (int k = 1; k < N; ++k)
        ASSERT_TRUE(w[k] <= w[k - 1]);

    /* kappa=1, nu=1: w[k] = 1/(1+mu[k]), all in (0,1] */
    ASSERT_EQ(cheap_weights_matern_ev(N, mu, 1.0, 1.0, w), CHEAP_OK);
    for (int k = 0; k < N; ++k) {
        ASSERT_TRUE(w[k] > 0.0);
        ASSERT_TRUE(w[k] <= 1.0);
    }
    ASSERT_NEAR(w[0], 1.0, 1e-15);

    /* Error codes */
    ASSERT_EQ(cheap_weights_matern_ev(1, mu, 1.0, 1.0, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_matern_ev(N, NULL, 1.0, 1.0, w),  CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_matern_ev(N, mu, 0.0, 1.0, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_matern_ev(N, mu, -1.0, 1.0, w),   CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_matern_ev(N, mu, 1.0, 0.0, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_matern_ev(N, mu, 1.0, 1.0, NULL), CHEAP_EINVAL);

    /* EDOM on NaN input */
    double nan_mu[64];
    memcpy(nan_mu, mu, N * sizeof(double));
    nan_mu[3] = 0.0 / 0.0;
    ASSERT_EQ(cheap_weights_matern_ev(N, nan_mu, 1.0, 1.0, w), CHEAP_EDOM);
}

/* =========================================================================
 * cheap_weights_matern_2d
 * ========================================================================= */
static void test_weights_matern_2d(void)
{
    printf("  test_weights_matern_2d\n");

    const int NX = 16, NY = 16;
    double w[16 * 16];

    ASSERT_EQ(cheap_weights_matern_2d(NX, NY, 1.0, 1.5, w), CHEAP_OK);
    ASSERT_TRUE(all_finite(w, NX * NY));

    /* DC: pow(kappa², -nu) = pow(1, -1.5) = 1.0 */
    ASSERT_NEAR(w[0], 1.0, 1e-14);

    /* All positive */
    for (int i = 0; i < NX * NY; ++i)
        ASSERT_TRUE(w[i] > 0.0);

    /* Isotropy symmetry: w[j*NY+k] == w[k*NX+j] when NX==NY */
    for (int j = 0; j < NX; ++j)
        for (int k = 0; k < NY; ++k)
            ASSERT_NEAR(w[j * NY + k], w[k * NX + j], 1e-14);

    /* Formula spot-check at (1,0): base = kappa² + lx[1] + ly[0] */
    {
        double sx = sin(M_PI * 1.0 / (2.0 * (double)NX));
        double lx = 4.0 * sx * sx;
        double expected = pow(1.0 + lx, -1.5);
        ASSERT_NEAR(w[1 * NY + 0], expected, 1e-14);
    }

    /* Error codes */
    ASSERT_EQ(cheap_weights_matern_2d(1, NY, 1.0, 1.0, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_matern_2d(NX, 1, 1.0, 1.0, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_matern_2d(NX, NY, 0.0, 1.0, w),   CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_matern_2d(NX, NY, 1.0, 0.0, w),   CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_matern_2d(NX, NY, 1.0, 1.0, NULL),CHEAP_EINVAL);
}

/* =========================================================================
 * cheap_weights_matern_3d (smoke test)
 * ========================================================================= */
static void test_weights_matern_3d(void)
{
    printf("  test_weights_matern_3d\n");

    const int NX = 8, NY = 8, NZ = 8;
    double w[8 * 8 * 8];

    ASSERT_EQ(cheap_weights_matern_3d(NX, NY, NZ, 1.0, 1.0, w), CHEAP_OK);
    ASSERT_TRUE(all_finite(w, NX * NY * NZ));

    /* DC: pow(1, -1) = 1.0 */
    ASSERT_NEAR(w[0], 1.0, 1e-14);

    for (int i = 0; i < NX * NY * NZ; ++i)
        ASSERT_TRUE(w[i] > 0.0);

    ASSERT_EQ(cheap_weights_matern_3d(1, NY, NZ, 1.0, 1.0, w),     CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_matern_3d(NX, NY, NZ, 1.0, 0.0, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_matern_3d(NX, NY, NZ, 1.0, 1.0, NULL), CHEAP_EINVAL);
}

/* =========================================================================
 * cheap_weights_anisotropic_matern_2d
 * ========================================================================= */
static void test_weights_anisotropic_matern_2d(void)
{
    printf("  test_weights_anisotropic_matern_2d\n");

    const int NX = 16, NY = 16;
    double w[16 * 16];

    /* Anisotropic: kappa_x != kappa_y */
    ASSERT_EQ(cheap_weights_anisotropic_matern_2d(NX, NY, 2.0, 0.5, 1.5, w),
              CHEAP_OK);
    ASSERT_TRUE(all_finite(w, NX * NY));

    /* DC: pow(CHEAP_EPS_LOG, -1.5) — large but finite and positive */
    ASSERT_TRUE(w[0] > 0.0);
    ASSERT_TRUE(isfinite(w[0]));
    ASSERT_TRUE(w[0] > 1e10);  /* must be large since base = 1e-12 */

    /* Anisotropy breaks symmetry: w[1*NY+0] != w[0*NX+1] for kappa_x != kappa_y */
    {
        double sx = sin(M_PI * 1.0 / (2.0 * (double)NX));
        double sy = sin(M_PI * 1.0 / (2.0 * (double)NY));
        double lx = 4.0 * sx * sx;
        double ly = 4.0 * sy * sy;
        /* w[1*NY+0] uses kappa_x²*lx + kappa_y²*0 = 4*lx */
        double expected_10 = pow(4.0 * lx + CHEAP_EPS_LOG, -1.5);
        /* w[0*NY+1] uses kappa_x²*0 + kappa_y²*ly = 0.25*ly */
        double expected_01 = pow(0.25 * ly + CHEAP_EPS_LOG, -1.5);
        ASSERT_NEAR(w[1 * NY + 0], expected_10, 1e-12);
        ASSERT_NEAR(w[0 * NY + 1], expected_01, 1e-12);
        /* Confirm asymmetry */
        ASSERT_TRUE(fabs(w[1 * NY + 0] - w[0 * NY + 1]) > 1e-6);
    }

    /* Isotropic case (kappa_x==kappa_y): symmetry holds */
    ASSERT_EQ(cheap_weights_anisotropic_matern_2d(NX, NY, 1.0, 1.0, 1.5, w),
              CHEAP_OK);
    for (int j = 0; j < NX; ++j)
        for (int k = 0; k < NY; ++k)
            ASSERT_NEAR(w[j * NY + k], w[k * NX + j], 1e-13);

    /* Error codes */
    ASSERT_EQ(cheap_weights_anisotropic_matern_2d(NX, NY, 0.0, 1.0, 1.0, w),
              CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_anisotropic_matern_2d(NX, NY, 1.0, 0.0, 1.0, w),
              CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_anisotropic_matern_2d(NX, NY, 1.0, 1.0, 0.0, w),
              CHEAP_EINVAL);
}

/* =========================================================================
 * cheap_weights_heat_propagator_ev
 * ========================================================================= */
static void test_weights_heat_propagator_ev(void)
{
    printf("  test_weights_heat_propagator_ev\n");

    const int N = 64;
    double mu[64], w1[64], w2[64], w3[64];
    ASSERT_EQ(cheap_weights_laplacian_ev(N, mu), CHEAP_OK);

    ASSERT_EQ(cheap_weights_heat_propagator_ev(N, mu, 0.1, w1), CHEAP_OK);
    ASSERT_TRUE(all_finite(w1, N));

    /* DC: exp(-t*0) = 1.0 */
    ASSERT_NEAR(w1[0], 1.0, 1e-15);

    /* All values in (0, 1] since mu >= 0, t > 0 */
    for (int k = 0; k < N; ++k) {
        ASSERT_TRUE(w1[k] > 0.0);
        ASSERT_TRUE(w1[k] <= 1.0 + 1e-15);
    }

    /* Monotone decreasing (larger mu → smaller weight) */
    for (int k = 1; k < N; ++k)
        ASSERT_TRUE(w1[k] <= w1[k - 1]);

    /* Semigroup: heat(t1)[k] * heat(t2)[k] == heat(t1+t2)[k] */
    ASSERT_EQ(cheap_weights_heat_propagator_ev(N, mu, 0.2, w2), CHEAP_OK);
    ASSERT_EQ(cheap_weights_heat_propagator_ev(N, mu, 0.3, w3), CHEAP_OK);
    for (int k = 0; k < N; ++k)
        ASSERT_NEAR(w1[k] * w2[k], w3[k], 1e-14);

    /* Error codes */
    ASSERT_EQ(cheap_weights_heat_propagator_ev(1, mu, 0.1, w1),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_heat_propagator_ev(N, NULL, 0.1, w1),  CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_heat_propagator_ev(N, mu, 0.0, w1),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_heat_propagator_ev(N, mu, -0.1, w1),   CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_heat_propagator_ev(N, mu, 0.1, NULL),  CHEAP_EINVAL);

    /* EDOM on NaN input */
    double nan_mu[64];
    memcpy(nan_mu, mu, N * sizeof(double));
    nan_mu[5] = 0.0 / 0.0;
    ASSERT_EQ(cheap_weights_heat_propagator_ev(N, nan_mu, 0.1, w1), CHEAP_EDOM);
}

/* =========================================================================
 * cheap_weights_heat_propagator_2d (smoke test)
 * ========================================================================= */
static void test_weights_heat_propagator_2d(void)
{
    printf("  test_weights_heat_propagator_2d\n");

    const int NX = 16, NY = 16;
    double w[16 * 16];

    ASSERT_EQ(cheap_weights_heat_propagator_2d(NX, NY, 0.05, w), CHEAP_OK);
    ASSERT_TRUE(all_finite(w, NX * NY));

    /* DC: 1.0 */
    ASSERT_NEAR(w[0], 1.0, 1e-15);

    /* All in (0, 1] */
    for (int i = 0; i < NX * NY; ++i) {
        ASSERT_TRUE(w[i] > 0.0);
        ASSERT_TRUE(w[i] <= 1.0 + 1e-15);
    }

    /* Symmetry when NX==NY */
    for (int j = 0; j < NX; ++j)
        for (int k = 0; k < NY; ++k)
            ASSERT_NEAR(w[j * NY + k], w[k * NX + j], 1e-14);

    ASSERT_EQ(cheap_weights_heat_propagator_2d(1, NY, 0.1, w),     CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_heat_propagator_2d(NX, NY, 0.0, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_heat_propagator_2d(NX, NY, 0.1, NULL), CHEAP_EINVAL);
}

/* =========================================================================
 * cheap_weights_biharmonic_ev
 * ========================================================================= */
static void test_weights_biharmonic_ev(void)
{
    printf("  test_weights_biharmonic_ev\n");

    const int N = 64;
    const double EPS = 1e-4;
    double mu[64], w[64];
    ASSERT_EQ(cheap_weights_laplacian_ev(N, mu), CHEAP_OK);

    ASSERT_EQ(cheap_weights_biharmonic_ev(N, mu, EPS, w), CHEAP_OK);
    ASSERT_TRUE(all_finite(w, N));

    /* DC: 1/eps */
    ASSERT_NEAR(w[0], 1.0 / EPS, 1e-10);

    /* All positive */
    for (int k = 0; k < N; ++k)
        ASSERT_TRUE(w[k] > 0.0);

    /* Monotone decreasing for k > 0 (mu² increasing) */
    for (int k = 1; k < N; ++k)
        ASSERT_TRUE(w[k] <= w[k - 1]);

    /* Formula spot-check */
    for (int k = 0; k < N; ++k) {
        double m2 = mu[k] * mu[k];
        ASSERT_NEAR(w[k], 1.0 / (m2 + EPS), 1e-14);
    }

    /* SIMD vs scalar agreement: compute reference with scalar formula */
    {
        double ref[64];
        for (int k = 0; k < N; ++k)
            ref[k] = 1.0 / (mu[k] * mu[k] + EPS);
        for (int k = 0; k < N; ++k)
            ASSERT_NEAR(w[k], ref[k], 1e-14);
    }

    /* Error codes */
    ASSERT_EQ(cheap_weights_biharmonic_ev(1, mu, EPS, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_biharmonic_ev(N, NULL, EPS, w),  CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_biharmonic_ev(N, mu, 0.0, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_biharmonic_ev(N, mu, -1.0, w),   CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_biharmonic_ev(N, mu, EPS, NULL), CHEAP_EINVAL);

    /* EDOM on NaN */
    double nan_mu[64];
    memcpy(nan_mu, mu, N * sizeof(double));
    nan_mu[2] = 0.0 / 0.0;
    ASSERT_EQ(cheap_weights_biharmonic_ev(N, nan_mu, EPS, w), CHEAP_EDOM);
}

/* =========================================================================
 * cheap_weights_biharmonic_2d
 * ========================================================================= */
static void test_weights_biharmonic_2d(void)
{
    printf("  test_weights_biharmonic_2d\n");

    const int NX = 16, NY = 16;
    const double EPS = 1e-4;
    double w[16 * 16];

    ASSERT_EQ(cheap_weights_biharmonic_2d(NX, NY, EPS, w), CHEAP_OK);
    ASSERT_TRUE(all_finite(w, NX * NY));

    /* DC: 1/eps */
    ASSERT_NEAR(w[0], 1.0 / EPS, 1e-10);

    /* All positive */
    for (int i = 0; i < NX * NY; ++i)
        ASSERT_TRUE(w[i] > 0.0);

    /* Symmetry when NX==NY */
    for (int j = 0; j < NX; ++j)
        for (int k = 0; k < NY; ++k)
            ASSERT_NEAR(w[j * NY + k], w[k * NX + j], 1e-14);

    /* Formula spot-check vs 2D Laplacian + biharmonic formula */
    {
        double lap2d[16 * 16];
        cheap_weights_laplacian_2d(NX, NY, lap2d);
        for (int i = 0; i < NX * NY; ++i) {
            double expected = 1.0 / (lap2d[i] * lap2d[i] + EPS);
            ASSERT_NEAR(w[i], expected, 1e-13);
        }
    }

    ASSERT_EQ(cheap_weights_biharmonic_2d(1, NY, EPS, w),     CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_biharmonic_2d(NX, 1, EPS, w),     CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_biharmonic_2d(NX, NY, 0.0, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_biharmonic_2d(NX, NY, EPS, NULL), CHEAP_EINVAL);
}

/* =========================================================================
 * cheap_weights_biharmonic_3d (smoke test)
 * ========================================================================= */
static void test_weights_biharmonic_3d(void)
{
    printf("  test_weights_biharmonic_3d\n");

    const int NX = 8, NY = 8, NZ = 8;
    const double EPS = 1e-4;
    double w[8 * 8 * 8];

    ASSERT_EQ(cheap_weights_biharmonic_3d(NX, NY, NZ, EPS, w), CHEAP_OK);
    ASSERT_TRUE(all_finite(w, NX * NY * NZ));
    ASSERT_NEAR(w[0], 1.0 / EPS, 1e-10);
    for (int i = 0; i < NX * NY * NZ; ++i)
        ASSERT_TRUE(w[i] > 0.0);

    ASSERT_EQ(cheap_weights_biharmonic_3d(1, NY, NZ, EPS, w),     CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_biharmonic_3d(NX, NY, NZ, 0.0, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_biharmonic_3d(NX, NY, NZ, EPS, NULL), CHEAP_EINVAL);
}

/* =========================================================================
 * cheap_weights_poisson_ev
 * ========================================================================= */
static void test_weights_poisson_ev(void)
{
    printf("  test_weights_poisson_ev\n");

    const int N = 64;
    const double EPS = 1e-6;
    double mu[64], w[64];
    ASSERT_EQ(cheap_weights_laplacian_ev(N, mu), CHEAP_OK);

    ASSERT_EQ(cheap_weights_poisson_ev(N, mu, EPS, w), CHEAP_OK);
    ASSERT_TRUE(all_finite(w, N));

    /* DC must be exactly 0.0 */
    ASSERT_NEAR(w[0], 0.0, 0.0);

    /* All k > 0: positive */
    for (int k = 1; k < N; ++k)
        ASSERT_TRUE(w[k] > 0.0);

    /* Monotone decreasing for k > 0 */
    for (int k = 2; k < N; ++k)
        ASSERT_TRUE(w[k] <= w[k - 1]);

    /* Formula spot-check */
    for (int k = 1; k < N; ++k)
        ASSERT_NEAR(w[k], 1.0 / (mu[k] + EPS), 1e-14);

    /* SIMD vs scalar reference */
    {
        double ref[64];
        ref[0] = 0.0;
        for (int k = 1; k < N; ++k)
            ref[k] = 1.0 / (mu[k] + EPS);
        for (int k = 0; k < N; ++k)
            ASSERT_NEAR(w[k], ref[k], 1e-14);
    }

    /* eps=0 permitted (mu[k] > 0 for k > 0 from Laplacian) */
    ASSERT_EQ(cheap_weights_poisson_ev(N, mu, 0.0, w), CHEAP_OK);
    ASSERT_NEAR(w[0], 0.0, 0.0);
    for (int k = 1; k < N; ++k)
        ASSERT_NEAR(w[k], 1.0 / mu[k], 1e-13);

    /* Error codes */
    ASSERT_EQ(cheap_weights_poisson_ev(1, mu, EPS, w),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_poisson_ev(N, NULL, EPS, w),  CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_poisson_ev(N, mu, -1.0, w),   CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_poisson_ev(N, mu, EPS, NULL), CHEAP_EINVAL);

    /* EDOM on NaN */
    double nan_mu[64];
    memcpy(nan_mu, mu, N * sizeof(double));
    nan_mu[10] = 0.0 / 0.0;
    ASSERT_EQ(cheap_weights_poisson_ev(N, nan_mu, EPS, w), CHEAP_EDOM);
}

/* =========================================================================
 * cheap_weights_poisson_2d
 * ========================================================================= */
static void test_weights_poisson_2d(void)
{
    printf("  test_weights_poisson_2d\n");

    const int NX = 16, NY = 16;
    const double EPS = 1e-6;
    double w[16 * 16];

    ASSERT_EQ(cheap_weights_poisson_2d(NX, NY, EPS, w), CHEAP_OK);
    ASSERT_TRUE(all_finite(w, NX * NY));

    /* DC exactly 0.0 */
    ASSERT_NEAR(w[0], 0.0, 0.0);

    /* All others positive */
    for (int i = 1; i < NX * NY; ++i)
        ASSERT_TRUE(w[i] > 0.0);

    /* Symmetry when NX==NY */
    for (int j = 0; j < NX; ++j)
        for (int k = 0; k < NY; ++k)
            ASSERT_NEAR(w[j * NY + k], w[k * NX + j], 1e-14);

    /* Formula spot-check vs 2D Laplacian */
    {
        double lap2d[16 * 16];
        cheap_weights_laplacian_2d(NX, NY, lap2d);
        for (int i = 1; i < NX * NY; ++i)
            ASSERT_NEAR(w[i], 1.0 / (lap2d[i] + EPS), 1e-13);
    }

    ASSERT_EQ(cheap_weights_poisson_2d(1, NY, EPS, w),     CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_poisson_2d(NX, 1, EPS, w),     CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_poisson_2d(NX, NY, -1.0, w),   CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_poisson_2d(NX, NY, EPS, NULL), CHEAP_EINVAL);
}

/* =========================================================================
 * cheap_weights_poisson_3d (smoke test)
 * ========================================================================= */
static void test_weights_poisson_3d(void)
{
    printf("  test_weights_poisson_3d\n");

    const int NX = 8, NY = 8, NZ = 8;
    const double EPS = 1e-6;
    double w[8 * 8 * 8];

    ASSERT_EQ(cheap_weights_poisson_3d(NX, NY, NZ, EPS, w), CHEAP_OK);
    ASSERT_TRUE(all_finite(w, NX * NY * NZ));
    ASSERT_NEAR(w[0], 0.0, 0.0);
    for (int i = 1; i < NX * NY * NZ; ++i)
        ASSERT_TRUE(w[i] > 0.0);

    ASSERT_EQ(cheap_weights_poisson_3d(1, NY, NZ, EPS, w),     CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_poisson_3d(NX, NY, NZ, -1.0, w),   CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_poisson_3d(NX, NY, NZ, EPS, NULL), CHEAP_EINVAL);
}

/* =========================================================================
 * cheap_weights_higher_order_tikhonov_deconv_ev
 * ========================================================================= */
static void test_weights_higher_order_tikhonov_deconv_ev(void)
{
    printf("  test_weights_higher_order_tikhonov_deconv_ev\n");

    const int N = 64;
    const double ALPHA = 0.01, P = 2.0, EPS = 1e-8;
    double mu[64], psf[64], w_null[64], w_explicit[64];

    ASSERT_EQ(cheap_weights_laplacian_ev(N, mu), CHEAP_OK);

    /* Flat PSF (all-ones): uniform frequency response */
    for (int k = 0; k < N; ++k) psf[k] = 1.0;

    /* NULL lap path (computes 1D Laplacian internally) */
    ASSERT_EQ(cheap_weights_higher_order_tikhonov_deconv_ev(
                  N, psf, NULL, ALPHA, P, EPS, w_null),
              CHEAP_OK);
    ASSERT_TRUE(all_finite(w_null, N));

    /* Explicit lap path: must give identical result */
    ASSERT_EQ(cheap_weights_higher_order_tikhonov_deconv_ev(
                  N, psf, mu, ALPHA, P, EPS, w_explicit),
              CHEAP_OK);
    for (int k = 0; k < N; ++k)
        ASSERT_NEAR(w_null[k], w_explicit[k], 1e-15);

    /* DC: psf[0]/(psf[0]² + 0 + eps) = 1/(1+eps) */
    ASSERT_NEAR(w_null[0], 1.0 / (1.0 + EPS), 1e-14);

    /* Formula spot-check at k=3 */
    {
        double psi = 1.0, lv = mu[3];
        double pen = (lv > CHEAP_EPS_LOG) ? ALPHA * pow(lv, P) : 0.0;
        double den = psi * psi + pen + EPS;
        ASSERT_NEAR(w_null[3], psi / den, 1e-14);
    }

    /* Larger p penalizes high frequencies more: compare p=1 vs p=2 */
    {
        double w_p1[64], w_p2[64];
        ASSERT_EQ(cheap_weights_higher_order_tikhonov_deconv_ev(
                      N, psf, mu, ALPHA, 1.0, EPS, w_p1), CHEAP_OK);
        ASSERT_EQ(cheap_weights_higher_order_tikhonov_deconv_ev(
                      N, psf, mu, ALPHA, 2.0, EPS, w_p2), CHEAP_OK);
        /* For mu[k] > 1, p=2 produces more suppression than p=1 */
        for (int k = 1; k < N; ++k) {
            if (mu[k] > 1.0)
                ASSERT_TRUE(w_p2[k] <= w_p1[k]);
        }
    }

    /* Error codes */
    ASSERT_EQ(cheap_weights_higher_order_tikhonov_deconv_ev(
                  1, psf, NULL, ALPHA, P, EPS, w_null),     CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_higher_order_tikhonov_deconv_ev(
                  N, NULL, NULL, ALPHA, P, EPS, w_null),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_higher_order_tikhonov_deconv_ev(
                  N, psf, NULL, -0.1, P, EPS, w_null),      CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_higher_order_tikhonov_deconv_ev(
                  N, psf, NULL, ALPHA, 0.0, EPS, w_null),   CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_higher_order_tikhonov_deconv_ev(
                  N, psf, NULL, ALPHA, P, -1.0, w_null),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_weights_higher_order_tikhonov_deconv_ev(
                  N, psf, NULL, ALPHA, P, EPS, NULL),        CHEAP_EINVAL);

    /* EDOM on NaN in psf */
    double nan_psf[64];
    memcpy(nan_psf, psf, N * sizeof(double));
    nan_psf[7] = 0.0 / 0.0;
    ASSERT_EQ(cheap_weights_higher_order_tikhonov_deconv_ev(
                  N, nan_psf, NULL, ALPHA, P, EPS, w_null), CHEAP_EDOM);
}

/* =========================================================================
 * cheap_sample_matern_2d / cheap_sample_matern_3d
 * ========================================================================= */
static void test_sample_matern_2d(void)
{
    printf("  test_sample_matern_2d\n");

    const int NX = 16, NY = 16, N = NX * NY;
    cheap_ctx_2d ctx;
    ASSERT_EQ(cheap_init_2d(&ctx, NX, NY, 0.7, 0.7), CHEAP_OK);

    /* Simple non-constant input: alternating ±1 */
    double noise[16 * 16], grf[16 * 16];
    for (int i = 0; i < N; ++i) noise[i] = (i % 2 == 0) ? 1.0 : -1.0;

    ASSERT_EQ(cheap_sample_matern_2d(&ctx, noise, 1.0, 1.5, grf), CHEAP_OK);
    ASSERT_TRUE(all_finite(grf, N));

    /* Result must differ from input (the filter is non-trivial) */
    int differs = 0;
    for (int i = 0; i < N; ++i)
        if (fabs(grf[i] - noise[i]) > 1e-10) { differs = 1; break; }
    ASSERT_TRUE(differs);

    /* Cross-check: cheap_sample_matern_2d == cheap_apply_2d with Matérn weights */
    {
        double w[16 * 16], ref[16 * 16];
        cheap_weights_matern_2d(NX, NY, 1.0, 1.5, w);
        cheap_apply_2d(&ctx, noise, w, ref);
        for (int i = 0; i < N; ++i)
            ASSERT_NEAR(grf[i], ref[i], 1e-13);
    }

    /* Error codes */
    ASSERT_EQ(cheap_sample_matern_2d(NULL, noise, 1.0, 1.5, grf),    CHEAP_EUNINIT);
    ASSERT_EQ(cheap_sample_matern_2d(&ctx, NULL, 1.0, 1.5, grf),     CHEAP_EINVAL);
    ASSERT_EQ(cheap_sample_matern_2d(&ctx, noise, 0.0, 1.5, grf),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_sample_matern_2d(&ctx, noise, 1.0, 0.0, grf),    CHEAP_EINVAL);
    ASSERT_EQ(cheap_sample_matern_2d(&ctx, noise, 1.0, 1.5, NULL),   CHEAP_EINVAL);

    cheap_destroy_2d(&ctx);
}

static void test_sample_matern_3d(void)
{
    printf("  test_sample_matern_3d\n");

    const int NX = 8, NY = 8, NZ = 8, N = NX * NY * NZ;
    cheap_ctx_3d ctx;
    ASSERT_EQ(cheap_init_3d(&ctx, NX, NY, NZ, 0.7, 0.7, 0.7), CHEAP_OK);

    double noise[8 * 8 * 8], grf[8 * 8 * 8];
    for (int i = 0; i < N; ++i) noise[i] = (i % 2 == 0) ? 1.0 : -1.0;

    ASSERT_EQ(cheap_sample_matern_3d(&ctx, noise, 1.0, 2.5, grf), CHEAP_OK);
    ASSERT_TRUE(all_finite(grf, N));

    /* Cross-check vs cheap_apply_3d */
    {
        double w[8 * 8 * 8], ref[8 * 8 * 8];
        cheap_weights_matern_3d(NX, NY, NZ, 1.0, 2.5, w);
        cheap_apply_3d(&ctx, noise, w, ref);
        for (int i = 0; i < N; ++i)
            ASSERT_NEAR(grf[i], ref[i], 1e-13);
    }

    ASSERT_EQ(cheap_sample_matern_3d(NULL, noise, 1.0, 2.5, grf),  CHEAP_EUNINIT);
    ASSERT_EQ(cheap_sample_matern_3d(&ctx, NULL, 1.0, 2.5, grf),   CHEAP_EINVAL);
    ASSERT_EQ(cheap_sample_matern_3d(&ctx, noise, 0.0, 2.5, grf),  CHEAP_EINVAL);

    cheap_destroy_3d(&ctx);
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(void)
{
    printf("=== test_cheap_weights (v0.3.0-tensor-weights) ===\n\n");

    test_weights_laplacian_ev();
    test_weights_matern_ev();
    test_weights_matern_2d();
    test_weights_matern_3d();
    test_weights_anisotropic_matern_2d();
    test_weights_heat_propagator_ev();
    test_weights_heat_propagator_2d();
    test_weights_biharmonic_ev();
    test_weights_biharmonic_2d();
    test_weights_biharmonic_3d();
    test_weights_poisson_ev();
    test_weights_poisson_2d();
    test_weights_poisson_3d();
    test_weights_higher_order_tikhonov_deconv_ev();
    test_sample_matern_2d();
    test_sample_matern_3d();

    printf("\n%d tests run, %d failed\n", g_tests_run, g_tests_failed);
    return (g_tests_failed == 0) ? 0 : 1;
}
