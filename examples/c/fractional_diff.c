/*
 * fractional_diff.c — Fractional differentiation and integration via DCT.
 *
 * Demonstrates: cheap_forward, workspace manipulation, cheap_inverse,
 *               and the cheap_apply shorthand.
 *
 * Build:
 *   gcc -std=c99 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include fractional_diff.c -o fractional_diff -lfftw3 -lm
 */

#include "cheap.h"
#include <stdio.h>
#include <math.h>

#define N 128

/*
 * Fractional differentiation weight: w[k] = (2*sin(omega_k/2))^d
 * where omega_k = pi*k/N.
 */
static void compute_frac_weights(double* w, int n, double d_order)
{
    for (int k = 0; k < n; ++k) {
        double omega = M_PI * (double)k / (double)n;
        double sin_half = sin(0.5 * omega);
        if (sin_half < CHEAP_EPS_LOG) sin_half = CHEAP_EPS_LOG;
        w[k] = pow(2.0 * sin_half, d_order);
    }
}

int main(void)
{
    cheap_ctx ctx;
    if (cheap_init(&ctx, N, 0.5) != CHEAP_OK) {
        fprintf(stderr, "cheap_init failed\n");
        return 1;
    }

    /* Test signal: smooth ramp */
    double z[N];
    for (int i = 0; i < N; ++i)
        z[i] = (double)i / N;

    /* --- Fractional differentiation (d=0.4) --- */
    double w_diff[N], z_diff[N];
    compute_frac_weights(w_diff, N, 0.4);
    cheap_apply(&ctx, z, w_diff, z_diff);

    printf("Fractional diff (d=0.4):\n");
    printf("  z_diff[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n",
           z_diff[0], z_diff[1], z_diff[2], z_diff[3], z_diff[4]);

    /* --- Fractional integration (d=-0.4) to recover original --- */
    double w_int[N], z_back[N];
    compute_frac_weights(w_int, N, -0.4);
    cheap_apply(&ctx, z_diff, w_int, z_back);

    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = fabs(z_back[i] - z[i]);
        if (err > max_err) max_err = err;
    }
    printf("\nRoundtrip error: max |z_back - z| = %.3e\n", max_err);

    /* --- Identity test: d=0 should be passthrough --- */
    double w_id[N], z_id[N];
    compute_frac_weights(w_id, N, 0.0);
    cheap_apply(&ctx, z, w_id, z_id);

    double max_id_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = fabs(z_id[i] - z[i]);
        if (err > max_id_err) max_id_err = err;
    }
    printf("Identity (d=0) error: max |z_id - z| = %.3e\n", max_id_err);

    /* --- Using forward/inverse directly for custom manipulation --- */
    printf("\nDirect forward/inverse demo:\n");
    cheap_forward(&ctx, z);
    printf("  Spectral coeffs [0..4] = %.4f, %.4f, %.4f, %.4f, %.4f\n",
           ctx.workspace[0], ctx.workspace[1], ctx.workspace[2],
           ctx.workspace[3], ctx.workspace[4]);

    /* Apply fractional weights manually */
    for (int k = 0; k < N; ++k)
        ctx.workspace[k] *= w_diff[k];

    double z_diff2[N];
    cheap_inverse(&ctx, z_diff2);

    /* Should match cheap_apply result */
    double max_diff = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = fabs(z_diff2[i] - z_diff[i]);
        if (err > max_diff) max_diff = err;
    }
    printf("  forward/inverse vs apply diff: %.3e\n", max_diff);

    cheap_destroy(&ctx);
    printf("\nFractional diff example completed.\n");
    return 0;
}
