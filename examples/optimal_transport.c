/*
 * optimal_transport.c — Sinkhorn optimal transport via spectral acceleration.
 *
 * Demonstrates: cheap_sinkhorn with fBm cost kernel.
 *
 * Build:
 *   gcc -std=c99 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include optimal_transport.c -o optimal_transport -lfftw3 -lm
 */

#include "cheap.h"
#include <stdio.h>
#include <math.h>

#define N 128

int main(void)
{
    cheap_ctx ctx;
    if (cheap_init(&ctx, N, 0.7) != CHEAP_OK) {
        fprintf(stderr, "cheap_init failed\n");
        return 1;
    }

    /* Two probability distributions: shifted Gaussians */
    double a[N], b[N];
    double sum_a = 0.0, sum_b = 0.0;
    for (int i = 0; i < N; ++i) {
        double x = (double)i / N;
        a[i] = exp(-0.5 * (x - 0.3) * (x - 0.3) / (0.05 * 0.05));
        b[i] = exp(-0.5 * (x - 0.7) * (x - 0.7) / (0.05 * 0.05));
        sum_a += a[i];
        sum_b += b[i];
    }
    /* Normalize to equal mass */
    for (int i = 0; i < N; ++i) {
        a[i] /= sum_a;
        b[i] /= sum_b;
    }

    double f[N], g[N];
    double eps = 0.01;
    int max_iter = 200;
    double tol = 1e-8;

    int ret = cheap_sinkhorn(&ctx, a, b, eps, max_iter, tol, f, g);
    if (ret == CHEAP_OK) {
        printf("Sinkhorn converged.\n");
        printf("  f[0..4] = %.4f, %.4f, %.4f, %.4f, %.4f\n",
               f[0], f[1], f[2], f[3], f[4]);
        printf("  g[0..4] = %.4f, %.4f, %.4f, %.4f, %.4f\n",
               g[0], g[1], g[2], g[3], g[4]);
    } else if (ret == CHEAP_ENOCONV) {
        printf("Sinkhorn did not converge within %d iterations.\n", max_iter);
    } else {
        printf("Sinkhorn error: %d\n", ret);
    }

    cheap_destroy(&ctx);
    printf("\nOptimal transport example completed.\n");
    return 0;
}
