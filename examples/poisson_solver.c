/*
 * poisson_solver.c — 1D spectral Poisson solve with Neumann BC.
 *
 * Solves: -d^2(phi)/dx^2 = f using DCT diagonalization.
 * DC mode set to zero (zero-mean gauge).
 *
 * Demonstrates: cheap_forward, workspace manipulation, cheap_inverse.
 *
 * Build:
 *   gcc -std=c99 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include poisson_solver.c -o poisson_solver -lfftw3 -lm
 */

#include "cheap.h"
#include <stdio.h>
#include <math.h>

#define N 256

int main(void)
{
    cheap_ctx ctx;
    if (cheap_init(&ctx, N, 0.5) != CHEAP_OK) {
        fprintf(stderr, "cheap_init failed\n");
        return 1;
    }

    double dx = 1.0 / N;
    double dx2 = dx * dx;

    /* Source term: smooth bump */
    double f[N];
    for (int i = 0; i < N; ++i) {
        double x = (double)i / N;
        f[i] = sin(2.0 * M_PI * x);
    }

    /* Poisson solve: forward transform, divide by Laplacian eigenvalues, inverse */
    cheap_forward(&ctx, f);

    /* DC mode: zero-mean gauge */
    ctx.workspace[0] = 0.0;
    for (int k = 1; k < N; ++k) {
        double s = sin(M_PI * (double)k / (2.0 * (double)N));
        double lambda_k = 4.0 * s * s;
        if (lambda_k < CHEAP_EPS_DIV) lambda_k = CHEAP_EPS_DIV;
        ctx.workspace[k] *= dx2 / lambda_k;
    }

    double phi[N];
    cheap_inverse(&ctx, phi);

    printf("Poisson solve: -phi'' = f\n");
    printf("  phi[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n",
           phi[0], phi[1], phi[2], phi[3], phi[4]);

    /* Verify: discrete Laplacian of phi should approximate f (up to DC) */
    double max_err = 0.0;
    double f_mean = 0.0;
    for (int i = 0; i < N; ++i) f_mean += f[i];
    f_mean /= N;

    for (int i = 1; i < N - 1; ++i) {
        double lap = -(phi[i-1] - 2.0 * phi[i] + phi[i+1]) / dx2;
        double err = fabs(lap - (f[i] - f_mean));
        if (err > max_err) max_err = err;
    }
    printf("  Max Laplacian residual (interior): %.3e\n", max_err);

    /* Roundtrip test: Laplacian of phi -> forward -> multiply by eigenvalues -> inverse
       should recover f (minus DC) */
    cheap_forward(&ctx, phi);
    ctx.workspace[0] = 0.0;
    for (int k = 1; k < N; ++k) {
        double s = sin(M_PI * (double)k / (2.0 * (double)N));
        double lambda_k = 4.0 * s * s;
        ctx.workspace[k] *= lambda_k / dx2;
    }
    double f_rec[N];
    cheap_inverse(&ctx, f_rec);

    double max_rt_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = fabs(f_rec[i] - (f[i] - f_mean));
        if (err > max_rt_err) max_rt_err = err;
    }
    printf("  Roundtrip error: %.3e\n", max_rt_err);

    cheap_destroy(&ctx);
    printf("\nPoisson solver example completed.\n");
    return 0;
}
