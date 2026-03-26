/*
 * ns_dissipation.c — Fractional Navier-Stokes dissipation step.
 *
 * Applies: u_new[k] = iDCT( DCT(u)[k] * exp(-nu * lambda_k^alpha * dt) )
 * where lambda_k = (2*sin(pi*k/(2*N)))^2 are discrete Laplacian eigenvalues.
 *
 * Demonstrates: cheap_forward, workspace manipulation, cheap_inverse.
 *
 * Build:
 *   gcc -std=c99 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include ns_dissipation.c -o ns_dissipation -lfftw3 -lm
 */

#include "cheap.h"
#include <stdio.h>
#include <math.h>

#define N 128

int main(void)
{
    cheap_ctx ctx;
    if (cheap_init(&ctx, N, 0.5) != CHEAP_OK) {
        fprintf(stderr, "cheap_init failed\n");
        return 1;
    }

    double nu = 0.01;
    double alpha = 1.0;   /* standard diffusion; try 0.5 for fractional */
    double dt = 0.001;
    int nsteps = 100;

    /* Initial condition: single Fourier mode */
    double u[N];
    for (int i = 0; i < N; ++i)
        u[i] = cos(2.0 * M_PI * (double)i / N);

    printf("NS dissipation: nu=%.3f, alpha=%.1f, dt=%.4f, steps=%d\n",
           nu, alpha, dt, nsteps);
    printf("  Initial: u[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n",
           u[0], u[1], u[2], u[3], u[4]);

    /* Precompute dissipation weights for one timestep */
    double w_diss[N];
    for (int k = 0; k < N; ++k) {
        double s = sin(M_PI * (double)k / (2.0 * (double)N));
        double lambda_k = 4.0 * s * s;
        double exponent = -nu * pow(lambda_k, alpha) * dt;
        w_diss[k] = (exponent < -700.0) ? 0.0 : exp(exponent);
    }

    /* Time-stepping */
    double u_new[N];
    for (int step = 0; step < nsteps; ++step) {
        cheap_apply(&ctx, u, w_diss, u_new);
        memcpy(u, u_new, (size_t)N * sizeof(double));
    }

    printf("  After %d steps: u[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n",
           nsteps, u[0], u[1], u[2], u[3], u[4]);

    /* Check energy decay */
    double energy = 0.0;
    for (int i = 0; i < N; ++i) energy += u[i] * u[i];
    energy /= N;
    printf("  Energy (mean u^2): %.6e\n", energy);
    printf("  Expected decay factor per step for mode k=1: exp(%.4f)\n",
           -nu * pow(4.0 * sin(M_PI / (2.0 * N)) * sin(M_PI / (2.0 * N)), alpha) * dt);

    cheap_destroy(&ctx);
    printf("\nNS dissipation example completed.\n");
    return 0;
}
