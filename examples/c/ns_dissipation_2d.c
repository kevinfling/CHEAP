/*
 * Heat / fractional dissipation 2D — 128×128 grid, 100 steps.
 * u_{t+1} = iDCT( DCT(u) * exp(-nu * lambda_laplacian^alpha * dt) ) / norm
 */

#include "cheap.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    const int nx = 128, ny = 128, n = nx * ny;
    const double nu = 0.1, alpha = 1.0, dt = 0.01;
    const int nsteps = 100;

    cheap_ctx_2d ctx;
    if (cheap_init_2d(&ctx, nx, ny, 0.5, 0.5) != CHEAP_OK) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    double *u = (double *)malloc((size_t)n * sizeof(double));
    double *w = (double *)malloc((size_t)n * sizeof(double));

    /* Initial condition: Gaussian bump */
    for (int j = 0; j < nx; ++j) {
        for (int k = 0; k < ny; ++k) {
            double x = (double)j / nx - 0.5;
            double y = (double)k / ny - 0.5;
            u[j * ny + k] = exp(-50.0 * (x * x + y * y));
        }
    }

    /* Precompute fractional Laplacian weights scaled by -nu*dt */
    cheap_weights_fractional_laplacian_2d(nx, ny, alpha, w);
    for (int i = 0; i < n; ++i) w[i] = exp(-nu * dt * w[i]);

    double ke0 = 0.0;
    for (int i = 0; i < n; ++i) ke0 += u[i] * u[i];
    ke0 /= (double)n;

    for (int step = 0; step < nsteps; ++step) {
        cheap_apply_2d(&ctx, u, w, u);
    }

    double ke_final = 0.0;
    for (int i = 0; i < n; ++i) ke_final += u[i] * u[i];
    ke_final /= (double)n;

    /* Kinetic energy should decay monotonically */
    printf("ns_dissipation_2d: KE0 = %.6e, KE_final = %.6e, ratio = %.6f\n",
           ke0, ke_final, ke_final / ke0);

    cheap_destroy_2d(&ctx);
    free(u); free(w);
    return (ke_final < ke0) ? 0 : 1;
}
