/*
 * Heat / fractional dissipation 3D — 32³ grid, 100 steps.
 */

#include "cheap.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    const int nx = 32, ny = 32, nz = 32, n = nx * ny * nz;
    const double nu = 0.05, alpha = 1.0, dt = 0.01;
    const int nsteps = 100;

    cheap_ctx_3d ctx;
    if (cheap_init_3d(&ctx, nx, ny, nz, 0.5, 0.5, 0.5) != CHEAP_OK) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    double *u = (double *)malloc((size_t)n * sizeof(double));
    double *w = (double *)malloc((size_t)n * sizeof(double));

    /* Initial condition: Gaussian bump */
    for (int j = 0; j < nx; ++j) {
        for (int k = 0; k < ny; ++k) {
            for (int l = 0; l < nz; ++l) {
                double x = (double)j / nx - 0.5;
                double y = (double)k / ny - 0.5;
                double z = (double)l / nz - 0.5;
                u[(j * ny + k) * nz + l] = exp(-50.0 * (x * x + y * y + z * z));
            }
        }
    }

    cheap_weights_fractional_laplacian_3d(nx, ny, nz, alpha, w);
    for (int i = 0; i < n; ++i) w[i] = exp(-nu * dt * w[i]);

    double ke0 = 0.0;
    for (int i = 0; i < n; ++i) ke0 += u[i] * u[i];
    ke0 /= (double)n;

    for (int step = 0; step < nsteps; ++step) {
        cheap_apply_3d(&ctx, u, w, u);
    }

    double ke_final = 0.0;
    for (int i = 0; i < n; ++i) ke_final += u[i] * u[i];
    ke_final /= (double)n;

    printf("ns_dissipation_3d: KE0 = %.6e, KE_final = %.6e, ratio = %.6f\n",
           ke0, ke_final, ke_final / ke0);

    cheap_destroy_3d(&ctx);
    free(u); free(w);
    return (ke_final < ke0) ? 0 : 1;
}
