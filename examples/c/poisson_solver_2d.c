/*
 * Poisson solver 2D — Neumann BC on a 128×128 grid.
 * Solve -Δφ = f via DCT diagonalization.
 */

#include "cheap.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    const int nx = 128, ny = 128, n = nx * ny;

    cheap_ctx_2d ctx;
    if (cheap_init_2d(&ctx, nx, ny, 0.5, 0.5) != CHEAP_OK) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    double *f = (double *)malloc((size_t)n * sizeof(double));
    double *phi = (double *)malloc((size_t)n * sizeof(double));
    double *w = (double *)malloc((size_t)n * sizeof(double));

    /* Gaussian source centered in the domain */
    for (int j = 0; j < nx; ++j) {
        for (int k = 0; k < ny; ++k) {
            double x = (double)j / (double)(nx - 1) - 0.5;
            double y = (double)k / (double)(ny - 1) - 0.5;
            f[j * ny + k] = exp(-100.0 * (x * x + y * y));
        }
    }

    /* Enforce zero-mean compatibility for Neumann Poisson */
    double f_mean = 0.0;
    for (int i = 0; i < n; ++i) f_mean += f[i];
    f_mean /= (double)n;
    for (int i = 0; i < n; ++i) f[i] -= f_mean;

    /* Laplacian eigenvalues with DC regularization */
    cheap_weights_laplacian_2d(nx, ny, w);
    w[0] = 1.0; /* regularize DC to avoid division by zero */

    /* Spectral solve: φ = iDCT( DCT(f) / w ) / (4*nx*ny) */
    cheap_forward_2d(&ctx, f);
    for (int i = 0; i < n; ++i) ctx.workspace[i] /= w[i];
    cheap_inverse_2d(&ctx, phi);

    /* Compute residual: max | -Δφ - f | */
    double max_res = 0.0;
    for (int j = 1; j < nx - 1; ++j) {
        for (int k = 1; k < ny - 1; ++k) {
            int idx = j * ny + k;
            double lap = 4.0 * phi[idx]
                       - phi[(j - 1) * ny + k] - phi[(j + 1) * ny + k]
                       - phi[j * ny + (k - 1)] - phi[j * ny + (k + 1)];
            double res = fabs(lap - f[idx]);
            if (res > max_res) max_res = res;
        }
    }

    printf("poisson_solver_2d: max residual = %.3e\n", max_res);

    cheap_destroy_2d(&ctx);
    free(f); free(phi); free(w);
    return (max_res < 1e-10) ? 0 : 1;
}
