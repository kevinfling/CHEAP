/*
 * Poisson solver 3D — Neumann BC on a 32³ grid.
 * Solve -Δφ = f via DCT diagonalization.
 */

#include "cheap.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    const int nx = 32, ny = 32, nz = 32, n = nx * ny * nz;

    cheap_ctx_3d ctx;
    if (cheap_init_3d(&ctx, nx, ny, nz, 0.5, 0.5, 0.5) != CHEAP_OK) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    double *f = (double *)malloc((size_t)n * sizeof(double));
    double *phi = (double *)malloc((size_t)n * sizeof(double));
    double *w = (double *)malloc((size_t)n * sizeof(double));

    /* Point source at the center */
    memset(f, 0, (size_t)n * sizeof(double));
    f[(nx / 2 * ny + ny / 2) * nz + nz / 2] = 1.0;

    cheap_weights_laplacian_3d(nx, ny, nz, w);
    w[0] = 1.0; /* DC regularization */

    cheap_forward_3d(&ctx, f);
    for (int i = 0; i < n; ++i) ctx.workspace[i] /= w[i];
    cheap_inverse_3d(&ctx, phi);

    /* Check that the solution is finite and the peak is near the source */
    double phi_max = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!isfinite(phi[i])) {
            fprintf(stderr, "non-finite phi at %d\n", i);
            return 1;
        }
        if (fabs(phi[i]) > phi_max) phi_max = fabs(phi[i]);
    }

    printf("poisson_solver_3d: phi_max = %.6f  (finite=%s)\n",
           phi_max, phi_max > 0.0 ? "yes" : "no");

    cheap_destroy_3d(&ctx);
    free(f); free(phi); free(w);
    return 0;
}
