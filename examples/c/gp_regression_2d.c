/*
 * GP regression 2D — 64×64 grid of synthetic observations.
 * Posterior mean via spectral KRR on the 2D Flandrin grid.
 */

#include "cheap.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    const int nx = 64, ny = 64, n = nx * ny;
    const double noise_var = 0.01;

    cheap_ctx_2d ctx;
    if (cheap_init_2d(&ctx, nx, ny, 0.5, 0.5) != CHEAP_OK) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    double *f_clean = (double *)malloc((size_t)n * sizeof(double));
    double *y_obs   = (double *)malloc((size_t)n * sizeof(double));
    double *f_post  = (double *)malloc((size_t)n * sizeof(double));
    double *w       = (double *)malloc((size_t)n * sizeof(double));

    /* Clean function: smooth analytic surface */
    for (int j = 0; j < nx; ++j) {
        for (int k = 0; k < ny; ++k) {
            double x = (double)j / nx;
            double y = (double)k / ny;
            f_clean[j * ny + k] = sin(2.0 * M_PI * x) * cos(2.0 * M_PI * y)
                                + 0.5 * sin(4.0 * M_PI * x);
        }
    }

    /* Add noise */
    srand(42);
    for (int i = 0; i < n; ++i) {
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        if (u1 < 1e-300) u1 = 1e-300;
        double r = sqrt(-2.0 * log(u1));
        double theta = 2.0 * M_PI * u2;
        double noise = sqrt(noise_var) * r * cos(theta);
        y_obs[i] = f_clean[i] + noise;
    }

    /* Posterior mean weights: K / (K + sigma^2*I) with inverse-Laplacian K */
    cheap_weights_laplacian_2d(nx, ny, w);
    for (int i = 0; i < n; ++i) {
        double inv_lam = 1.0 / fmax(w[i], 1e-12);
        w[i] = inv_lam / (inv_lam + noise_var);
    }

    cheap_apply_2d(&ctx, y_obs, w, f_post);

    /* RMSE to clean f */
    double rmse = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = f_post[i] - f_clean[i];
        rmse += d * d;
    }
    rmse = sqrt(rmse / (double)n);

    printf("gp_regression_2d: RMSE = %.3e  (noise sigma = %.3e)\n",
           rmse, sqrt(noise_var));

    cheap_destroy_2d(&ctx);
    free(f_clean); free(y_obs); free(f_post); free(w);
    return (rmse < sqrt(noise_var)) ? 0 : 1;
}
