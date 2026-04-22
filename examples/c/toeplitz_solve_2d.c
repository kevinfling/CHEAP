/*
 * Toeplitz solve 2D — solve (T_BTTB + lambda*I) x = y for a separable
 * block-Toeplitz-with-Toeplitz-blocks covariance.
 */

#include "cheap.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    const int nx = 32, ny = 32, n = nx * ny;
    const double lambda_reg = 1e-3;

    /* Build separable Toeplitz first columns: exponential decay */
    double *t_col = (double *)malloc((size_t)nx * sizeof(double));
    double *t_row = (double *)malloc((size_t)ny * sizeof(double));
    for (int i = 0; i < nx; ++i) t_col[i] = exp(-0.1 * i);
    for (int i = 0; i < ny; ++i) t_row[i] = exp(-0.05 * i);

    cheap_ctx_2d ctx;
    if (cheap_init_from_toeplitz_2d(&ctx, nx, ny, t_col, t_row) != CHEAP_OK) {
        fprintf(stderr, "init_from_toeplitz_2d failed\n");
        return 1;
    }

    double *y = (double *)malloc((size_t)n * sizeof(double));
    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *w = (double *)malloc((size_t)n * sizeof(double));

    for (int i = 0; i < n; ++i) y[i] = sin(0.1 * i) + 0.5;

    /* Spectral solve: x = iDCT( DCT(y) / (lambda + reg) ) */
    for (int i = 0; i < n; ++i)
        w[i] = 1.0 / (ctx.lambda[i] + lambda_reg);

    cheap_apply_2d(&ctx, y, w, x);

    /* Verify: compute (T + reg*I)x via spectral matvec and compare to y */
    double *check = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i) w[i] = ctx.lambda[i];
    cheap_apply_2d(&ctx, x, w, check);
    for (int i = 0; i < n; ++i) check[i] += lambda_reg * x[i];

    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double err = fabs(check[i] - y[i]);
        if (err > max_err) max_err = err;
    }

    printf("toeplitz_solve_2d: max residual = %.3e\n", max_err);

    cheap_destroy_2d(&ctx);
    free(t_col); free(t_row);
    free(y); free(x); free(w); free(check);
    return (max_err < 1e-10) ? 0 : 1;
}
