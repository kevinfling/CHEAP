/*
 * toeplitz_solve.c — Generic Toeplitz matvec and solve via DCT diagonalization.
 *
 * Demonstrates: cheap_toeplitz_eigenvalues, cheap_apply, cheap_toeplitz_solve_precomp.
 *
 * Build:
 *   gcc -std=c99 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include toeplitz_solve.c -o toeplitz_solve -lfftw3 -lm
 */

#include "cheap.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define N 64

int main(void)
{
    cheap_ctx ctx;
    /* H value is arbitrary here — we only use the FFTW plans, not ctx.lambda */
    if (cheap_init(&ctx, N, 0.5) != CHEAP_OK) {
        fprintf(stderr, "cheap_init failed\n");
        return 1;
    }

    /* Tridiagonal Toeplitz: t = [2, -1, 0, 0, ...] */
    double t[N];
    t[0] = 2.0;
    t[1] = -1.0;
    for (int i = 2; i < N; ++i) t[i] = 0.0;

    /* Compute eigenvalues once */
    double lambda_t[N];
    cheap_toeplitz_eigenvalues(&ctx, t, lambda_t);
    printf("Toeplitz eigenvalues [0..4]: %.4f, %.4f, %.4f, %.4f, %.4f\n",
           lambda_t[0], lambda_t[1], lambda_t[2], lambda_t[3], lambda_t[4]);

    /* Matvec: y = T * x using cheap_apply with eigenvalues as weights */
    double x[N], y[N];
    for (int i = 0; i < N; ++i) x[i] = (double)(i + 1);
    cheap_apply(&ctx, x, lambda_t, y);

    /* Verify against brute-force */
    double y_ref[N];
    for (int i = 0; i < N; ++i) {
        y_ref[i] = 0.0;
        for (int j = 0; j < N; ++j)
            y_ref[i] += t[abs(i - j)] * x[j];
    }
    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = fabs(y[i] - y_ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("Matvec error vs brute-force: %.3e\n", max_err);

    /* Solve: x_sol = (T + lambda*I)^{-1} y */
    double lambda_reg = 1e-3;
    double x_sol[N];
    cheap_toeplitz_solve_precomp(&ctx, lambda_t, y, lambda_reg, x_sol);

    /* Verify residual: ||T*x_sol + lambda*x_sol - y|| */
    double Tx_sol[N];
    cheap_apply(&ctx, x_sol, lambda_t, Tx_sol);
    double max_res = 0.0;
    for (int i = 0; i < N; ++i) {
        double res = fabs(Tx_sol[i] + lambda_reg * x_sol[i] - y[i]);
        if (res > max_res) max_res = res;
    }
    printf("Solve residual: ||T*x + lambda*x - y||_inf = %.3e\n", max_res);

    cheap_destroy(&ctx);
    printf("\nToeplitz solve example completed.\n");
    return 0;
}
