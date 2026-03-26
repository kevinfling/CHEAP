/*
 * lqr_mpc.c — LQR Tikhonov solve, Kalman prediction, and MPC step.
 *
 * Demonstrates: cheap_apply with control-specific weight vectors.
 *
 * Build:
 *   gcc -std=c99 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include lqr_mpc.c -o lqr_mpc -lfftw3 -lm
 */

#include "cheap.h"
#include <stdio.h>
#include <math.h>

#define N 64

int main(void)
{
    cheap_ctx ctx;
    if (cheap_init(&ctx, N, 0.7) != CHEAP_OK) {
        fprintf(stderr, "cheap_init failed\n");
        return 1;
    }

    /* --- LQR Tikhonov: P = iDCT( DCT(Q) / (lambda + R) ) / (2N) --- */
    double Q[N], P[N];
    for (int i = 0; i < N; ++i)
        Q[i] = 1.0 + 0.5 * cos(2.0 * M_PI * (double)i / N);

    double R_val = 0.1;
    double w_lqr[N];
    for (int k = 0; k < N; ++k) {
        double denom = ctx.lambda[k] + R_val;
        if (denom < CHEAP_EPS_DIV) denom = CHEAP_EPS_DIV;
        w_lqr[k] = 1.0 / denom;
    }
    cheap_apply(&ctx, Q, w_lqr, P);

    printf("LQR Tikhonov:\n");
    printf("  P[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n",
           P[0], P[1], P[2], P[3], P[4]);

    /* --- MPC step: u = (K + 0*I)^{-1} Q  (i.e., lambda_reg = 0) --- */
    double u[N];
    double w_mpc[N];
    for (int k = 0; k < N; ++k) {
        double denom = ctx.lambda[k];
        if (denom < CHEAP_EPS_DIV) denom = CHEAP_EPS_DIV;
        w_mpc[k] = 1.0 / denom;
    }
    cheap_apply(&ctx, Q, w_mpc, u);

    printf("\nMPC step:\n");
    printf("  u[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n",
           u[0], u[1], u[2], u[3], u[4]);

    /* --- Kalman predict: P_minus[k] = P_diag[k] + lambda[k] --- */
    double P_diag[N], P_minus[N];
    for (int k = 0; k < N; ++k) P_diag[k] = 1.0;
    for (int k = 0; k < N; ++k)
        P_minus[k] = P_diag[k] + ctx.lambda[k];

    printf("\nKalman predict:\n");
    printf("  P_minus[0..4] = %.4f, %.4f, %.4f, %.4f, %.4f\n",
           P_minus[0], P_minus[1], P_minus[2], P_minus[3], P_minus[4]);

    cheap_destroy(&ctx);
    printf("\nLQR/MPC example completed.\n");
    return 0;
}
