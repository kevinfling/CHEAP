/*
 * gp_regression.c — Gaussian process regression via spectral KRR + reparameterization.
 *
 * Demonstrates: cheap_apply with KRR weights and sqrt-lambda weights.
 *
 * Build:
 *   gcc -std=c99 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include gp_regression.c -o gp_regression -lfftw3 -lm
 */

#include "cheap.h"
#include <stdio.h>
#include <math.h>

#define N 128

int main(void)
{
    cheap_ctx ctx;
    if (cheap_init(&ctx, N, 0.7) != CHEAP_OK) {
        fprintf(stderr, "cheap_init failed\n");
        return 1;
    }

    /* --- KRR solve: alpha = (K + lambda*I)^{-1} y --- */
    double y[N], alpha[N];
    for (int i = 0; i < N; ++i)
        y[i] = sin(2.0 * M_PI * (double)i / N);

    double lambda_reg = 1e-3;
    double w_krr[N];
    for (int k = 0; k < N; ++k) {
        double denom = ctx.lambda[k] + lambda_reg;
        if (denom < CHEAP_EPS_DIV) denom = CHEAP_EPS_DIV;
        w_krr[k] = 1.0 / denom;
    }

    cheap_apply(&ctx, y, w_krr, alpha);

    /* Verify: K*alpha should approximate y */
    double K_alpha[N];
    cheap_apply(&ctx, alpha, ctx.lambda, K_alpha);
    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = fabs(K_alpha[i] + lambda_reg * alpha[i] - y[i]);
        if (err > max_err) max_err = err;
    }
    printf("KRR solve: max |K*alpha + lambda*alpha - y| = %.3e\n", max_err);

    /* --- Reparameterization: z = mu + iDCT(DCT(eps) * sqrt(lambda)) / (2N) --- */
    double mu[N], eps[N], z[N];
    for (int i = 0; i < N; ++i) {
        mu[i] = 0.0;
        eps[i] = (i == 3) ? 1.0 : 0.0;  /* single impulse */
    }

    cheap_apply(&ctx, eps, ctx.sqrt_lambda, z);
    for (int i = 0; i < N; ++i) z[i] += mu[i];

    printf("Reparam sample: z[0..4] = %.4f, %.4f, %.4f, %.4f, %.4f\n",
           z[0], z[1], z[2], z[3], z[4]);

    /* --- Batch KRR: solve multiple RHS --- */
    printf("\nBatch KRR solve (4 columns):\n");
    for (int c = 0; c < 4; ++c) {
        double y_c[N], a_c[N];
        for (int i = 0; i < N; ++i)
            y_c[i] = sin(2.0 * M_PI * (double)(c + 1) * (double)i / N);
        cheap_apply(&ctx, y_c, w_krr, a_c);

        double K_a[N];
        cheap_apply(&ctx, a_c, ctx.lambda, K_a);
        double me = 0.0;
        for (int i = 0; i < N; ++i) {
            double err = fabs(K_a[i] + lambda_reg * a_c[i] - y_c[i]);
            if (err > me) me = err;
        }
        printf("  col %d: max residual = %.3e\n", c, me);
    }

    cheap_destroy(&ctx);
    printf("\nGP regression example completed.\n");
    return 0;
}
