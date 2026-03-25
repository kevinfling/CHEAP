/*
 * online_filter.c — Online kernel adaptive filter (NT-KLMS) using RFF.
 *
 * Demonstrates: cheap_rff_*, cheap_toeplitz_solve_precomp, cheap_toeplitz_eigenvalues.
 *
 * Build:
 *   gcc -std=c99 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include online_filter.c -o online_filter -lfftw3 -lm
 */

#include "cheap.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define D 64         /* RFF feature dimension */
#define N_TRAIN 500  /* training samples */
#define N_TEST 50    /* test samples */

/* Target function: sin(x) + 0.1*noise */
static double target_fn(double x)
{
    return sin(x);
}

int main(void)
{
    cheap_rff_ctx rctx;
    if (cheap_rff_init(&rctx, D, 1, 1.0, 42) != CHEAP_OK) {
        fprintf(stderr, "RFF init failed\n");
        return 1;
    }

    /* Simple online KLMS with RFF features */
    double w[D];
    memset(w, 0, sizeof(w));
    double eta = 0.01;

    double z[D];
    double mse_window = 0.0;
    int window = 50;

    printf("Online KLMS filter (D=%d, eta=%.3f):\n", D, eta);

    for (int t = 0; t < N_TRAIN; ++t) {
        double x = 6.0 * (double)t / N_TRAIN - 3.0;  /* [-3, 3] */
        double y_true = target_fn(x);

        cheap_rff_map(&rctx, &x, z);

        /* Predict */
        double y_pred = 0.0;
        for (int i = 0; i < D; ++i) y_pred += w[i] * z[i];

        /* Update */
        double e = y_true - y_pred;
        for (int i = 0; i < D; ++i) w[i] += eta * e * z[i];

        /* Track MSE */
        mse_window += e * e;
        if ((t + 1) % window == 0) {
            printf("  step %3d: MSE(last %d) = %.6f\n", t + 1, window, mse_window / window);
            mse_window = 0.0;
        }
    }

    /* Test */
    double test_mse = 0.0;
    for (int t = 0; t < N_TEST; ++t) {
        double x = 6.0 * (double)t / N_TEST - 3.0;
        double y_true = target_fn(x);

        cheap_rff_map(&rctx, &x, z);
        double y_pred = 0.0;
        for (int i = 0; i < D; ++i) y_pred += w[i] * z[i];

        test_mse += (y_true - y_pred) * (y_true - y_pred);
    }
    test_mse /= N_TEST;
    printf("\nTest MSE: %.6f\n", test_mse);

    /* --- Demonstrate batch correction via Toeplitz solve --- */
    printf("\nBatch correction demo (Toeplitz solve on RFF Gram):\n");

    /* Build feature matrix and targets for a small batch */
    int L = 32;
    cheap_ctx dctx;
    if (cheap_init(&dctx, L, 0.5) != CHEAP_OK) {
        fprintf(stderr, "DCT context init failed\n");
        cheap_rff_destroy(&rctx);
        return 1;
    }

    double* Z_buf = (double*)calloc((size_t)(L * D), sizeof(double));
    double* Y_buf = (double*)malloc((size_t)L * sizeof(double));
    if (!Z_buf || !Y_buf) {
        free(Z_buf); free(Y_buf);
        cheap_destroy(&dctx);
        cheap_rff_destroy(&rctx);
        return 1;
    }

    for (int i = 0; i < L; ++i) {
        double x = 6.0 * (double)i / L - 3.0;
        Y_buf[i] = target_fn(x);
        cheap_rff_map(&rctx, &x, Z_buf + i * D);
    }

    /* Form Gram first-column (approximate Toeplitz) */
    double gram_col[32];  /* VLA avoided */
    for (int j = 0; j < L; ++j) {
        double sum = 0.0;
        for (int i = 0; i < L; ++i) {
            const double* zi = Z_buf + i * D;
            const double* zj = Z_buf + ((i + j) % L) * D;
            double dp = 0.0;
            for (int d = 0; d < D; ++d) dp += zi[d] * zj[d];
            sum += dp;
        }
        gram_col[j] = sum / L;
    }

    double lambda_g[32], alpha[32];
    cheap_toeplitz_eigenvalues(&dctx, gram_col, lambda_g);
    cheap_toeplitz_solve_precomp(&dctx, lambda_g, Y_buf, 0.1, alpha);

    /* Project: w_batch = Z^T * alpha */
    double w_batch[D];
    memset(w_batch, 0, sizeof(w_batch));
    for (int i = 0; i < L; ++i) {
        const double* zi = Z_buf + i * D;
        for (int d = 0; d < D; ++d)
            w_batch[d] += alpha[i] * zi[d];
    }

    /* Test batch-corrected weights */
    double batch_mse = 0.0;
    for (int t = 0; t < N_TEST; ++t) {
        double x = 6.0 * (double)t / N_TEST - 3.0;
        double y_true = target_fn(x);

        cheap_rff_map(&rctx, &x, z);
        double y_pred = 0.0;
        for (int i = 0; i < D; ++i) y_pred += w_batch[i] * z[i];

        batch_mse += (y_true - y_pred) * (y_true - y_pred);
    }
    batch_mse /= N_TEST;
    printf("  Batch-corrected test MSE: %.6f\n", batch_mse);

    free(Z_buf);
    free(Y_buf);
    cheap_destroy(&dctx);
    cheap_rff_destroy(&rctx);
    printf("\nOnline filter example completed.\n");
    return 0;
}
