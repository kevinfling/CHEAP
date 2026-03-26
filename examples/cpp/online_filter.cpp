/*
 * online_filter.cpp — Online kernel adaptive filter (NT-KLMS) using RFF.
 *
 * Demonstrates: cheap::RffContext, cheap::Context::toeplitz_eigenvalues,
 *               cheap::Context::toeplitz_solve_precomp.
 *
 * Build:
 *   g++ -std=c++17 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include online_filter.cpp -o online_filter -lfftw3 -lm
 */

#include "cheap.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

constexpr int D = 64;         /* RFF feature dimension */
constexpr int N_TRAIN = 500;  /* training samples */
constexpr int N_TEST = 50;    /* test samples */

static double target_fn(double x) { return std::sin(x); }

int main()
{
    cheap::RffContext rff(D, 1, 1.0, 42);

    /* Simple online KLMS with RFF features */
    std::vector<double> w(D, 0.0);
    double eta = 0.01;

    std::vector<double> z(D);
    double mse_window = 0.0;
    int window = 50;

    std::printf("Online KLMS filter (D=%d, eta=%.3f):\n", D, eta);

    for (int t = 0; t < N_TRAIN; ++t) {
        double x = 6.0 * static_cast<double>(t) / N_TRAIN - 3.0;
        double y_true = target_fn(x);

        rff.map(&x, z.data());

        /* Predict */
        double y_pred = 0.0;
        for (int i = 0; i < D; ++i) y_pred += w[i] * z[i];

        /* Update */
        double e = y_true - y_pred;
        for (int i = 0; i < D; ++i) w[i] += eta * e * z[i];

        /* Track MSE */
        mse_window += e * e;
        if ((t + 1) % window == 0) {
            std::printf("  step %3d: MSE(last %d) = %.6f\n", t + 1, window, mse_window / window);
            mse_window = 0.0;
        }
    }

    /* Test */
    double test_mse = 0.0;
    for (int t = 0; t < N_TEST; ++t) {
        double x = 6.0 * static_cast<double>(t) / N_TEST - 3.0;
        double y_true = target_fn(x);

        rff.map(&x, z.data());
        double y_pred = 0.0;
        for (int i = 0; i < D; ++i) y_pred += w[i] * z[i];

        test_mse += (y_true - y_pred) * (y_true - y_pred);
    }
    test_mse /= N_TEST;
    std::printf("\nTest MSE: %.6f\n", test_mse);

    /* --- Demonstrate batch correction via Toeplitz solve --- */
    std::printf("\nBatch correction demo (Toeplitz solve on RFF Gram):\n");

    constexpr int L = 32;
    cheap::Context dctx(L, 0.5);

    std::vector<double> Z_buf(L * D, 0.0), Y_buf(L);
    for (int i = 0; i < L; ++i) {
        double x = 6.0 * static_cast<double>(i) / L - 3.0;
        Y_buf[i] = target_fn(x);
        rff.map(&x, Z_buf.data() + i * D);
    }

    /* Form Gram first-column (approximate Toeplitz) */
    std::vector<double> gram_col(L);
    for (int j = 0; j < L; ++j) {
        double sum = 0.0;
        for (int i = 0; i < L; ++i) {
            const double* zi = Z_buf.data() + i * D;
            const double* zj = Z_buf.data() + ((i + j) % L) * D;
            double dp = 0.0;
            for (int d = 0; d < D; ++d) dp += zi[d] * zj[d];
            sum += dp;
        }
        gram_col[j] = sum / L;
    }

    auto lambda_g = dctx.toeplitz_eigenvalues(gram_col.data());
    auto alpha = dctx.toeplitz_solve_precomp(lambda_g.data(), Y_buf.data(), 0.1);

    /* Project: w_batch = Z^T * alpha */
    std::vector<double> w_batch(D, 0.0);
    for (int i = 0; i < L; ++i) {
        const double* zi = Z_buf.data() + i * D;
        for (int d = 0; d < D; ++d)
            w_batch[d] += alpha[i] * zi[d];
    }

    /* Test batch-corrected weights */
    double batch_mse = 0.0;
    for (int t = 0; t < N_TEST; ++t) {
        double x = 6.0 * static_cast<double>(t) / N_TEST - 3.0;
        double y_true = target_fn(x);

        rff.map(&x, z.data());
        double y_pred = 0.0;
        for (int i = 0; i < D; ++i) y_pred += w_batch[i] * z[i];

        batch_mse += (y_true - y_pred) * (y_true - y_pred);
    }
    batch_mse /= N_TEST;
    std::printf("  Batch-corrected test MSE: %.6f\n", batch_mse);

    std::printf("\nOnline filter example completed.\n");
    return 0;
}
