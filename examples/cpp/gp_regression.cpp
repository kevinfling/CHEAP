/*
 * gp_regression.cpp — Gaussian process regression via spectral KRR + reparameterization.
 *
 * Demonstrates: cheap::Context::apply with KRR weights and sqrt-lambda weights.
 *
 * Build:
 *   g++ -std=c++17 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include gp_regression.cpp -o gp_regression -lfftw3 -lm
 */

#include "cheap.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int N = 128;

int main()
{
    cheap::Context ctx(N, 0.7);

    /* --- KRR solve: alpha = (K + lambda*I)^{-1} y --- */
    std::vector<double> y(N), alpha(N);
    for (int i = 0; i < N; ++i)
        y[i] = std::sin(2.0 * M_PI * static_cast<double>(i) / N);

    double lambda_reg = 1e-3;
    std::vector<double> w_krr(N);
    for (int k = 0; k < N; ++k) {
        double denom = ctx.lambda()[k] + lambda_reg;
        if (denom < CHEAP_EPS_DIV) denom = CHEAP_EPS_DIV;
        w_krr[k] = 1.0 / denom;
    }

    ctx.apply(y.data(), w_krr.data(), alpha.data());

    /* Verify: K*alpha should approximate y */
    std::vector<double> K_alpha(N);
    ctx.apply(alpha.data(), ctx.lambda(), K_alpha.data());
    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = std::fabs(K_alpha[i] + lambda_reg * alpha[i] - y[i]);
        if (err > max_err) max_err = err;
    }
    std::printf("KRR solve: max |K*alpha + lambda*alpha - y| = %.3e\n", max_err);

    /* --- Reparameterization: z = mu + iDCT(DCT(eps) * sqrt(lambda)) / (2N) --- */
    std::vector<double> eps(N, 0.0), z(N);
    eps[3] = 1.0; /* single impulse */

    ctx.apply(eps.data(), ctx.sqrt_lambda(), z.data());

    std::printf("Reparam sample: z[0..4] = %.4f, %.4f, %.4f, %.4f, %.4f\n",
                z[0], z[1], z[2], z[3], z[4]);

    /* --- Batch KRR: solve multiple RHS --- */
    std::printf("\nBatch KRR solve (4 columns):\n");
    for (int c = 0; c < 4; ++c) {
        std::vector<double> y_c(N), a_c(N);
        for (int i = 0; i < N; ++i)
            y_c[i] = std::sin(2.0 * M_PI * static_cast<double>(c + 1) * static_cast<double>(i) / N);
        ctx.apply(y_c.data(), w_krr.data(), a_c.data());

        std::vector<double> K_a(N);
        ctx.apply(a_c.data(), ctx.lambda(), K_a.data());
        double me = 0.0;
        for (int i = 0; i < N; ++i) {
            double err = std::fabs(K_a[i] + lambda_reg * a_c[i] - y_c[i]);
            if (err > me) me = err;
        }
        std::printf("  col %d: max residual = %.3e\n", c, me);
    }

    std::printf("\nGP regression example completed.\n");
    return 0;
}
