/*
 * lqr_mpc.cpp — LQR Tikhonov solve, Kalman prediction, and MPC step.
 *
 * Demonstrates: cheap::Context::apply with control-specific weight vectors.
 *
 * Build:
 *   g++ -std=c++17 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include lqr_mpc.cpp -o lqr_mpc -lfftw3 -lm
 */

#include "cheap.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int N = 64;

int main()
{
    cheap::Context ctx(N, 0.7);

    /* --- LQR Tikhonov: P = iDCT( DCT(Q) / (lambda + R) ) / (2N) --- */
    std::vector<double> Q(N), P(N);
    for (int i = 0; i < N; ++i)
        Q[i] = 1.0 + 0.5 * std::cos(2.0 * M_PI * static_cast<double>(i) / N);

    double R_val = 0.1;
    std::vector<double> w_lqr(N);
    for (int k = 0; k < N; ++k) {
        double denom = ctx.lambda()[k] + R_val;
        if (denom < CHEAP_EPS_DIV) denom = CHEAP_EPS_DIV;
        w_lqr[k] = 1.0 / denom;
    }
    ctx.apply(Q.data(), w_lqr.data(), P.data());

    std::printf("LQR Tikhonov:\n");
    std::printf("  P[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n",
                P[0], P[1], P[2], P[3], P[4]);

    /* --- MPC step: u = (K + 0*I)^{-1} Q  (i.e., lambda_reg = 0) --- */
    std::vector<double> u(N), w_mpc(N);
    for (int k = 0; k < N; ++k) {
        double denom = ctx.lambda()[k];
        if (denom < CHEAP_EPS_DIV) denom = CHEAP_EPS_DIV;
        w_mpc[k] = 1.0 / denom;
    }
    ctx.apply(Q.data(), w_mpc.data(), u.data());

    std::printf("\nMPC step:\n");
    std::printf("  u[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n",
                u[0], u[1], u[2], u[3], u[4]);

    /* --- Kalman predict: P_minus[k] = P_diag[k] + lambda[k] --- */
    std::vector<double> P_minus(N);
    for (int k = 0; k < N; ++k)
        P_minus[k] = 1.0 + ctx.lambda()[k];

    std::printf("\nKalman predict:\n");
    std::printf("  P_minus[0..4] = %.4f, %.4f, %.4f, %.4f, %.4f\n",
                P_minus[0], P_minus[1], P_minus[2], P_minus[3], P_minus[4]);

    std::printf("\nLQR/MPC example completed.\n");
    return 0;
}
