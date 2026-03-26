/*
 * toeplitz_solve.cpp — Generic Toeplitz matvec and solve via DCT diagonalization.
 *
 * Demonstrates: cheap::Context::toeplitz_eigenvalues, apply, toeplitz_solve_precomp.
 *
 * Build:
 *   g++ -std=c++17 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include toeplitz_solve.cpp -o toeplitz_solve -lfftw3 -lm
 */

#include "cheap.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int N = 64;

int main()
{
    /* H value is arbitrary here — we only use the FFTW plans, not ctx.lambda() */
    cheap::Context ctx(N, 0.5);

    /* Tridiagonal Toeplitz: t = [2, -1, 0, 0, ...] */
    std::vector<double> t(N, 0.0);
    t[0] = 2.0;
    t[1] = -1.0;

    /* Compute eigenvalues once */
    auto lambda_t = ctx.toeplitz_eigenvalues(t.data());
    std::printf("Toeplitz eigenvalues [0..4]: %.4f, %.4f, %.4f, %.4f, %.4f\n",
                lambda_t[0], lambda_t[1], lambda_t[2], lambda_t[3], lambda_t[4]);

    /* Matvec: y = T * x using apply with eigenvalues as weights */
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = static_cast<double>(i + 1);
    auto y = ctx.apply(x.data(), lambda_t.data());

    /* Verify against brute-force */
    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double y_ref = 0.0;
        for (int j = 0; j < N; ++j)
            y_ref += t[std::abs(i - j)] * x[j];
        double err = std::fabs(y[i] - y_ref);
        if (err > max_err) max_err = err;
    }
    std::printf("Matvec error vs brute-force: %.3e\n", max_err);

    /* Solve: x_sol = (T + lambda*I)^{-1} y */
    double lambda_reg = 1e-3;
    auto x_sol = ctx.toeplitz_solve_precomp(lambda_t.data(), y.data(), lambda_reg);

    /* Verify residual: ||T*x_sol + lambda*x_sol - y|| */
    std::vector<double> Tx_sol(N);
    ctx.apply(x_sol.data(), lambda_t.data(), Tx_sol.data());
    double max_res = 0.0;
    for (int i = 0; i < N; ++i) {
        double res = std::fabs(Tx_sol[i] + lambda_reg * x_sol[i] - y[i]);
        if (res > max_res) max_res = res;
    }
    std::printf("Solve residual: ||T*x + lambda*x - y||_inf = %.3e\n", max_res);

    std::printf("\nToeplitz solve example completed.\n");
    return 0;
}
