/*
 * poisson_solver.cpp — 1D spectral Poisson solve with Neumann BC.
 *
 * Solves: -d^2(phi)/dx^2 = f using DCT diagonalization.
 * DC mode set to zero (zero-mean gauge).
 *
 * Demonstrates: cheap::Context::forward, workspace manipulation,
 *               cheap::Context::inverse.
 *
 * Build:
 *   g++ -std=c++17 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include poisson_solver.cpp -o poisson_solver -lfftw3 -lm
 */

#include "cheap.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int N = 256;

int main()
{
    cheap::Context ctx(N, 0.5);

    double dx = 1.0 / N;
    double dx2 = dx * dx;

    /* Source term: smooth bump */
    std::vector<double> f(N);
    for (int i = 0; i < N; ++i) {
        double x = static_cast<double>(i) / N;
        f[i] = std::sin(2.0 * M_PI * x);
    }

    /* Poisson solve: forward transform, divide by Laplacian eigenvalues, inverse */
    ctx.forward(f.data());

    /* DC mode: zero-mean gauge */
    ctx.workspace()[0] = 0.0;
    for (int k = 1; k < N; ++k) {
        double s = std::sin(M_PI * static_cast<double>(k) / (2.0 * static_cast<double>(N)));
        double lambda_k = 4.0 * s * s;
        if (lambda_k < CHEAP_EPS_DIV) lambda_k = CHEAP_EPS_DIV;
        ctx.workspace()[k] *= dx2 / lambda_k;
    }

    std::vector<double> phi(N);
    ctx.inverse(phi.data());

    std::printf("Poisson solve: -phi'' = f\n");
    std::printf("  phi[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n",
                phi[0], phi[1], phi[2], phi[3], phi[4]);

    /* Verify: discrete Laplacian of phi should approximate f (up to DC) */
    double max_err = 0.0;
    double f_mean = 0.0;
    for (int i = 0; i < N; ++i) f_mean += f[i];
    f_mean /= N;

    for (int i = 1; i < N - 1; ++i) {
        double lap = -(phi[i-1] - 2.0 * phi[i] + phi[i+1]) / dx2;
        double err = std::fabs(lap - (f[i] - f_mean));
        if (err > max_err) max_err = err;
    }
    std::printf("  Max Laplacian residual (interior): %.3e\n", max_err);

    /* Roundtrip test */
    ctx.forward(phi.data());
    ctx.workspace()[0] = 0.0;
    for (int k = 1; k < N; ++k) {
        double s = std::sin(M_PI * static_cast<double>(k) / (2.0 * static_cast<double>(N)));
        double lambda_k = 4.0 * s * s;
        ctx.workspace()[k] *= lambda_k / dx2;
    }
    std::vector<double> f_rec(N);
    ctx.inverse(f_rec.data());

    double max_rt_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = std::fabs(f_rec[i] - (f[i] - f_mean));
        if (err > max_rt_err) max_rt_err = err;
    }
    std::printf("  Roundtrip error: %.3e\n", max_rt_err);

    std::printf("\nPoisson solver example completed.\n");
    return 0;
}
