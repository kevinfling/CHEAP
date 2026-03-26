/*
 * optimal_transport.cpp — Sinkhorn optimal transport via spectral acceleration.
 *
 * Demonstrates: cheap::Context::sinkhorn with fBm cost kernel.
 *
 * Build:
 *   g++ -std=c++17 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include optimal_transport.cpp -o optimal_transport -lfftw3 -lm
 */

#include "cheap.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int N = 128;

int main()
{
    cheap::Context ctx(N, 0.7);

    /* Two probability distributions: shifted Gaussians */
    std::vector<double> a(N), b(N);
    double sum_a = 0.0, sum_b = 0.0;
    for (int i = 0; i < N; ++i) {
        double x = static_cast<double>(i) / N;
        a[i] = std::exp(-0.5 * (x - 0.3) * (x - 0.3) / (0.05 * 0.05));
        b[i] = std::exp(-0.5 * (x - 0.7) * (x - 0.7) / (0.05 * 0.05));
        sum_a += a[i];
        sum_b += b[i];
    }
    for (int i = 0; i < N; ++i) {
        a[i] /= sum_a;
        b[i] /= sum_b;
    }

    std::vector<double> f(N), g(N);
    double eps = 0.01;
    int max_iter = 200;
    double tol = 1e-8;

    int rc = ctx.try_sinkhorn(a.data(), b.data(), eps, max_iter, tol, f.data(), g.data());
    if (rc == CHEAP_OK) {
        std::printf("Sinkhorn converged.\n");
        std::printf("  f[0..4] = %.4f, %.4f, %.4f, %.4f, %.4f\n",
                    f[0], f[1], f[2], f[3], f[4]);
        std::printf("  g[0..4] = %.4f, %.4f, %.4f, %.4f, %.4f\n",
                    g[0], g[1], g[2], g[3], g[4]);
    } else if (rc == CHEAP_ENOCONV) {
        std::printf("Sinkhorn did not converge within %d iterations.\n", max_iter);
    } else {
        std::printf("Sinkhorn error: %d\n", rc);
    }

    std::printf("\nOptimal transport example completed.\n");
    return 0;
}
