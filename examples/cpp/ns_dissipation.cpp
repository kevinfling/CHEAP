/*
 * ns_dissipation.cpp — Fractional Navier-Stokes dissipation step.
 *
 * Applies: u_new[k] = iDCT( DCT(u)[k] * exp(-nu * lambda_k^alpha * dt) )
 * where lambda_k = (2*sin(pi*k/(2*N)))^2 are discrete Laplacian eigenvalues.
 *
 * Demonstrates: cheap::Context::apply with precomputed dissipation weights.
 *
 * Build:
 *   g++ -std=c++17 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include ns_dissipation.cpp -o ns_dissipation -lfftw3 -lm
 */

#include "cheap.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

constexpr int N = 128;

int main()
{
    cheap::Context ctx(N, 0.5);

    double nu = 0.01;
    double alpha = 1.0;   /* standard diffusion; try 0.5 for fractional */
    double dt = 0.001;
    int nsteps = 100;

    /* Initial condition: single Fourier mode */
    std::vector<double> u(N);
    for (int i = 0; i < N; ++i)
        u[i] = std::cos(2.0 * M_PI * static_cast<double>(i) / N);

    std::printf("NS dissipation: nu=%.3f, alpha=%.1f, dt=%.4f, steps=%d\n",
                nu, alpha, dt, nsteps);
    std::printf("  Initial: u[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n",
                u[0], u[1], u[2], u[3], u[4]);

    /* Precompute dissipation weights for one timestep */
    std::vector<double> w_diss(N);
    for (int k = 0; k < N; ++k) {
        double s = std::sin(M_PI * static_cast<double>(k) / (2.0 * static_cast<double>(N)));
        double lambda_k = 4.0 * s * s;
        double exponent = -nu * std::pow(lambda_k, alpha) * dt;
        w_diss[k] = (exponent < -700.0) ? 0.0 : std::exp(exponent);
    }

    /* Time-stepping */
    std::vector<double> u_new(N);
    for (int step = 0; step < nsteps; ++step) {
        ctx.apply(u.data(), w_diss.data(), u_new.data());
        std::memcpy(u.data(), u_new.data(), static_cast<std::size_t>(N) * sizeof(double));
    }

    std::printf("  After %d steps: u[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n",
                nsteps, u[0], u[1], u[2], u[3], u[4]);

    /* Check energy decay */
    double energy = 0.0;
    for (int i = 0; i < N; ++i) energy += u[i] * u[i];
    energy /= N;
    std::printf("  Energy (mean u^2): %.6e\n", energy);
    std::printf("  Expected decay factor per step for mode k=1: exp(%.4f)\n",
                -nu * std::pow(4.0 * std::sin(M_PI / (2.0 * N)) * std::sin(M_PI / (2.0 * N)), alpha) * dt);

    std::printf("\nNS dissipation example completed.\n");
    return 0;
}
