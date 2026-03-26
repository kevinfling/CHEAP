/*
 * fractional_diff.cpp — Fractional differentiation and integration via DCT.
 *
 * Demonstrates: cheap::Context::forward, workspace manipulation,
 *               cheap::Context::inverse, and cheap::Context::apply.
 *
 * Build:
 *   g++ -std=c++17 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include fractional_diff.cpp -o fractional_diff -lfftw3 -lm
 */

#include "cheap.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int N = 128;

/*
 * Fractional differentiation weight: w[k] = (2*sin(omega_k/2))^d
 * where omega_k = pi*k/N.
 */
static void compute_frac_weights(std::vector<double>& w, double d_order)
{
    const int n = static_cast<int>(w.size());
    for (int k = 0; k < n; ++k) {
        double omega = M_PI * static_cast<double>(k) / static_cast<double>(n);
        double sin_half = std::sin(0.5 * omega);
        if (sin_half < CHEAP_EPS_LOG) sin_half = CHEAP_EPS_LOG;
        w[k] = std::pow(2.0 * sin_half, d_order);
    }
}

int main()
{
    cheap::Context ctx(N, 0.5);

    /* Test signal: smooth ramp */
    std::vector<double> z(N);
    for (int i = 0; i < N; ++i)
        z[i] = static_cast<double>(i) / N;

    /* --- Fractional differentiation (d=0.4) --- */
    std::vector<double> w_diff(N), z_diff(N);
    compute_frac_weights(w_diff, 0.4);
    ctx.apply(z.data(), w_diff.data(), z_diff.data());

    std::printf("Fractional diff (d=0.4):\n");
    std::printf("  z_diff[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n",
                z_diff[0], z_diff[1], z_diff[2], z_diff[3], z_diff[4]);

    /* --- Fractional integration (d=-0.4) to recover original --- */
    std::vector<double> w_int(N), z_back(N);
    compute_frac_weights(w_int, -0.4);
    ctx.apply(z_diff.data(), w_int.data(), z_back.data());

    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = std::fabs(z_back[i] - z[i]);
        if (err > max_err) max_err = err;
    }
    std::printf("\nRoundtrip error: max |z_back - z| = %.3e\n", max_err);

    /* --- Identity test: d=0 should be passthrough --- */
    std::vector<double> w_id(N), z_id(N);
    compute_frac_weights(w_id, 0.0);
    ctx.apply(z.data(), w_id.data(), z_id.data());

    double max_id_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = std::fabs(z_id[i] - z[i]);
        if (err > max_id_err) max_id_err = err;
    }
    std::printf("Identity (d=0) error: max |z_id - z| = %.3e\n", max_id_err);

    /* --- Using forward/inverse directly for custom manipulation --- */
    std::printf("\nDirect forward/inverse demo:\n");
    ctx.forward(z.data());
    std::printf("  Spectral coeffs [0..4] = %.4f, %.4f, %.4f, %.4f, %.4f\n",
                ctx.workspace()[0], ctx.workspace()[1], ctx.workspace()[2],
                ctx.workspace()[3], ctx.workspace()[4]);

    /* Apply fractional weights manually */
    for (int k = 0; k < N; ++k)
        ctx.workspace()[k] *= w_diff[k];

    std::vector<double> z_diff2(N);
    ctx.inverse(z_diff2.data());

    /* Should match apply result */
    double max_diff = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = std::fabs(z_diff2[i] - z_diff[i]);
        if (err > max_diff) max_diff = err;
    }
    std::printf("  forward/inverse vs apply diff: %.3e\n", max_diff);

    std::printf("\nFractional diff example completed.\n");
    return 0;
}
