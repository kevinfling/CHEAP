/*
 * rmt_denoise.cpp — Random Matrix Theory denoising via Marchenko-Pastur.
 *
 * Demonstrates: cheap::weights_rmt_hard, cheap::weights_rmt_shrink,
 *               comparison of hard thresholding vs optimal shrinkage.
 *
 * Build:
 *   g++ -std=c++17 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include rmt_denoise.cpp -o rmt_denoise -lfftw3 -lm
 */

#include "cheap.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int N = 64;

int main()
{
    const double sigma_sq = 1.0;
    const double c = 0.5;
    const double sc = std::sqrt(c);
    const double lambda_plus = sigma_sq * (1.0 + sc) * (1.0 + sc);
    const double lambda_minus = sigma_sq * (1.0 - sc) * (1.0 - sc);

    std::printf("RMT Denoising Example (C++)\n");
    std::printf("===========================\n");
    std::printf("N = %d, sigma^2 = %.1f, c = %.2f\n", N, sigma_sq, c);
    std::printf("Marchenko-Pastur edges: lambda- = %.4f, lambda+ = %.4f\n",
                lambda_minus, lambda_plus);

    /* Create synthetic eigenvalue spectrum */
    std::vector<double> eigenvalues(N);
    eigenvalues[0] = 20.0;
    eigenvalues[1] = 10.0;
    eigenvalues[2] = 5.0;
    for (int k = 3; k < N; ++k)
        eigenvalues[k] = lambda_minus
            + (lambda_plus - lambda_minus) * static_cast<double>(k - 3)
            / static_cast<double>(N - 4);

    /* Hard thresholding */
    auto w_hard = cheap::weights_rmt_hard(eigenvalues.data(), N, sigma_sq, c);

    int hard_retained = 0;
    for (int k = 0; k < N; ++k)
        if (w_hard[k] > 0.0) ++hard_retained;

    std::printf("\nHard threshold: retained %d / %d eigenvalues\n", hard_retained, N);
    std::printf("  Spike eigenvalues kept: ");
    for (int k = 0; k < 3; ++k) std::printf("%.2f ", w_hard[k]);
    std::printf("\n");

    /* Optimal shrinkage */
    auto w_shrink = cheap::weights_rmt_shrink(eigenvalues.data(), N, sigma_sq, c);

    int shrink_retained = 0;
    for (int k = 0; k < N; ++k)
        if (w_shrink[k] > 0.0) ++shrink_retained;

    std::printf("\nOptimal shrinkage: retained %d / %d eigenvalues\n",
                shrink_retained, N);

    std::printf("\nComparison (spikes):\n");
    std::printf("  %-12s %-12s %-12s %-12s\n", "True", "Observed", "Hard", "Shrinkage");
    double true_spikes[] = {20.0, 10.0, 5.0};
    for (int k = 0; k < 3; ++k)
        std::printf("  %-12.4f %-12.4f %-12.4f %-12.4f\n",
                    true_spikes[k], eigenvalues[k], w_hard[k], w_shrink[k]);

    std::printf("\nRMT denoising example completed.\n");
    return 0;
}
