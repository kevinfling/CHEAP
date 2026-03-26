/*
 * rmt_denoise.c — Random Matrix Theory denoising via Marchenko-Pastur.
 *
 * Demonstrates: cheap_weights_rmt_hard, cheap_weights_rmt_shrink,
 *               comparison of hard thresholding vs optimal shrinkage.
 *
 * Build:
 *   gcc -std=c99 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include rmt_denoise.c -o rmt_denoise -lfftw3 -lm
 */

#include "cheap.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 64

int main(void)
{
    /*
     * Simulate a spiked covariance model:
     *   - True eigenvalues: 3 spikes above the bulk, rest = noise floor
     *   - Noise: sigma^2 = 1.0, aspect ratio c = 0.5
     */
    const double sigma_sq = 1.0;
    const double c = 0.5;
    const double sc = sqrt(c);
    const double lambda_plus = sigma_sq * (1.0 + sc) * (1.0 + sc);
    const double lambda_minus = sigma_sq * (1.0 - sc) * (1.0 - sc);

    printf("RMT Denoising Example\n");
    printf("=====================\n");
    printf("N = %d, sigma^2 = %.1f, c = %.2f\n", N, sigma_sq, c);
    printf("Marchenko-Pastur edges: lambda- = %.4f, lambda+ = %.4f\n",
           lambda_minus, lambda_plus);

    /* Create synthetic eigenvalue spectrum */
    double eigenvalues[N];
    /* 3 signal spikes well above the bulk */
    eigenvalues[0] = 20.0;
    eigenvalues[1] = 10.0;
    eigenvalues[2] = 5.0;
    /* Remaining eigenvalues in the MP bulk: uniform in [lambda_-, lambda_+] */
    for (int k = 3; k < N; ++k)
        eigenvalues[k] = lambda_minus
            + (lambda_plus - lambda_minus) * (double)(k - 3) / (double)(N - 4);

    /* Hard thresholding */
    double w_hard[N];
    cheap_weights_rmt_hard(eigenvalues, N, sigma_sq, c, w_hard);

    int hard_retained = 0;
    for (int k = 0; k < N; ++k)
        if (w_hard[k] > 0.0) ++hard_retained;

    printf("\nHard threshold: retained %d / %d eigenvalues\n", hard_retained, N);
    printf("  Spike eigenvalues kept: ");
    for (int k = 0; k < 3; ++k) printf("%.2f ", w_hard[k]);
    printf("\n  Bulk eigenvalues zeroed: %s\n",
           (w_hard[3] == 0.0 && w_hard[N-1] == 0.0) ? "yes" : "no");

    /* Optimal shrinkage */
    double w_shrink[N];
    cheap_weights_rmt_shrink(eigenvalues, N, sigma_sq, c, w_shrink);

    int shrink_retained = 0;
    for (int k = 0; k < N; ++k)
        if (w_shrink[k] > 0.0) ++shrink_retained;

    printf("\nOptimal shrinkage: retained %d / %d eigenvalues\n",
           shrink_retained, N);
    printf("  Shrunk spike eigenvalues: ");
    for (int k = 0; k < 3; ++k) printf("%.4f ", w_shrink[k]);
    printf("\n");

    /* Compare: shrinkage reduces spike eigenvalues vs passthrough */
    printf("\nComparison (spikes):\n");
    printf("  %-12s %-12s %-12s %-12s\n", "True", "Observed", "Hard", "Shrinkage");
    double true_spikes[] = {20.0, 10.0, 5.0};
    for (int k = 0; k < 3; ++k)
        printf("  %-12.4f %-12.4f %-12.4f %-12.4f\n",
               true_spikes[k], eigenvalues[k], w_hard[k], w_shrink[k]);

    printf("\nRMT denoising example completed.\n");
    return 0;
}
