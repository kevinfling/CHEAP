/*
 * wiener_denoise.c — Signal denoising via Wiener filter weights.
 *
 * Demonstrates: cheap_weights_wiener, cheap_apply, SNR improvement.
 *
 * Build:
 *   gcc -std=c99 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include wiener_denoise.c -o wiener_denoise -lfftw3 -lm
 */

#include "cheap.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 256

int main(void)
{
    cheap_ctx ctx;
    if (cheap_init(&ctx, N, 0.5) != CHEAP_OK) {
        fprintf(stderr, "cheap_init failed\n");
        return 1;
    }

    /* Generate signal: sum of two sinusoids */
    double signal[N], noisy[N];
    for (int i = 0; i < N; ++i)
        signal[i] = sin(2.0 * M_PI * 3.0 * i / N)
                   + 0.5 * sin(2.0 * M_PI * 7.0 * i / N);

    /* Add white noise (simple LCG PRNG) */
    double sigma = 0.5;
    uint32_t lcg = 12345u;
    for (int i = 0; i < N; ++i) {
        lcg = lcg * 1664525u + 1013904223u;
        double u1 = ((double)(lcg >> 8) + 0.5) / (double)(1 << 24);
        lcg = lcg * 1664525u + 1013904223u;
        double u2 = ((double)(lcg >> 8) + 0.5) / (double)(1 << 24);
        if (u1 < 1e-12) u1 = 1e-12;
        double noise = sigma * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        noisy[i] = signal[i] + noise;
    }

    /* Compute input SNR */
    double sig_power = 0.0, noise_power = 0.0;
    for (int i = 0; i < N; ++i) {
        sig_power += signal[i] * signal[i];
        noise_power += (noisy[i] - signal[i]) * (noisy[i] - signal[i]);
    }
    double snr_in = 10.0 * log10(sig_power / noise_power);

    /* Apply Wiener filter */
    double weights[N], denoised[N];
    cheap_weights_wiener(N, sigma * sigma, weights);
    cheap_apply(&ctx, noisy, weights, denoised);

    /* Compute output SNR */
    double err_power = 0.0;
    for (int i = 0; i < N; ++i) {
        double e = denoised[i] - signal[i];
        err_power += e * e;
    }
    double snr_out = 10.0 * log10(sig_power / err_power);

    printf("Wiener Denoising Example\n");
    printf("========================\n");
    printf("Signal: 2 sinusoids, N=%d\n", N);
    printf("Noise: Gaussian, sigma=%.2f\n", sigma);
    printf("Input  SNR: %6.2f dB\n", snr_in);
    printf("Output SNR: %6.2f dB\n", snr_out);
    printf("Improvement: %+.2f dB\n", snr_out - snr_in);

    /* Show a few weight values */
    printf("\nWiener weights [0..7]: ");
    for (int k = 0; k < 8; ++k) printf("%.4f ", weights[k]);
    printf("...\n");

    cheap_destroy(&ctx);
    return 0;
}
