/*
 * Wiener denoise 2D — 256×256 synthetic image with additive white noise.
 */

#include "cheap.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double compute_snr(const double *signal, const double *noisy, int n)
{
    double sig_pow = 0.0, noise_pow = 0.0;
    for (int i = 0; i < n; ++i) {
        sig_pow += signal[i] * signal[i];
        double d = noisy[i] - signal[i];
        noise_pow += d * d;
    }
    return 10.0 * log10(sig_pow / fmax(noise_pow, 1e-300));
}

int main(void)
{
    const int nx = 256, ny = 256, n = nx * ny;
    const double sigma_noise = 0.5;

    cheap_ctx_2d ctx;
    if (cheap_init_2d(&ctx, nx, ny, 0.5, 0.5) != CHEAP_OK) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    double *clean = (double *)malloc((size_t)n * sizeof(double));
    double *noisy = (double *)malloc((size_t)n * sizeof(double));
    double *denoised = (double *)malloc((size_t)n * sizeof(double));
    double *w = (double *)malloc((size_t)n * sizeof(double));

    /* Synthetic image: sum of a few Gaussian blobs */
    for (int j = 0; j < nx; ++j) {
        for (int k = 0; k < ny; ++k) {
            double x = (double)j / nx;
            double y = (double)k / ny;
            double v = 0.0;
            v += exp(-50.0 * ((x - 0.3) * (x - 0.3) + (y - 0.4) * (y - 0.4)));
            v += exp(-80.0 * ((x - 0.7) * (x - 0.7) + (y - 0.6) * (y - 0.6)));
            v += 0.5 * sin(2.0 * M_PI * x * 3.0) * cos(2.0 * M_PI * y * 5.0);
            clean[j * ny + k] = v;
        }
    }

    /* Add white noise */
    srand(42);
    for (int i = 0; i < n; ++i) {
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        if (u1 < 1e-300) u1 = 1e-300;
        double r = sqrt(-2.0 * log(u1));
        double theta = 2.0 * M_PI * u2;
        double noise = sigma_noise * r * cos(theta);
        noisy[i] = clean[i] + noise;
    }

    /* Wiener filter with inverse-Laplacian signal power spectrum */
    double sigma_sq = sigma_noise * sigma_noise;
    cheap_weights_laplacian_2d(nx, ny, w);
    for (int i = 0; i < n; ++i) {
        double inv_lam = 1.0 / fmax(w[i], 1e-12);
        w[i] = inv_lam / (inv_lam + sigma_sq);
    }
    cheap_apply_2d(&ctx, noisy, w, denoised);

    double snr_in  = compute_snr(clean, noisy, n);
    double snr_out = compute_snr(clean, denoised, n);

    printf("wiener_denoise_2d: input SNR = %.2f dB, output SNR = %.2f dB\n",
           snr_in, snr_out);

    cheap_destroy_2d(&ctx);
    free(clean); free(noisy); free(denoised); free(w);
    return (snr_out > snr_in) ? 0 : 1;
}
