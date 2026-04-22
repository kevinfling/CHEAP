/*
 * Wiener denoise 2D — C++ wrapper version.
 */

#include "cheap.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double compute_snr(const std::vector<double>& signal,
                          const std::vector<double>& noisy)
{
    double sig_pow = 0.0, noise_pow = 0.0;
    for (size_t i = 0; i < signal.size(); ++i) {
        sig_pow += signal[i] * signal[i];
        double d = noisy[i] - signal[i];
        noise_pow += d * d;
    }
    return 10.0 * std::log10(sig_pow / std::max(noise_pow, 1e-300));
}

int main()
{
    const int nx = 256, ny = 256, n = nx * ny;
    const double sigma_noise = 0.5;

    cheap::Context2D ctx(nx, ny, 0.5, 0.5);

    std::vector<double> clean(n), noisy(n), denoised(n);

    for (int j = 0; j < nx; ++j) {
        for (int k = 0; k < ny; ++k) {
            double x = static_cast<double>(j) / nx;
            double y = static_cast<double>(k) / ny;
            double v = 0.0;
            v += std::exp(-50.0 * ((x - 0.3) * (x - 0.3) + (y - 0.4) * (y - 0.4)));
            v += std::exp(-80.0 * ((x - 0.7) * (x - 0.7) + (y - 0.6) * (y - 0.6)));
            v += 0.5 * std::sin(2.0 * M_PI * x * 3.0) * std::cos(2.0 * M_PI * y * 5.0);
            clean[j * ny + k] = v;
        }
    }

    std::srand(42);
    for (int i = 0; i < n; ++i) {
        double u1 = static_cast<double>(std::rand()) / RAND_MAX;
        double u2 = static_cast<double>(std::rand()) / RAND_MAX;
        if (u1 < 1e-300) u1 = 1e-300;
        double r = std::sqrt(-2.0 * std::log(u1));
        double theta = 2.0 * M_PI * u2;
        double noise = sigma_noise * r * std::cos(theta);
        noisy[i] = clean[i] + noise;
    }

    double sigma_sq = sigma_noise * sigma_noise;
    auto w = cheap::weights_laplacian_2d(nx, ny);
    for (int i = 0; i < n; ++i) {
        double inv_lam = 1.0 / std::max(w[i], 1e-12);
        w[i] = inv_lam / (inv_lam + sigma_sq);
    }
    denoised = ctx.apply(noisy.data(), w.data());

    double snr_in  = compute_snr(clean, noisy);
    double snr_out = compute_snr(clean, denoised);

    std::printf("wiener_denoise_2d_cpp: input SNR = %.2f dB, output SNR = %.2f dB\n",
                snr_in, snr_out);

    return (snr_out > snr_in) ? 0 : 1;
}
