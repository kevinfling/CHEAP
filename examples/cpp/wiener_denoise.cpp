/*
 * wiener_denoise.cpp — Signal denoising via Wiener filter weights.
 *
 * Demonstrates: cheap::weights_wiener, cheap::Context::apply, SNR improvement.
 *
 * Build:
 *   g++ -std=c++17 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       -I../include wiener_denoise.cpp -o wiener_denoise -lfftw3 -lm
 */

#include "cheap.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

constexpr int N = 256;

int main()
{
    cheap::Context ctx(N, 0.5);

    /* Generate signal: sum of two sinusoids */
    std::vector<double> signal(N), noisy(N);
    for (int i = 0; i < N; ++i)
        signal[i] = std::sin(2.0 * M_PI * 3.0 * i / N)
                   + 0.5 * std::sin(2.0 * M_PI * 7.0 * i / N);

    /* Add white noise (simple LCG PRNG) */
    double sigma = 0.5;
    std::uint32_t lcg = 12345u;
    for (int i = 0; i < N; ++i) {
        lcg = lcg * 1664525u + 1013904223u;
        double u1 = (static_cast<double>(lcg >> 8) + 0.5) / static_cast<double>(1 << 24);
        lcg = lcg * 1664525u + 1013904223u;
        double u2 = (static_cast<double>(lcg >> 8) + 0.5) / static_cast<double>(1 << 24);
        if (u1 < 1e-12) u1 = 1e-12;
        double noise = sigma * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        noisy[i] = signal[i] + noise;
    }

    /* Compute input SNR */
    double sig_power = 0.0, noise_power = 0.0;
    for (int i = 0; i < N; ++i) {
        sig_power += signal[i] * signal[i];
        noise_power += (noisy[i] - signal[i]) * (noisy[i] - signal[i]);
    }
    double snr_in = 10.0 * std::log10(sig_power / noise_power);

    /* Apply Wiener filter */
    auto weights = cheap::weights_wiener(N, sigma * sigma);
    auto denoised = ctx.apply(noisy.data(), weights.data());

    /* Compute output SNR */
    double err_power = 0.0;
    for (int i = 0; i < N; ++i) {
        double e = denoised[i] - signal[i];
        err_power += e * e;
    }
    double snr_out = 10.0 * std::log10(sig_power / err_power);

    std::printf("Wiener Denoising Example (C++)\n");
    std::printf("==============================\n");
    std::printf("Signal: 2 sinusoids, N=%d\n", N);
    std::printf("Noise: Gaussian, sigma=%.2f\n", sigma);
    std::printf("Input  SNR: %6.2f dB\n", snr_in);
    std::printf("Output SNR: %6.2f dB\n", snr_out);
    std::printf("Improvement: %+.2f dB\n", snr_out - snr_in);

    std::printf("\nWiener weights [0..7]: ");
    for (int k = 0; k < 8; ++k) std::printf("%.4f ", weights[k]);
    std::printf("...\n");

    return 0;
}
