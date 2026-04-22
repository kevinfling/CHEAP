/*
 * GP regression 2D — C++ wrapper version.
 */

#include "cheap.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main()
{
    const int nx = 64, ny = 64, n = nx * ny;
    const double noise_var = 0.01;

    cheap::Context2D ctx(nx, ny, 0.5, 0.5);

    std::vector<double> f_clean(n), y_obs(n), f_post(n);

    for (int j = 0; j < nx; ++j) {
        for (int k = 0; k < ny; ++k) {
            double x = static_cast<double>(j) / nx;
            double y = static_cast<double>(k) / ny;
            f_clean[j * ny + k] = std::sin(2.0 * M_PI * x) * std::cos(2.0 * M_PI * y)
                                  + 0.5 * std::sin(4.0 * M_PI * x);
        }
    }

    std::srand(42);
    for (int i = 0; i < n; ++i) {
        double u1 = static_cast<double>(std::rand()) / RAND_MAX;
        double u2 = static_cast<double>(std::rand()) / RAND_MAX;
        if (u1 < 1e-300) u1 = 1e-300;
        double r = std::sqrt(-2.0 * std::log(u1));
        double theta = 2.0 * M_PI * u2;
        double noise = std::sqrt(noise_var) * r * std::cos(theta);
        y_obs[i] = f_clean[i] + noise;
    }

    auto w = cheap::weights_laplacian_2d(nx, ny);
    for (int i = 0; i < n; ++i) {
        double inv_lam = 1.0 / std::max(w[i], 1e-12);
        w[i] = inv_lam / (inv_lam + noise_var);
    }
    f_post = ctx.apply(y_obs.data(), w.data());

    double rmse = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = f_post[i] - f_clean[i];
        rmse += d * d;
    }
    rmse = std::sqrt(rmse / static_cast<double>(n));

    std::printf("gp_regression_2d_cpp: RMSE = %.3e  (noise sigma = %.3e)\n",
                rmse, std::sqrt(noise_var));

    return (rmse < std::sqrt(noise_var)) ? 0 : 1;
}
