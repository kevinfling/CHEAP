/*
 * Poisson solver 2D — C++ wrapper version.
 */

#include "cheap.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main()
{
    const int nx = 128, ny = 128, n = nx * ny;

    cheap::Context2D ctx(nx, ny, 0.5, 0.5);

    std::vector<double> f(n), phi(n);

    for (int j = 0; j < nx; ++j) {
        for (int k = 0; k < ny; ++k) {
            double x = static_cast<double>(j) / (nx - 1) - 0.5;
            double y = static_cast<double>(k) / (ny - 1) - 0.5;
            f[j * ny + k] = std::exp(-100.0 * (x * x + y * y));
        }
    }

    double f_mean = 0.0;
    for (int i = 0; i < n; ++i) f_mean += f[i];
    f_mean /= static_cast<double>(n);
    for (int i = 0; i < n; ++i) f[i] -= f_mean;

    std::vector<double> w = cheap::weights_laplacian_2d(nx, ny);
    w[0] = 1.0;

    ctx.forward(f.data());
    for (int i = 0; i < n; ++i) ctx.workspace()[i] /= w[i];
    ctx.inverse(phi.data());

    double max_res = 0.0;
    for (int j = 1; j < nx - 1; ++j) {
        for (int k = 1; k < ny - 1; ++k) {
            int idx = j * ny + k;
            double lap = 4.0 * phi[idx]
                       - phi[(j - 1) * ny + k] - phi[(j + 1) * ny + k]
                       - phi[j * ny + (k - 1)] - phi[j * ny + (k + 1)];
            double res = std::abs(lap - f[idx]);
            if (res > max_res) max_res = res;
        }
    }

    std::printf("poisson_solver_2d_cpp: max residual = %.3e\n", max_res);
    return (max_res < 1e-10) ? 0 : 1;
}
