/*
 * Heat / fractional dissipation 3D — C++ wrapper version.
 */

#include "cheap.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

int main()
{
    const int nx = 32, ny = 32, nz = 32, n = nx * ny * nz;
    const double nu = 0.05, alpha = 1.0, dt = 0.01;
    const int nsteps = 100;

    cheap::Context3D ctx(nx, ny, nz, 0.5, 0.5, 0.5);

    std::vector<double> u(n);
    for (int j = 0; j < nx; ++j) {
        for (int k = 0; k < ny; ++k) {
            for (int l = 0; l < nz; ++l) {
                double x = static_cast<double>(j) / nx - 0.5;
                double y = static_cast<double>(k) / ny - 0.5;
                double z = static_cast<double>(l) / nz - 0.5;
                u[(j * ny + k) * nz + l] = std::exp(-50.0 * (x * x + y * y + z * z));
            }
        }
    }

    auto w = cheap::weights_fractional_laplacian_3d(nx, ny, nz, alpha);
    for (int i = 0; i < n; ++i) w[i] = std::exp(-nu * dt * w[i]);

    double ke0 = 0.0;
    for (int i = 0; i < n; ++i) ke0 += u[i] * u[i];
    ke0 /= static_cast<double>(n);

    for (int step = 0; step < nsteps; ++step) {
        u = ctx.apply(u.data(), w.data());
    }

    double ke_final = 0.0;
    for (int i = 0; i < n; ++i) ke_final += u[i] * u[i];
    ke_final /= static_cast<double>(n);

    std::printf("ns_dissipation_3d_cpp: KE0 = %.6e, KE_final = %.6e, ratio = %.6f\n",
                ke0, ke_final, ke_final / ke0);

    return (ke_final < ke0) ? 0 : 1;
}
