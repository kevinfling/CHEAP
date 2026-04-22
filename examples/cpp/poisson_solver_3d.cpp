/*
 * Poisson solver 3D — C++ wrapper version.
 */

#include "cheap.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

int main()
{
    const int nx = 32, ny = 32, nz = 32, n = nx * ny * nz;

    cheap::Context3D ctx(nx, ny, nz, 0.5, 0.5, 0.5);

    std::vector<double> f(n, 0.0), phi(n);
    std::vector<double> w = cheap::weights_laplacian_3d(nx, ny, nz);
    w[0] = 1.0;

    f[(nx / 2 * ny + ny / 2) * nz + nz / 2] = 1.0;

    ctx.forward(f.data());
    for (int i = 0; i < n; ++i) ctx.workspace()[i] /= w[i];
    ctx.inverse(phi.data());

    double phi_max = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(phi[i])) {
            std::fprintf(stderr, "non-finite phi at %d\n", i);
            return 1;
        }
        if (std::abs(phi[i]) > phi_max) phi_max = std::abs(phi[i]);
    }

    std::printf("poisson_solver_3d_cpp: phi_max = %.6f\n", phi_max);
    return 0;
}
