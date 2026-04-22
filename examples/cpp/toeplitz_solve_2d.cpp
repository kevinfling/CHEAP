/*
 * Toeplitz solve 2D — C++ wrapper version.
 */

#include "cheap.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

int main()
{
    const int nx = 32, ny = 32, n = nx * ny;
    const double lambda_reg = 1e-3;

    std::vector<double> t_col(nx), t_row(ny);
    for (int i = 0; i < nx; ++i) t_col[i] = std::exp(-0.1 * i);
    for (int i = 0; i < ny; ++i) t_row[i] = std::exp(-0.05 * i);

    cheap_ctx_2d ctx;
    if (cheap_init_from_toeplitz_2d(&ctx, nx, ny, t_col.data(), t_row.data()) != CHEAP_OK) {
        std::fprintf(stderr, "init_from_toeplitz_2d failed\n");
        return 1;
    }

    std::vector<double> y(n), x(n), w(n), check(n);
    for (int i = 0; i < n; ++i) y[i] = std::sin(0.1 * i) + 0.5;

    for (int i = 0; i < n; ++i) w[i] = 1.0 / (ctx.lambda[i] + lambda_reg);
    cheap_apply_2d(&ctx, y.data(), w.data(), x.data());

    for (int i = 0; i < n; ++i) w[i] = ctx.lambda[i];
    cheap_apply_2d(&ctx, x.data(), w.data(), check.data());
    for (int i = 0; i < n; ++i) check[i] += lambda_reg * x[i];

    cheap_destroy_2d(&ctx);

    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double err = std::abs(check[i] - y[i]);
        if (err > max_err) max_err = err;
    }

    std::printf("toeplitz_solve_2d_cpp: max residual = %.3e\n", max_err);
    return (max_err < 1e-10) ? 0 : 1;
}
