/*
 * bench_cheap_cpp.cpp — micro-benchmarks for the CHEAP C++ wrapper.
 *
 * Mirrors bench_cheap.c exactly so output can be directly compared.
 * Algorithm names have a "_cpp" suffix.
 *
 * Build (standalone):
 *   g++ -std=c++17 -Wall -Wextra -Werror -march=native -O3 \
 *       -D_POSIX_C_SOURCE=199309L \
 *       bench_cheap_cpp.cpp -o bench_cheap_cpp -lfftw3 -lm
 */

#include "cheap.hpp"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* =========================================================================
 * Portable wall clock
 * ========================================================================= */

static double wall_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

static constexpr int WARMUP_ITERS = 3;
static constexpr int BENCH_ITERS  = 10;

template <typename F>
static void run_bench(F&& fn, double& out_wall_ms, uint64_t& out_ticks) {
    for (int i = 0; i < WARMUP_ITERS; ++i) fn();
    double t0 = wall_seconds();
    uint64_t c0 = cheap::rdtsc();
    for (int i = 0; i < BENCH_ITERS; ++i) fn();
    uint64_t c1 = cheap::rdtsc();
    double t1 = wall_seconds();
    out_wall_ms = (t1 - t0) * 1e3 / BENCH_ITERS;
    out_ticks   = (c1 - c0) / static_cast<uint64_t>(BENCH_ITERS);
}

static void print_result(const char* algo, int n, double wall_ms, uint64_t ticks) {
    std::printf("%-24s %6d   %10.6f   %12" PRIu64 "\n", algo, n, wall_ms, ticks);
}

/* =========================================================================
 * Core algorithm benchmarks
 * ========================================================================= */

static void run_core_benchmarks(int n) {
    const double H = 0.7;
    if (n >= 8192) std::fprintf(stderr, "  Planning FFTW for n=%d ...\n", n);

    /* cheap_apply with KRR weights */
    {
        cheap::Context ctx(n, H);
        auto* input   = static_cast<double*>(fftw_malloc(static_cast<std::size_t>(n) * sizeof(double)));
        auto* weights = static_cast<double*>(fftw_malloc(static_cast<std::size_t>(n) * sizeof(double)));
        auto* output  = static_cast<double*>(fftw_malloc(static_cast<std::size_t>(n) * sizeof(double)));
        for (int i = 0; i < n; ++i) input[i] = std::sin(2.0 * M_PI * i / n) + 1.0;
        for (int k = 0; k < n; ++k) weights[k] = 1.0 / (ctx.lambda()[k] + 1e-3);

        double wms; uint64_t tk;
        run_bench([&]{ ctx.apply(input, weights, output); }, wms, tk);
        print_result("apply_krr_cpp", n, wms, tk);

        fftw_free(input); fftw_free(weights); fftw_free(output);
    }

    /* cheap_apply with sqrt_lambda (reparam) */
    {
        cheap::Context ctx(n, H);
        auto* input  = static_cast<double*>(fftw_malloc(static_cast<std::size_t>(n) * sizeof(double)));
        auto* output = static_cast<double*>(fftw_malloc(static_cast<std::size_t>(n) * sizeof(double)));
        for (int i = 0; i < n; ++i) input[i] = 1.0;

        double wms; uint64_t tk;
        run_bench([&]{ ctx.apply(input, ctx.sqrt_lambda(), output); }, wms, tk);
        print_result("apply_reparam_cpp", n, wms, tk);

        fftw_free(input); fftw_free(output);
    }

    /* forward */
    {
        cheap::Context ctx(n, H);
        auto* input = static_cast<double*>(fftw_malloc(static_cast<std::size_t>(n) * sizeof(double)));
        for (int i = 0; i < n; ++i) input[i] = std::sin(2.0 * M_PI * i / n);

        double wms; uint64_t tk;
        run_bench([&]{ ctx.forward(input); }, wms, tk);
        print_result("forward_cpp", n, wms, tk);

        fftw_free(input);
    }

    /* inverse */
    {
        cheap::Context ctx(n, H);
        auto* input  = static_cast<double*>(fftw_malloc(static_cast<std::size_t>(n) * sizeof(double)));
        auto* output = static_cast<double*>(fftw_malloc(static_cast<std::size_t>(n) * sizeof(double)));
        for (int i = 0; i < n; ++i) input[i] = std::sin(2.0 * M_PI * i / n);
        ctx.forward(input);

        double wms; uint64_t tk;
        run_bench([&]{ ctx.inverse(output); }, wms, tk);
        print_result("inverse_cpp", n, wms, tk);

        fftw_free(input); fftw_free(output);
    }

    /* Sinkhorn */
    {
        cheap::Context ctx(n, H);
        auto* a = static_cast<double*>(fftw_malloc(static_cast<std::size_t>(n) * sizeof(double)));
        auto* b = static_cast<double*>(fftw_malloc(static_cast<std::size_t>(n) * sizeof(double)));
        auto* f = static_cast<double*>(fftw_malloc(static_cast<std::size_t>(n) * sizeof(double)));
        auto* g = static_cast<double*>(fftw_malloc(static_cast<std::size_t>(n) * sizeof(double)));
        for (int i = 0; i < n; ++i) a[i] = b[i] = 1.0 / static_cast<double>(n);

        double wms; uint64_t tk;
        run_bench([&]{ ctx.try_sinkhorn(a, b, 0.5, 50, 1e-15, f, g); }, wms, tk);
        print_result("sinkhorn_50_cpp", n, wms, tk);

        fftw_free(a); fftw_free(b); fftw_free(f); fftw_free(g);
    }
}

/* =========================================================================
 * Toeplitz benchmarks
 * ========================================================================= */

static void run_toeplitz_benchmarks(int n) {
    if (n >= 8192) std::fprintf(stderr, "  Planning FFTW for n=%d ...\n", n);

    cheap::Context ctx(n, 0.5);

    std::vector<double> t(n, 0.0);
    t[0] = 4.0; t[1] = -1.0;

    auto* lam = static_cast<double*>(fftw_malloc(static_cast<std::size_t>(n) * sizeof(double)));
    auto* x   = static_cast<double*>(fftw_malloc(static_cast<std::size_t>(n) * sizeof(double)));
    auto* y   = static_cast<double*>(fftw_malloc(static_cast<std::size_t>(n) * sizeof(double)));
    for (int i = 0; i < n; ++i) x[i] = std::sin(2.0 * M_PI * i / n) + 1.0;
    ctx.toeplitz_eigenvalues(t.data(), lam);

    double wms; uint64_t tk;

    run_bench([&]{ ctx.apply(x, lam, y); }, wms, tk);
    print_result("toeplitz_matvec_pre_cpp", n, wms, tk);

    run_bench([&]{ ctx.toeplitz_solve_precomp(lam, x, 1e-3, y); }, wms, tk);
    print_result("toeplitz_solve_pre_cpp", n, wms, tk);

    fftw_free(lam); fftw_free(x); fftw_free(y);
}

/* =========================================================================
 * RFF benchmarks
 * ========================================================================= */

static void run_rff_benchmarks() {
    int D_vals[] = {64, 256, 1024};
    double wms; uint64_t tk;

    for (int di = 0; di < 3; ++di) {
        int D = D_vals[di];
        cheap::RffContext rff(D, 1, 1.0, 42);
        double x_in = 0.5;
        std::vector<double> z_out(D);

        run_bench([&]{ rff.map(&x_in, z_out.data()); }, wms, tk);
        print_result("rff_map_cpp", D, wms, tk);
    }

    int N_vals[] = {1024, 8192};
    for (int ni = 0; ni < 2; ++ni) {
        int N = N_vals[ni];
        cheap::RffContext rff(256, 1, 1.0, 42);
        std::vector<double> X_in(N);
        std::vector<double> Z_out(static_cast<std::size_t>(N) * 256);
        for (int i = 0; i < N; ++i) X_in[i] = static_cast<double>(i) * 0.001;

        run_bench([&]{ rff.map_batch(X_in.data(), N, Z_out.data()); }, wms, tk);
        print_result("rff_map_batch_256_cpp", N, wms, tk);
    }
}

/* =========================================================================
 * Spectral weight constructor benchmarks
 * ========================================================================= */

static void run_weight_benchmarks(int n)
{
    double wms; uint64_t tk;
    std::vector<double> w(static_cast<std::size_t>(n));

    run_bench([&]{ cheap::weights_fractional(n, 0.4); }, wms, tk);
    print_result("wt_fractional_cpp", n, wms, tk);

    run_bench([&]{ cheap::weights_wiener(n, 1.0); }, wms, tk);
    print_result("wt_wiener_cpp", n, wms, tk);

    run_bench([&]{ cheap::weights_specnorm(n, 1e-3); }, wms, tk);
    print_result("wt_specnorm_cpp", n, wms, tk);

    run_bench([&]{ cheap::weights_mandelbrot(n, 0.7); }, wms, tk);
    print_result("wt_mandelbrot_cpp", n, wms, tk);
}

/* =========================================================================
 * main
 * ========================================================================= */

int main() {
    static const int sizes[] = {1024, 8192, 65536};
    static const int nsizes  = 3;

    std::printf("# %-22s %6s   %10s   %12s\n", "algo", "N", "wall_ms", "ticks");

    for (int i = 0; i < nsizes; ++i)
        run_core_benchmarks(sizes[i]);

    for (int i = 0; i < nsizes; ++i)
        run_toeplitz_benchmarks(sizes[i]);

    run_rff_benchmarks();

    for (int i = 0; i < nsizes; ++i)
        run_weight_benchmarks(sizes[i]);

    return 0;
}
