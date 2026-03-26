/*
 * test_cheap_cpp.cpp — unit tests for the CHEAP C++ wrapper.
 *
 * Build (standalone):
 *   g++ -std=c++17 -Wall -Wextra -Werror -march=native -O2 \
 *       test_cheap_cpp.cpp -o test_cheap_cpp -lfftw3 -lm
 */

#include "cheap.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* =========================================================================
 * Minimal test framework
 * ========================================================================= */

static int g_tests_run    = 0;
static int g_tests_failed = 0;

#define ASSERT_TRUE(expr) do { \
    ++g_tests_run; \
    if (!(expr)) { \
        ++g_tests_failed; \
        std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #expr); \
    } \
} while (0)

#define ASSERT_EQ(a, b) do { \
    ++g_tests_run; \
    if ((a) != (b)) { \
        ++g_tests_failed; \
        std::fprintf(stderr, "FAIL %s:%d  %s == %s\n", __FILE__, __LINE__, #a, #b); \
    } \
} while (0)

#define ASSERT_NEAR(a, b, tol) do { \
    ++g_tests_run; \
    if (std::fabs((a) - (b)) > (tol)) { \
        ++g_tests_failed; \
        std::fprintf(stderr, "FAIL %s:%d  |%s - %s| = %.3e > %.3e\n", \
                     __FILE__, __LINE__, #a, #b, std::fabs((a)-(b)), (tol)); \
    } \
} while (0)

#define ASSERT_THROWS(expr, etype) do { \
    ++g_tests_run; \
    bool caught = false; \
    try { expr; } catch (const etype&) { caught = true; } \
    if (!caught) { \
        ++g_tests_failed; \
        std::fprintf(stderr, "FAIL %s:%d  expected exception from: %s\n", \
                     __FILE__, __LINE__, #expr); \
    } \
} while (0)

#define ASSERT_THROWS_CODE(expr, expected_code) do { \
    ++g_tests_run; \
    bool caught = false; \
    try { expr; } catch (const cheap::Error& e) { \
        caught = true; \
        if (e.code() != (expected_code)) { \
            ++g_tests_failed; \
            std::fprintf(stderr, "FAIL %s:%d  expected code %d, got %d\n", \
                         __FILE__, __LINE__, static_cast<int>(expected_code), \
                         static_cast<int>(e.code())); \
        } \
    } \
    if (!caught) { \
        ++g_tests_failed; \
        std::fprintf(stderr, "FAIL %s:%d  expected cheap::Error from: %s\n", \
                     __FILE__, __LINE__, #expr); \
    } \
} while (0)

#define RUN_TEST(fn) do { \
    std::fprintf(stderr, "  %-48s", #fn); \
    int before = g_tests_failed; \
    fn(); \
    std::fprintf(stderr, "%s\n", (g_tests_failed == before) ? "OK" : "FAILED"); \
} while (0)

/* =========================================================================
 * Tests
 * ========================================================================= */

static void test_context_lifecycle() {
    cheap::Context ctx(128, 0.7);
    ASSERT_EQ(ctx.n(), 128);
    ASSERT_NEAR(ctx.H(), 0.7, 1e-15);
    ASSERT_TRUE(ctx.lambda() != nullptr);
    ASSERT_TRUE(ctx.sqrt_lambda() != nullptr);
    ASSERT_TRUE(ctx.workspace() != nullptr);
    ASSERT_TRUE(ctx.ctx() != nullptr);
    ASSERT_TRUE(ctx.ctx()->is_initialized == 1);
}

static void test_context_invalid_params() {
    ASSERT_THROWS_CODE(cheap::Context(1, 0.5), cheap::ErrorCode::einval);
    ASSERT_THROWS_CODE(cheap::Context(128, 0.0), cheap::ErrorCode::einval);
    ASSERT_THROWS_CODE(cheap::Context(128, 1.0), cheap::ErrorCode::einval);
    ASSERT_THROWS_CODE(cheap::Context(128, -0.1), cheap::ErrorCode::einval);
}

static void test_move_semantics() {
    cheap::Context a(64, 0.5);
    const double* a_lambda = a.lambda();
    ASSERT_TRUE(a_lambda != nullptr);

    /* Move construct */
    cheap::Context b(std::move(a));
    ASSERT_EQ(b.n(), 64);
    ASSERT_TRUE(b.lambda() == a_lambda); /* same pointer transferred */
    ASSERT_TRUE(a.ctx()->is_initialized == 0); /* moved-from is zeroed */
    ASSERT_TRUE(a.lambda() == nullptr);

    /* Move assign */
    cheap::Context c(32, 0.3);
    c = std::move(b);
    ASSERT_EQ(c.n(), 64);
    ASSERT_TRUE(c.lambda() == a_lambda);
    ASSERT_TRUE(b.ctx()->is_initialized == 0);
}

static void test_forward_inverse_roundtrip() {
    const int n = 128;
    cheap::Context ctx(n, 0.7);

    std::vector<double> input(n);
    for (int i = 0; i < n; ++i)
        input[i] = std::sin(2.0 * M_PI * i / n) + 1.0;

    /* Apply with unit weights should recover input */
    std::vector<double> ones(n, 1.0);
    std::vector<double> output(n);
    ctx.apply(input.data(), ones.data(), output.data());

    for (int i = 0; i < n; ++i)
        ASSERT_NEAR(output[i], input[i], 1e-10);
}

static void test_apply_convenience() {
    const int n = 64;
    cheap::Context ctx(n, 0.5);

    std::vector<double> input(n);
    std::vector<double> weights(n, 1.0);
    for (int i = 0; i < n; ++i) input[i] = static_cast<double>(i);

    auto result = ctx.apply(input.data(), weights.data());
    ASSERT_EQ(static_cast<int>(result.size()), n);
    for (int i = 0; i < n; ++i)
        ASSERT_NEAR(result[i], input[i], 1e-10);
}

static void test_try_methods() {
    const int n = 64;
    cheap::Context ctx(n, 0.5);

    int rc = ctx.try_forward(nullptr);
    ASSERT_EQ(rc, CHEAP_EINVAL);

    rc = ctx.try_inverse(nullptr);
    ASSERT_EQ(rc, CHEAP_EINVAL);

    rc = ctx.try_apply(nullptr, nullptr, nullptr);
    ASSERT_EQ(rc, CHEAP_EINVAL);

    rc = ctx.try_recompute_gibbs(-1.0);
    ASSERT_EQ(rc, CHEAP_EINVAL);
}

static void test_error_code_mapping() {
    try {
        cheap::Context(1, 0.5);
        ASSERT_TRUE(false); /* should not reach */
    } catch (const cheap::Error& e) {
        ASSERT_EQ(e.code(), cheap::ErrorCode::einval);
        ASSERT_TRUE(std::string(e.what()).find("invalid argument") != std::string::npos);
    }
}

static void test_sinkhorn() {
    const int n = 64;
    cheap::Context ctx(n, 0.7);

    std::vector<double> a(n, 1.0 / n);
    std::vector<double> b(n, 1.0 / n);
    std::vector<double> f(n, 0.0);
    std::vector<double> g(n, 0.0);

    int rc = ctx.try_sinkhorn(a.data(), b.data(), 0.5, 100, 1e-6, f.data(), g.data());
    ASSERT_EQ(rc, CHEAP_OK);
}

static void test_toeplitz_solve() {
    const int n = 64;
    cheap::Context ctx(n, 0.5);

    /* Tridiagonal: T = [2, -1, 0, ...] */
    std::vector<double> t(n, 0.0);
    t[0] = 2.0;
    t[1] = -1.0;

    auto lam = ctx.toeplitz_eigenvalues(t.data());
    ASSERT_EQ(static_cast<int>(lam.size()), n);

    /* Solve (T + 0.01*I)x = y */
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) y[i] = std::sin(2.0 * M_PI * i / n);

    auto x = ctx.toeplitz_solve_precomp(lam.data(), y.data(), 0.01);
    ASSERT_EQ(static_cast<int>(x.size()), n);

    /* Verify: compute Tx + 0.01*x and compare to y */
    std::vector<double> check(n);
    ctx.apply(x.data(), lam.data(), check.data());
    for (int i = 0; i < n; ++i)
        check[i] += 0.01 * x[i];
    for (int i = 0; i < n; ++i)
        ASSERT_NEAR(check[i], y[i], 1e-6);
}

static void test_rff_lifecycle() {
    cheap::RffContext rff(256, 3, 1.0, 42);
    ASSERT_EQ(rff.D(), 256);
    ASSERT_EQ(rff.d_in(), 3);
    ASSERT_NEAR(rff.sigma(), 1.0, 1e-15);
    ASSERT_TRUE(rff.ctx() != nullptr);
}

static void test_rff_invalid_params() {
    ASSERT_THROWS_CODE(cheap::RffContext(1, 1, 1.0, 0), cheap::ErrorCode::einval);
    ASSERT_THROWS_CODE(cheap::RffContext(3, 1, 1.0, 0), cheap::ErrorCode::einval);
    ASSERT_THROWS_CODE(cheap::RffContext(4, 0, 1.0, 0), cheap::ErrorCode::einval);
    ASSERT_THROWS_CODE(cheap::RffContext(4, 1, 0.0, 0), cheap::ErrorCode::einval);
}

static void test_rff_map() {
    cheap::RffContext rff(64, 2, 1.0, 42);
    double x_in[2] = {0.5, -0.3};

    auto z = rff.map(x_in);
    ASSERT_EQ(static_cast<int>(z.size()), 64);

    /* All outputs should be finite */
    for (int i = 0; i < 64; ++i)
        ASSERT_TRUE(std::isfinite(z[i]));
}

static void test_rff_map_batch() {
    cheap::RffContext rff(32, 1, 1.0, 42);
    const int N = 10;
    std::vector<double> X(N, 0.0);
    for (int i = 0; i < N; ++i) X[i] = static_cast<double>(i) * 0.1;

    auto Z = rff.map_batch(X.data(), N);
    ASSERT_EQ(static_cast<int>(Z.size()), N * 32);

    /* Verify batch consistency with single map */
    std::vector<double> z_single(32);
    for (int i = 0; i < N; ++i) {
        rff.map(&X[i], z_single.data());
        for (int j = 0; j < 32; ++j)
            ASSERT_NEAR(Z[i * 32 + j], z_single[j], 1e-15);
    }
}

static void test_rff_move() {
    cheap::RffContext a(64, 1, 1.0, 42);
    cheap::RffContext b(std::move(a));
    ASSERT_EQ(b.D(), 64);
    ASSERT_TRUE(a.ctx()->is_initialized == 0);
}

static void test_rdtsc() {
    /* Just verify it returns without crashing; value is platform-dependent */
    std::uint64_t t1 = cheap::rdtsc();
    std::uint64_t t2 = cheap::rdtsc();
    (void)t1; (void)t2;
    ASSERT_TRUE(true);
}

#ifdef CHEAP_HAS_SPAN
static void test_span_forward_inverse() {
    const int n = 64;
    cheap::Context ctx(n, 0.5);

    std::vector<double> input(n);
    for (int i = 0; i < n; ++i) input[i] = static_cast<double>(i);

    std::vector<double> ones(n, 1.0);
    std::vector<double> output(n);

    ctx.apply(std::span<const double>(input), std::span<const double>(ones),
              std::span<double>(output));
    for (int i = 0; i < n; ++i)
        ASSERT_NEAR(output[i], input[i], 1e-10);
}

static void test_span_size_mismatch() {
    cheap::Context ctx(64, 0.5);
    std::vector<double> wrong_size(32);
    ASSERT_THROWS(ctx.forward(std::span<const double>(wrong_size)), cheap::Error);
    ASSERT_THROWS(ctx.inverse(std::span<double>(wrong_size)), cheap::Error);
}
#endif

/* =========================================================================
 * Section: Spectral weight constructor tests
 * ========================================================================= */

static void test_weights_fractional_cpp() {
    const int n = 64;
    auto w = cheap::weights_fractional(n, 0.0);
    ASSERT_EQ(static_cast<int>(w.size()), n);
    for (int k = 0; k < n; ++k) ASSERT_NEAR(w[k], 1.0, 1e-12);

    /* Roundtrip */
    auto wd = cheap::weights_fractional(n, 0.4);
    auto wi = cheap::weights_fractional(n, -0.4);
    for (int k = 1; k < n; ++k) ASSERT_NEAR(wd[k] * wi[k], 1.0, 1e-12);

    /* Invalid params throw */
    ASSERT_THROWS(cheap::weights_fractional(1, 0.5), cheap::Error);
}

static void test_weights_wiener_cpp() {
    const int n = 128;
    auto w = cheap::weights_wiener(n, 1.0);
    ASSERT_EQ(static_cast<int>(w.size()), n);
    ASSERT_NEAR(w[0], 0.0, 1e-16);
    for (int k = 1; k < n; ++k) {
        ASSERT_TRUE(w[k] >= 0.0);
        ASSERT_TRUE(w[k] < 1.0);
    }

    /* _ev matches simple */
    auto lap = cheap::weights_laplacian(n);
    auto w_ev = cheap::weights_wiener_ev(n, lap.data(), 1.0);
    for (int k = 0; k < n; ++k) ASSERT_NEAR(w[k], w_ev[k], 1e-14);
}

static void test_weights_specnorm_cpp() {
    const int n = 64;
    double eps = 1e-3;
    auto w = cheap::weights_specnorm(n, eps);
    ASSERT_EQ(static_cast<int>(w.size()), n);
    ASSERT_NEAR(w[0], 1.0 / std::sqrt(eps), 1e-10);
    for (int k = 0; k < n; ++k) ASSERT_TRUE(w[k] > 0.0);

    ASSERT_THROWS(cheap::weights_specnorm(n, 0.0), cheap::Error);
}

static void test_weights_kpca_soft_cpp() {
    cheap::Context ctx(64, 0.7);
    auto w = ctx.weights_kpca_soft(16);
    ASSERT_EQ(static_cast<int>(w.size()), 64);
    for (int k = 0; k < 64; ++k) {
        ASSERT_TRUE(w[k] >= -1e-15);
        ASSERT_TRUE(w[k] <= 1.0 + 1e-15);
    }
    ASSERT_NEAR(w[16], 0.0, 1e-12);
}

static void test_weights_mandelbrot_cpp() {
    const int n = 64;
    auto w = cheap::weights_mandelbrot(n, 0.5);
    ASSERT_EQ(static_cast<int>(w.size()), n);
    for (int k = 0; k < n; ++k) ASSERT_NEAR(w[k], 1.0, 1e-12);

    ASSERT_THROWS(cheap::weights_mandelbrot(n, 0.0), cheap::Error);
    ASSERT_THROWS(cheap::weights_mandelbrot(n, 1.0), cheap::Error);
}

static void test_weights_rmt_cpp() {
    const int n = 16;
    double sigma_sq = 1.0, c = 0.5;
    double lp = sigma_sq * (1.0 + std::sqrt(c)) * (1.0 + std::sqrt(c));

    std::vector<double> lam(n);
    for (int k = 0; k < n / 2; ++k) lam[k] = lp * 0.3;  /* below */
    for (int k = n / 2; k < n; ++k) lam[k] = lp * 2.0;   /* above */

    auto wh = cheap::weights_rmt_hard(lam.data(), n, sigma_sq, c);
    ASSERT_EQ(static_cast<int>(wh.size()), n);
    for (int k = 0; k < n / 2; ++k) ASSERT_NEAR(wh[k], 0.0, 1e-16);
    for (int k = n / 2; k < n; ++k) ASSERT_TRUE(wh[k] > 0.0);

    auto ws = cheap::weights_rmt_shrink(lam.data(), n, sigma_sq, c);
    for (int k = 0; k < n / 2; ++k) ASSERT_NEAR(ws[k], 0.0, 1e-16);
}

/* =========================================================================
 * main
 * ========================================================================= */

int main() {
    std::fprintf(stderr, "CHEAP C++ wrapper tests\n");
    std::fprintf(stderr, "=======================\n");

    RUN_TEST(test_context_lifecycle);
    RUN_TEST(test_context_invalid_params);
    RUN_TEST(test_move_semantics);
    RUN_TEST(test_forward_inverse_roundtrip);
    RUN_TEST(test_apply_convenience);
    RUN_TEST(test_try_methods);
    RUN_TEST(test_error_code_mapping);
    RUN_TEST(test_sinkhorn);
    RUN_TEST(test_toeplitz_solve);
    RUN_TEST(test_rff_lifecycle);
    RUN_TEST(test_rff_invalid_params);
    RUN_TEST(test_rff_map);
    RUN_TEST(test_rff_map_batch);
    RUN_TEST(test_rff_move);
    RUN_TEST(test_rdtsc);

#ifdef CHEAP_HAS_SPAN
    std::fprintf(stderr, "\n  --- span overloads (C++20) ---\n");
    RUN_TEST(test_span_forward_inverse);
    RUN_TEST(test_span_size_mismatch);
#endif

    std::fprintf(stderr, "\n  --- spectral weight constructors ---\n");
    RUN_TEST(test_weights_fractional_cpp);
    RUN_TEST(test_weights_wiener_cpp);
    RUN_TEST(test_weights_specnorm_cpp);
    RUN_TEST(test_weights_kpca_soft_cpp);
    RUN_TEST(test_weights_mandelbrot_cpp);
    RUN_TEST(test_weights_rmt_cpp);

    std::fprintf(stderr, "\n%d tests, %d failed\n", g_tests_run, g_tests_failed);
    return g_tests_failed ? 1 : 0;
}
