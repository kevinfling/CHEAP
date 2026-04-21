#ifndef CHEAP_HPP
#define CHEAP_HPP

/*
 * CHEAP C++ — Thin RAII wrapper for the CHEAP C99 library.
 * Header-only C++17. Requires cheap.h and FFTW3.
 *
 * Version is defined in cheap.h as CHEAP_VERSION.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Kevin Fling
 */

/* cheap.h uses C99 `restrict`; map it to the compiler extension in C++ */
#ifndef restrict
#define restrict __restrict__
#define CHEAP_HPP_UNDEF_RESTRICT
#endif

extern "C" {
#include "cheap.h"
}

#ifdef CHEAP_HPP_UNDEF_RESTRICT
#undef restrict
#undef CHEAP_HPP_UNDEF_RESTRICT
#endif

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if __cplusplus >= 202002L && !defined(CHEAP_NO_SPAN)
#include <span>
#define CHEAP_HAS_SPAN 1
#endif

namespace cheap {

/* ============================================================
 * Error handling
 * ============================================================ */

enum class ErrorCode : int {
    einval  = CHEAP_EINVAL,
    enomem  = CHEAP_ENOMEM,
    enoconv = CHEAP_ENOCONV,
    edom    = CHEAP_EDOM,
    euninit = CHEAP_EUNINIT,
};

class Error : public std::runtime_error {
public:
    explicit Error(int code)
        : std::runtime_error(make_message(code))
        , code_(static_cast<ErrorCode>(code)) {}

    ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;

    static std::string make_message(int code) {
        switch (code) {
            case CHEAP_EINVAL:  return "cheap: invalid argument";
            case CHEAP_ENOMEM:  return "cheap: memory allocation failed";
            case CHEAP_ENOCONV: return "cheap: Sinkhorn did not converge";
            case CHEAP_EDOM:    return "cheap: NaN/Inf in input data";
            case CHEAP_EUNINIT: return "cheap: context not initialized";
            default:              return "cheap: unknown error (" + std::to_string(code) + ")";
        }
    }
};

namespace detail {
inline void check(int rc) {
    if (rc != CHEAP_OK) throw Error(rc);
}
} // namespace detail

/* ============================================================
 * Context — RAII wrapper for cheap_ctx
 * ============================================================ */

class Context {
public:
    Context(int n, double H) {
        detail::check(cheap_init(&ctx_, n, H));
    }

    ~Context() noexcept {
        cheap_destroy(&ctx_);
    }

    Context(Context&& other) noexcept : ctx_(other.ctx_) {
        std::memset(&other.ctx_, 0, sizeof(cheap_ctx));
    }

    Context& operator=(Context&& other) noexcept {
        if (this != &other) {
            cheap_destroy(&ctx_);
            ctx_ = other.ctx_;
            std::memset(&other.ctx_, 0, sizeof(cheap_ctx));
        }
        return *this;
    }

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    /* --- Core spectral primitives --- */

    void forward(const double* input) { detail::check(try_forward(input)); }
    int try_forward(const double* input) noexcept { return cheap_forward(&ctx_, input); }

    void forward_inplace() { detail::check(try_forward_inplace()); }
    int try_forward_inplace() noexcept { return cheap_forward_inplace(&ctx_); }

    void inverse(double* output) { detail::check(try_inverse(output)); }
    int try_inverse(double* output) noexcept { return cheap_inverse(&ctx_, output); }

    void inverse_inplace() { detail::check(try_inverse_inplace()); }
    int try_inverse_inplace() noexcept { return cheap_inverse_inplace(&ctx_); }

    void apply(const double* input, const double* weights, double* output) {
        detail::check(try_apply(input, weights, output));
    }
    int try_apply(const double* input, const double* weights, double* output) noexcept {
        return cheap_apply(&ctx_, input, weights, output);
    }

    std::vector<double> apply(const double* input, const double* weights) {
        std::vector<double> out(static_cast<std::size_t>(ctx_.n));
        apply(input, weights, out.data());
        return out;
    }

    /* --- Sinkhorn optimal transport --- */

    void recompute_gibbs(double eps) { detail::check(try_recompute_gibbs(eps)); }
    int try_recompute_gibbs(double eps) noexcept { return cheap_recompute_gibbs(&ctx_, eps); }

    void apply_hybrid_log(const double* f, double* out) noexcept {
        cheap_apply_hybrid_log(&ctx_, f, out);
    }

    int sinkhorn(const double* a, const double* b,
                 double eps, int max_iter, double tol,
                 double* f, double* g) {
        int rc = try_sinkhorn(a, b, eps, max_iter, tol, f, g);
        detail::check(rc);
        return rc;
    }
    int try_sinkhorn(const double* a, const double* b,
                     double eps, int max_iter, double tol,
                     double* f, double* g) noexcept {
        return cheap_sinkhorn(&ctx_, a, b, eps, max_iter, tol, f, g);
    }

    /* --- Toeplitz utilities --- */

    void toeplitz_eigenvalues(const double* t, double* lambda_out) {
        detail::check(try_toeplitz_eigenvalues(t, lambda_out));
    }
    int try_toeplitz_eigenvalues(const double* t, double* lambda_out) noexcept {
        return cheap_toeplitz_eigenvalues(&ctx_, t, lambda_out);
    }

    std::vector<double> toeplitz_eigenvalues(const double* t) {
        std::vector<double> out(static_cast<std::size_t>(ctx_.n));
        toeplitz_eigenvalues(t, out.data());
        return out;
    }

    void toeplitz_solve_precomp(const double* lambda_t, const double* y,
                                double lambda_reg, double* x) {
        detail::check(try_toeplitz_solve_precomp(lambda_t, y, lambda_reg, x));
    }
    int try_toeplitz_solve_precomp(const double* lambda_t, const double* y,
                                   double lambda_reg, double* x) noexcept {
        return cheap_toeplitz_solve_precomp(&ctx_, lambda_t, y, lambda_reg, x);
    }

    std::vector<double> toeplitz_solve_precomp(const double* lambda_t,
                                               const double* y, double lambda_reg) {
        std::vector<double> out(static_cast<std::size_t>(ctx_.n));
        toeplitz_solve_precomp(lambda_t, y, lambda_reg, out.data());
        return out;
    }

    /* --- Span overloads (C++20) --- */

#ifdef CHEAP_HAS_SPAN
    void forward(std::span<const double> input) {
        if (static_cast<int>(input.size()) != ctx_.n) throw Error(CHEAP_EINVAL);
        forward(input.data());
    }

    void inverse(std::span<double> output) {
        if (static_cast<int>(output.size()) != ctx_.n) throw Error(CHEAP_EINVAL);
        inverse(output.data());
    }

    void apply(std::span<const double> input, std::span<const double> weights,
               std::span<double> output) {
        if (static_cast<int>(input.size()) != ctx_.n ||
            static_cast<int>(weights.size()) != ctx_.n ||
            static_cast<int>(output.size()) != ctx_.n)
            throw Error(CHEAP_EINVAL);
        apply(input.data(), weights.data(), output.data());
    }

    std::vector<double> apply(std::span<const double> input,
                              std::span<const double> weights) {
        if (static_cast<int>(input.size()) != ctx_.n ||
            static_cast<int>(weights.size()) != ctx_.n)
            throw Error(CHEAP_EINVAL);
        return apply(input.data(), weights.data());
    }

    int sinkhorn(std::span<const double> a, std::span<const double> b,
                 double eps, int max_iter, double tol,
                 std::span<double> f, std::span<double> g) {
        if (static_cast<int>(a.size()) != ctx_.n ||
            static_cast<int>(b.size()) != ctx_.n ||
            static_cast<int>(f.size()) != ctx_.n ||
            static_cast<int>(g.size()) != ctx_.n)
            throw Error(CHEAP_EINVAL);
        return sinkhorn(a.data(), b.data(), eps, max_iter, tol, f.data(), g.data());
    }

    void toeplitz_eigenvalues(std::span<const double> t, std::span<double> lambda_out) {
        if (static_cast<int>(t.size()) != ctx_.n ||
            static_cast<int>(lambda_out.size()) != ctx_.n)
            throw Error(CHEAP_EINVAL);
        toeplitz_eigenvalues(t.data(), lambda_out.data());
    }

    std::vector<double> toeplitz_eigenvalues(std::span<const double> t) {
        if (static_cast<int>(t.size()) != ctx_.n) throw Error(CHEAP_EINVAL);
        return toeplitz_eigenvalues(t.data());
    }

    void toeplitz_solve_precomp(std::span<const double> lambda_t,
                                std::span<const double> y,
                                double lambda_reg, std::span<double> x) {
        if (static_cast<int>(lambda_t.size()) != ctx_.n ||
            static_cast<int>(y.size()) != ctx_.n ||
            static_cast<int>(x.size()) != ctx_.n)
            throw Error(CHEAP_EINVAL);
        toeplitz_solve_precomp(lambda_t.data(), y.data(), lambda_reg, x.data());
    }
#endif

    /* --- Spectral weight constructors (ctx-dependent) --- */

    std::vector<double> weights_kpca_soft(int K) const {
        std::vector<double> w(static_cast<std::size_t>(ctx_.n));
        detail::check(cheap_weights_kpca_soft(&ctx_, K, w.data()));
        return w;
    }

    void weights_kpca_soft(int K, double* out) const {
        detail::check(cheap_weights_kpca_soft(&ctx_, K, out));
    }

    /* --- Accessors --- */

    int n() const noexcept { return ctx_.n; }
    double H() const noexcept { return ctx_.current_H; }
    double current_eps() const noexcept { return ctx_.current_eps; }

    const double* lambda() const noexcept { return ctx_.lambda; }
    const double* sqrt_lambda() const noexcept { return ctx_.sqrt_lambda; }
    const double* gibbs() const noexcept { return ctx_.gibbs; }

    double* workspace() noexcept { return ctx_.workspace; }
    const double* workspace() const noexcept { return ctx_.workspace; }

#ifdef CHEAP_HAS_SPAN
    std::span<const double> lambda_span() const noexcept {
        return {ctx_.lambda, static_cast<std::size_t>(ctx_.n)};
    }
    std::span<const double> sqrt_lambda_span() const noexcept {
        return {ctx_.sqrt_lambda, static_cast<std::size_t>(ctx_.n)};
    }
    std::span<const double> gibbs_span() const noexcept {
        return {ctx_.gibbs, static_cast<std::size_t>(ctx_.n)};
    }
    std::span<double> workspace_span() noexcept {
        return {ctx_.workspace, static_cast<std::size_t>(ctx_.n)};
    }
    std::span<const double> workspace_span() const noexcept {
        return {ctx_.workspace, static_cast<std::size_t>(ctx_.n)};
    }
#endif

    cheap_ctx* ctx() noexcept { return &ctx_; }
    const cheap_ctx* ctx() const noexcept { return &ctx_; }

private:
    cheap_ctx ctx_{};
};

/* ============================================================
 * RffContext — RAII wrapper for cheap_rff_ctx
 * ============================================================ */

class RffContext {
public:
    RffContext(int D, int d_in, double sigma, std::uint64_t seed) {
        detail::check(cheap_rff_init(&rctx_, D, d_in, sigma, seed));
    }

    ~RffContext() noexcept {
        cheap_rff_destroy(&rctx_);
    }

    RffContext(RffContext&& other) noexcept : rctx_(other.rctx_) {
        std::memset(&other.rctx_, 0, sizeof(cheap_rff_ctx));
    }

    RffContext& operator=(RffContext&& other) noexcept {
        if (this != &other) {
            cheap_rff_destroy(&rctx_);
            rctx_ = other.rctx_;
            std::memset(&other.rctx_, 0, sizeof(cheap_rff_ctx));
        }
        return *this;
    }

    RffContext(const RffContext&) = delete;
    RffContext& operator=(const RffContext&) = delete;

    /* --- Mapping --- */

    void map(const double* x_in, double* z_out) const {
        detail::check(try_map(x_in, z_out));
    }
    int try_map(const double* x_in, double* z_out) const noexcept {
        return cheap_rff_map(&rctx_, x_in, z_out);
    }

    std::vector<double> map(const double* x_in) const {
        std::vector<double> out(static_cast<std::size_t>(rctx_.D));
        map(x_in, out.data());
        return out;
    }

    void map_batch(const double* X_in, int N, double* Z_out) const {
        detail::check(try_map_batch(X_in, N, Z_out));
    }
    int try_map_batch(const double* X_in, int N, double* Z_out) const noexcept {
        return cheap_rff_map_batch(&rctx_, X_in, N, Z_out);
    }

    std::vector<double> map_batch(const double* X_in, int N) const {
        std::vector<double> out(static_cast<std::size_t>(N) * static_cast<std::size_t>(rctx_.D));
        map_batch(X_in, N, out.data());
        return out;
    }

#ifdef CHEAP_HAS_SPAN
    void map(std::span<const double> x_in, std::span<double> z_out) const {
        if (static_cast<int>(x_in.size()) != rctx_.d_in ||
            static_cast<int>(z_out.size()) != rctx_.D)
            throw Error(CHEAP_EINVAL);
        map(x_in.data(), z_out.data());
    }

    std::vector<double> map(std::span<const double> x_in) const {
        if (static_cast<int>(x_in.size()) != rctx_.d_in) throw Error(CHEAP_EINVAL);
        return map(x_in.data());
    }
#endif

    /* --- Accessors --- */

    int D() const noexcept { return rctx_.D; }
    int d_in() const noexcept { return rctx_.d_in; }
    double sigma() const noexcept { return rctx_.sigma; }

    cheap_rff_ctx* ctx() noexcept { return &rctx_; }
    const cheap_rff_ctx* ctx() const noexcept { return &rctx_; }

private:
    cheap_rff_ctx rctx_{};
};

/* ============================================================
 * Spectral weight constructors (free functions)
 * ============================================================ */

inline std::vector<double> weights_laplacian(int n) {
    std::vector<double> w(static_cast<std::size_t>(n));
    detail::check(cheap_weights_laplacian(n, w.data()));
    return w;
}

inline std::vector<double> weights_fractional(int n, double d) {
    std::vector<double> w(static_cast<std::size_t>(n));
    detail::check(cheap_weights_fractional(n, d, w.data()));
    return w;
}

inline std::vector<double> weights_kpca_hard(int n, int K) {
    std::vector<double> w(static_cast<std::size_t>(n));
    detail::check(cheap_weights_kpca_hard(n, K, w.data()));
    return w;
}

inline std::vector<double> weights_wiener(int n, double sigma_sq) {
    std::vector<double> w(static_cast<std::size_t>(n));
    detail::check(cheap_weights_wiener(n, sigma_sq, w.data()));
    return w;
}

inline std::vector<double> weights_wiener_ev(int n, const double* lambda,
                                               double sigma_sq) {
    std::vector<double> w(static_cast<std::size_t>(n));
    detail::check(cheap_weights_wiener_ev(n, lambda, sigma_sq, w.data()));
    return w;
}

inline std::vector<double> weights_specnorm(int n, double eps) {
    std::vector<double> w(static_cast<std::size_t>(n));
    detail::check(cheap_weights_specnorm(n, eps, w.data()));
    return w;
}

inline std::vector<double> weights_specnorm_ev(int n, const double* lambda,
                                                 double eps) {
    std::vector<double> w(static_cast<std::size_t>(n));
    detail::check(cheap_weights_specnorm_ev(n, lambda, eps, w.data()));
    return w;
}

inline std::vector<double> weights_mandelbrot(int n, double H) {
    std::vector<double> w(static_cast<std::size_t>(n));
    detail::check(cheap_weights_mandelbrot(n, H, w.data()));
    return w;
}

inline std::vector<double> weights_rmt_hard(const double* lambda, int n,
                                              double sigma_sq, double c) {
    std::vector<double> w(static_cast<std::size_t>(n));
    detail::check(cheap_weights_rmt_hard(lambda, n, sigma_sq, c, w.data()));
    return w;
}

inline std::vector<double> weights_rmt_shrink(const double* lambda, int n,
                                                double sigma_sq, double c) {
    std::vector<double> w(static_cast<std::size_t>(n));
    detail::check(cheap_weights_rmt_shrink(lambda, n, sigma_sq, c, w.data()));
    return w;
}

/* ============================================================
 * Free functions
 * ============================================================ */

inline std::uint64_t rdtsc() noexcept { return cheap_rdtsc(); }

} // namespace cheap

#endif /* CHEAP_HPP */
