/* Verified, additive spectral weight constructors for CHEAP.
 * Discrete DCT-diagonal operators; unit grid spacing unless mu is supplied.
 * MIT license. Strict C99.
 */
#ifndef CHEAP_WEIGHTS_VERIFIED_H
#define CHEAP_WEIGHTS_VERIFIED_H
#include <math.h>
#include "cheap.h"

/* mu is a nonnegative operator spectrum; amplitude is a covariance scale.
 * d is the spatial dimension, nu the conventional Matern smoothness.
 * out[k] = amplitude * (kappa^2 + mu[k])^(-(nu+d/2)).
 */
static inline int cheap_weights_matern_covariance_ev(
    int n, const double *mu, int d, double kappa, double nu,
    double amplitude, double *out)
{
    if (n < 1 || !mu || !out || d < 1 || d > 3 ||
        !isfinite(kappa) || kappa <= 0 ||
        !isfinite(nu) || nu <= 0 ||
        !isfinite(amplitude) || amplitude < 0)
        return CHEAP_EINVAL;
    const double mass = kappa * kappa;
    const double exponent = -(nu + 0.5 * (double)d);
    if (!isfinite(mass)) return CHEAP_EDOM;
    for (int k = 0; k < n; ++k) {
        if (!isfinite(mu[k]) || mu[k] < 0) return CHEAP_EDOM;
        double base = mass + mu[k];
        if (!isfinite(base) || base <= 0) return CHEAP_EDOM;
        double w = amplitude * pow(base, exponent);
        if (!isfinite(w)) return CHEAP_EDOM;
        out[k] = w;
    }
    return CHEAP_OK;
}

/* Square-root covariance multiplier: use with white noise in an
 * orthonormal DCT basis. Do not apply the covariance weights to noise.
 */
static inline int cheap_weights_matern_sampling_ev(
    int n, const double *mu, int d, double kappa, double nu,
    double amplitude, double *out)
{
    if (n < 1 || !mu || !out || d < 1 || d > 3 ||
        !isfinite(kappa) || kappa <= 0 ||
        !isfinite(nu) || nu <= 0 ||
        !isfinite(amplitude) || amplitude < 0)
        return CHEAP_EINVAL;
    const double mass = kappa * kappa;
    const double exponent = -0.5 * (nu + 0.5 * (double)d);
    if (!isfinite(mass)) return CHEAP_EDOM;
    const double scale = sqrt(amplitude);
    for (int k = 0; k < n; ++k) {
        if (!isfinite(mu[k]) || mu[k] < 0) return CHEAP_EDOM;
        double base = mass + mu[k];
        if (!isfinite(base) || base <= 0) return CHEAP_EDOM;
        double w = scale * pow(base, exponent);
        if (!isfinite(w)) return CHEAP_EDOM;
        out[k] = w;
    }
    return CHEAP_OK;
}

/* Separate positive mass from positive directional diffusion coefficients.
 * Row-major n0*n1 grid, discrete cell-centered Neumann Laplacian.
 */
static inline int cheap_weights_anisotropic_matern_2d_verified(
    int n0, int n1, double kappa, double a0, double a1,
    double nu, double amplitude, double *out)
{
    if (n0 < 2 || n1 < 2 || !out ||
        !isfinite(kappa) || kappa <= 0 ||
        !isfinite(a0) || a0 <= 0 ||
        !isfinite(a1) || a1 <= 0 ||
        !isfinite(nu) || nu <= 0 ||
        !isfinite(amplitude) || amplitude < 0)
        return CHEAP_EINVAL;
    const double mass = kappa * kappa;
    const double exponent = -(nu + 1.0);
    if (!isfinite(mass)) return CHEAP_EDOM;
    for (int i = 0; i < n0; ++i) {
        double si = sin(M_PI * (double)i / (2.0 * n0));
        double mui = 4.0 * si * si;
        for (int j = 0; j < n1; ++j) {
            double sj = sin(M_PI * (double)j / (2.0 * n1));
            double base = mass + a0 * mui + a1 * 4.0 * sj * sj;
            if (!isfinite(base) || base <= 0) return CHEAP_EDOM;
            double w = amplitude * pow(base, exponent);
            if (!isfinite(w)) return CHEAP_EDOM;
            out[(size_t)i * (size_t)n1 + (size_t)j] = w;
        }
    }
    return CHEAP_OK;
}

static inline int cheap_weights_resolvent_ev(
    int n, const double *mu, double shift, double *out)
{
    if (n < 1 || !mu || !out || !isfinite(shift) || shift <= 0)
        return CHEAP_EINVAL;
    for (int k = 0; k < n; ++k) {
        if (!isfinite(mu[k]) || mu[k] < 0) return CHEAP_EDOM;
        double w = 1.0 / (mu[k] + shift);
        if (!isfinite(w)) return CHEAP_EDOM;
        out[k] = w;
    }
    return CHEAP_OK;
}
static inline int cheap_weights_wave_cos_ev(
    int n, const double *mu, double t, double *out)
{
    if (n < 1 || !mu || !out || !isfinite(t)) return CHEAP_EINVAL;
    for (int k = 0; k < n; ++k) {
        if (!isfinite(mu[k]) || mu[k] < 0) return CHEAP_EDOM;
        out[k] = cos(t * sqrt(mu[k]));
    }
    return CHEAP_OK;
}
static inline int cheap_weights_wiener_signal_ev(
    int n, const double *signal_power, const double *noise_power,
    double *out)
{
    if (n < 1 || !signal_power || !noise_power || !out)
        return CHEAP_EINVAL;
    for (int k = 0; k < n; ++k) {
        double s = signal_power[k], z = noise_power[k];
        if (!isfinite(s) || !isfinite(z) || s < 0 || z < 0 ||
            (s == 0 && z == 0)) return CHEAP_EDOM;
        out[k] = s / (s + z);
    }
    return CHEAP_OK;
}
#endif
