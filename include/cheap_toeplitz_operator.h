#ifndef CHEAP_TOEPLITZ_OPERATOR_H
#define CHEAP_TOEPLITZ_OPERATOR_H

/* Exact symmetric Toeplitz matrix-vector product via circulant embedding.
 * C99, FFTW3 or CHEAP_USE_NOTORIOUS_FFT (v1.1.1). Unlike DCT-diagonal
 * projections, this evaluates the actual finite Toeplitz matrix.
 * SPDX-License-Identifier: MIT
 */
#include "cheap.h"
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

/* Status-returning operator callback. On error the destination is unchanged.
 * Exact in-place input==output is permitted. Partial overlaps are rejected.
 * A context may be used concurrently with other independent contexts, but
 * not concurrently with itself (its FFT scratch and plans are mutable).
 */
typedef int (*cheap_matvec_fn)(void *context, const double *input, double *output);
typedef struct {
    int n;
    void *context;
    cheap_matvec_fn apply;
} cheap_linear_operator;

static inline int cheap_linear_operator_apply(
    const cheap_linear_operator *op, const double *x, double *y)
{
    if (!op || !op->apply || !op->context || op->n < 1 || !x || !y)
        return CHEAP_EINVAL;
    return op->apply(op->context, x, y);
}

typedef struct {
    int n;
    int fft_length;
    double *time;
    fftw_complex *frequency;
    fftw_complex *kernel;
    fftw_plan forward;
    fftw_plan inverse;
} cheap_toeplitz_operator;

/* Safe on a zero-initialized, initialized, or already-destroyed context.
 * Do not copy initialized contexts: FFT plans own their work buffers.
 */
static inline void cheap_toeplitz_operator_destroy(cheap_toeplitz_operator *op)
{
    if (!op) return;
    if (op->forward) fftw_destroy_plan(op->forward);
    if (op->inverse) fftw_destroy_plan(op->inverse);
    if (op->time) fftw_free(op->time);
    if (op->frequency) fftw_free(op->frequency);
    if (op->kernel) fftw_free(op->kernel);
    memset(op, 0, sizeof(*op));
}

/* Transactional construction: on failure *op remains a zeroed context.
 * The caller must provide a fresh/zeroed or previously destroyed context.
 * This avoids discarding an existing context's owned plans and buffers.
 * Embedding length L is the smallest power of two >= max(1,2n-1).
 * Power-of-two padding is intentional: stable FFT performance on both
 * backends, with checked arithmetic and no assumptions about odd R2C.
 */
static inline int cheap_toeplitz_operator_init(
    cheap_toeplitz_operator *op, int n, const double *first_column)
{
    cheap_toeplitz_operator tmp = {0};
    size_t required, length, bins, k;
    if (!op || !first_column || n < 1) return CHEAP_EINVAL;
    if (op->time || op->frequency || op->kernel || op->forward || op->inverse)
        return CHEAP_EINVAL;
    for (k = 0; k < (size_t)n; ++k)
        if (!isfinite(first_column[k])) return CHEAP_EDOM;
    if ((size_t)n > ((size_t)INT_MAX + 1u)/2u) return CHEAP_EINVAL;
    required = (size_t)n * 2u - 1u;
    length = 1;
    while (length < required) {
        if (length > (size_t)INT_MAX / 2u) return CHEAP_EINVAL;
        length *= 2u;
    }
    if (length > (size_t)INT_MAX ||
        length > SIZE_MAX / sizeof(double)) return CHEAP_EINVAL;
    bins = length / 2u + 1u;
    if (bins > SIZE_MAX / sizeof(fftw_complex)) return CHEAP_EINVAL;
    tmp.n = n;
    tmp.fft_length = (int)length;
    tmp.time = (double *)fftw_malloc(length * sizeof(double));
    tmp.frequency = (fftw_complex *)fftw_malloc(bins * sizeof(fftw_complex));
    tmp.kernel = (fftw_complex *)fftw_malloc(bins * sizeof(fftw_complex));
    if (!tmp.time || !tmp.frequency || !tmp.kernel) goto allocation_failed;
    memset(tmp.time, 0, length * sizeof(double));
    tmp.time[0] = first_column[0];
    for (k = 1; k < (size_t)n; ++k) {
        tmp.time[k] = first_column[k];
        tmp.time[length-k] = first_column[k];
    }
    tmp.forward = fftw_plan_dft_r2c_1d(tmp.fft_length, tmp.time,
                                       tmp.frequency, FFTW_ESTIMATE);
    if (!tmp.forward) goto allocation_failed;
    tmp.inverse = fftw_plan_dft_c2r_1d(tmp.fft_length, tmp.frequency,
                                       tmp.time, FFTW_ESTIMATE);
    if (!tmp.inverse) goto allocation_failed;
    /* FFTW planning may overwrite buffers; reconstruct embedding afterward. */
    memset(tmp.time, 0, length * sizeof(double));
    tmp.time[0] = first_column[0];
    for (k = 1; k < (size_t)n; ++k) {
        tmp.time[k] = first_column[k];
        tmp.time[length-k] = first_column[k];
    }
    fftw_execute(tmp.forward);
    memcpy(tmp.kernel, tmp.frequency, bins * sizeof(fftw_complex));
    for (k = 0; k < bins; ++k)
        if (!isfinite(tmp.kernel[k][0]) || !isfinite(tmp.kernel[k][1]))
            goto domain_failed;
    *op = tmp;
    return CHEAP_OK;
domain_failed:
    cheap_toeplitz_operator_destroy(&tmp);
    return CHEAP_EDOM;
allocation_failed:
    cheap_toeplitz_operator_destroy(&tmp);
    return CHEAP_ENOMEM; /* FFT planner failure is currently conflated. */
}

static inline int cheap_toeplitz_operator_apply(
    cheap_toeplitz_operator *op, const double *x, double *y)
{
    size_t n, length, bins, k;
    uintptr_t xa, ya;
    if (!op || !x || !y || !op->forward || !op->inverse ||
        !op->time || !op->frequency || !op->kernel || op->n < 1)
        return CHEAP_EUNINIT;
    n = (size_t)op->n;
    length = (size_t)op->fft_length;
    xa = (uintptr_t)x; ya = (uintptr_t)y;
    if (x != y && ((xa <= ya && ya-xa < n*sizeof(double)) ||
                   (ya < xa && xa-ya < n*sizeof(double))))
        return CHEAP_EINVAL;
    for (k = 0; k < n; ++k) if (!isfinite(x[k])) return CHEAP_EDOM;
    memset(op->time, 0, length*sizeof(double));
    memcpy(op->time, x, n*sizeof(double));
    fftw_execute(op->forward);
    bins = length/2u + 1u;
    for (k = 0; k < bins; ++k) {
        double ar = op->frequency[k][0], ai = op->frequency[k][1];
        double br = op->kernel[k][0], bi = op->kernel[k][1];
        op->frequency[k][0] = ar*br - ai*bi;
        op->frequency[k][1] = ar*bi + ai*br;
    }
    fftw_execute(op->inverse);
    for (k = 0; k < n; ++k)
        if (!isfinite(op->time[k]/(double)length)) return CHEAP_EDOM;
    for (k = 0; k < n; ++k) y[k] = op->time[k]/(double)length;
    return CHEAP_OK;
}

static inline int cheap_toeplitz_operator_callback(
    void *context, const double *x, double *y)
{
    return cheap_toeplitz_operator_apply((cheap_toeplitz_operator *)context, x, y);
}

static inline cheap_linear_operator cheap_toeplitz_as_linear_operator(
    cheap_toeplitz_operator *op)
{
    cheap_linear_operator result;
    result.n = op ? op->n : 0;
    result.context = op;
    result.apply = cheap_toeplitz_operator_callback;
    return result;
}

#endif /* CHEAP_TOEPLITZ_OPERATOR_H */
