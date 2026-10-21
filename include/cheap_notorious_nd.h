/* CHEAP-only extension of Notorious-FFT's FFTW-shaped subset.
 * The upstream compatibility header currently exposes 1D r2r plans only.
 * Its multidimensional DCT2/DCT3 auxiliaries are used here without
 * allocating or planning during execute. Mixed r2r kinds are rejected.
 * Not an FFTW ABI shim; only compile with CHEAP_USE_NOTORIOUS_FFT.
 */
#ifndef CHEAP_NOTORIOUS_ND_H
#define CHEAP_NOTORIOUS_ND_H
#include "notorious_fft_fftw.h"
#include <stdlib.h>

static inline fftw_plan cheap_notorious_plan_r2r_nd(
    int rank, const int *dims, double *in, double *out,
    const fftw_r2r_kind *kinds, unsigned flags)
{
    if (!dims || !in || !out || !kinds || (rank != 2 && rank != 3))
        return NULL;
    for (int j = 0; j < rank; ++j) {
        if (dims[j] < 1 || kinds[j] != kinds[0]) return NULL;
    }
    if (kinds[0] != FFTW_REDFT10 && kinds[0] != FFTW_REDFT01)
        return NULL;
    notorious_fft_aux *aux = rank == 2
        ? notorious_fft_mkaux_t2t3_2d(dims[1], dims[0])
        : notorious_fft_mkaux_t2t3_3d(dims[2], dims[1], dims[0]);
    if (!aux) return NULL;
    notorious_fft_io_plan *p =
        (notorious_fft_io_plan *)calloc(1, sizeof(*p));
    if (!p) {
        notorious_fft_free_aux(aux);
        return NULL;
    }
    p->rank = rank;
    for (int j = 0; j < rank; ++j) {
        p->n[j] = dims[j];
        p->r2r_kind[j] = kinds[j];
    }
    p->kind = NOTORIOUS_FFT_IO_R2R;
    p->flags = flags;
    p->in = in;
    p->out = out;
    p->aux = aux;
    p->howmany = 1;
    p->istride = p->ostride = 1;
    p->idist = p->odist = 1;
    return p;
}
static inline fftw_plan cheap_notorious_plan_r2r_2d(
    int n0, int n1, double *in, double *out,
    fftw_r2r_kind k0, fftw_r2r_kind k1, unsigned flags)
{
    const int dims[2] = {n0, n1};
    const fftw_r2r_kind kinds[2] = {k0, k1};
    return cheap_notorious_plan_r2r_nd(2, dims, in, out, kinds, flags);
}
static inline fftw_plan cheap_notorious_plan_r2r_3d(
    int n0, int n1, int n2, double *in, double *out,
    fftw_r2r_kind k0, fftw_r2r_kind k1, fftw_r2r_kind k2, unsigned flags)
{
    const int dims[3] = {n0, n1, n2};
    const fftw_r2r_kind kinds[3] = {k0, k1, k2};
    return cheap_notorious_plan_r2r_nd(3, dims, in, out, kinds, flags);
}
#define fftw_plan_r2r_2d cheap_notorious_plan_r2r_2d
#define fftw_plan_r2r_3d cheap_notorious_plan_r2r_3d
#endif
