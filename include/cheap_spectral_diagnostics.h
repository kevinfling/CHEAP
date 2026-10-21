#ifndef CHEAP_SPECTRAL_DIAGNOSTICS_H
#define CHEAP_SPECTRAL_DIAGNOSTICS_H
/* Descriptive spectral and DCT-surrogate diagnostics; no exactness claims
 * for randomized probes. C99, deterministic seed and independent dense oracle.
 */
#include "cheap_toeplitz_operator.h"
#include <stdint.h>
typedef struct {
    double min_weight,max_weight;
    int neg_count,zero_count;
    double cond_est,proj_err,probe_err;
    int n_probes;
    uint64_t seed;
} cheap_spectral_diag;
static inline int cheap_diag_from_weights(
    const double *w,size_t n,cheap_spectral_diag *d)
{
    if(!w||!d||n<1||n>(size_t)INT_MAX)return CHEAP_EINVAL;
    cheap_spectral_diag t;
    t.min_weight=INFINITY;t.max_weight=-INFINITY;
    t.neg_count=0;t.zero_count=0;t.cond_est=NAN;
    t.proj_err=NAN;t.probe_err=NAN;t.n_probes=0;t.seed=0;
    for(size_t k=0;k<n;++k) {
        if(!isfinite(w[k]))return CHEAP_EDOM;
        if(w[k]<t.min_weight)t.min_weight=w[k];
        if(w[k]>t.max_weight)t.max_weight=w[k];
        if(w[k]<0)++t.neg_count;
        if(w[k]==0)++t.zero_count;
    }
    if(t.min_weight>0&&t.max_weight>0)
        t.cond_est=t.max_weight/t.min_weight;
    *d=t;return CHEAP_OK;
}
/* DCT-II orthonormal basis; matches the DCT-diagonal surrogate
 * Q diag(w) Q^T. n <= 256 to bound O(n^3) diagnostic work.
 */
static inline int cheap_diag_projection_error(
    const double *col,size_t n,const double *weights,cheap_spectral_diag *d)
{
    long double numerator=0,denominator=0;
    if(!col||!weights||!d||n<1||n>256)return CHEAP_EINVAL;
    for(size_t k=0;k<n;++k)
        if(!isfinite(col[k])||!isfinite(weights[k]))return CHEAP_EDOM;
    for(size_t i=0;i<n;++i)for(size_t j=0;j<n;++j) {
        long double approximation=0;
        for(size_t k=0;k<n;++k) {
            long double factor=k==0?1.0L/sqrtl((long double)n):
                sqrtl(2.0L/(long double)n);
            long double qi=factor*cosl(M_PI*((long double)i+.5L)*
                                              (long double)k/(long double)n);
            long double qj=factor*cosl(M_PI*((long double)j+.5L)*
                                              (long double)k/(long double)n);
            approximation+=(long double)weights[k]*qi*qj;
        }
        long double truth=col[i>j?i-j:j-i];
        long double diff=truth-approximation;
        numerator+=diff*diff;denominator+=truth*truth;
    }
    d->proj_err=denominator>0?(double)sqrtl(numerator/denominator):
                 (numerator==0?0.:INFINITY);
    return CHEAP_OK;
}
static inline uint64_t cheap_diag_next(uint64_t *s) {
    uint64_t x=*s;
    x^=x>>12;x^=x<<25;x^=x>>27;
    *s=x;return x*UINT64_C(2685821657736338717);
}
/* Report the RMS relative operator discrepancy across fixed-seed
 * Rademacher probes, NOT a deterministic or probabilistic upper bound.
 */
static inline int cheap_diag_stochastic_probe(
    cheap_matvec_fn A,void *A_user,
    cheap_matvec_fn approx,void *approx_user,
    size_t n,int probes,uint64_t seed,cheap_spectral_diag *d)
{
    if(!A||!approx||!A_user||!approx_user||!d||n<1||
       n>(size_t)INT_MAX||probes<1||seed==0||
       n>SIZE_MAX/(3*sizeof(double)))return CHEAP_EINVAL;
    double *mem=(double*)malloc(3*n*sizeof(double));
    if(!mem)return CHEAP_ENOMEM;
    double *v=mem,*a=mem+n,*b=mem+2*n;
    uint64_t state=seed;
    long double num=0,den=0;
    int rc=CHEAP_OK;
    for(int p=0;p<probes;++p) {
        for(size_t i=0;i<n;++i)v[i]=(cheap_diag_next(&state)&1)?1.:-1.;
        rc=A(A_user,v,a);if(rc!=CHEAP_OK)break;
        rc=approx(approx_user,v,b);if(rc!=CHEAP_OK)break;
        for(size_t i=0;i<n;++i) {
            if(!isfinite(a[i])||!isfinite(b[i])){rc=CHEAP_EDOM;break;}
            long double diff=(long double)a[i]-b[i];
            num+=diff*diff;den+=(long double)a[i]*a[i];
        }
        if(rc!=CHEAP_OK)break;
    }
    free(mem);
    if(rc!=CHEAP_OK)return rc;
    d->probe_err=den>0?(double)sqrtl(num/den):(num==0?0.:INFINITY);
    d->n_probes=probes;d->seed=seed;
    return CHEAP_OK;
}
#endif
