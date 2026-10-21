#ifndef CHEAP_CIRCULANT_SAMPLER_H
#define CHEAP_CIRCULANT_SAMPLER_H
/* Stationary 1D Gaussian covariance sampling by circulant embedding.
 * Distinct from the CHEAP DCT-diagonal surrogate. C99, header-only.
 * Backend selected at compile time by CHEAP_USE_NOTORIOUS_FFT.
 */
#include "cheap_toeplitz_operator.h"
#include <limits.h>
#include <stdint.h>
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524400844362104849039
#endif
typedef enum {CHEAP_EMBED_EXACT=0,CHEAP_EMBED_APPROX_CLIP=1,
              CHEAP_EMBED_NOT_PSD=2} cheap_embedding_status;
typedef struct {
    size_t n,L;
    double tol,min_eig,max_eig;
    int neg_count;
    cheap_embedding_status status;
} cheap_embedding_info;
typedef struct cheap_circulant_sampler {
    cheap_embedding_info info;
    double *time,*sqrt_eigs;
    fftw_complex *freq;
    fftw_plan forward,inverse;
} cheap_circulant_sampler;
static inline void cheap_circulant_sampler_destroy(cheap_circulant_sampler *s) {
    if(!s)return;
    if(s->forward)fftw_destroy_plan(s->forward);
    if(s->inverse)fftw_destroy_plan(s->inverse);
    if(s->time)fftw_free(s->time);
    if(s->freq)fftw_free(s->freq);
    if(s->sqrt_eigs)free(s->sqrt_eigs);
    memset(s,0,sizeof(*s));
}
/* Exact mode refuses materially negative eigenvalues. Tolerance-level
 * negative roundoff is set to zero and the resulting covariance differs
 * from the requested covariance by at most that spectral perturbation.
 * For strict mathematical exactness, pass tol=0.
 * Explicit clipping mode accepts materially negative spectra, reporting
 * CHEAP_EMBED_APPROX_CLIP; never describe those draws as exact.
 * L=0 chooses minimal power of two >=2n-1. Explicit L must be power of two.
 * backend is a compile-time selection, not a runtime choice: backend=0
 * denotes the configured FFTW-shaped backend.
 */
static inline int cheap_circulant_sampler_init_mode(
    cheap_circulant_sampler *s,const double *lags,size_t n,size_t L,
    double tol,int backend,int allow_approx)
{
    cheap_circulant_sampler t={0};
    size_t needed,bins,k;
    double threshold;
    if(!s||!lags||n<1||!isfinite(tol)||tol<0||backend!=0||
       s->time||s->freq||s->sqrt_eigs||s->forward||s->inverse)
        return CHEAP_EINVAL;
    if(n>(size_t)INT_MAX/2u) return CHEAP_EINVAL;
    needed=2*n-1;
    if(L==0) {
        L=1;
        while(L<needed) {
            if(L>(size_t)INT_MAX/2u)return CHEAP_EINVAL;
            L*=2;
        }
    }
    if(L<needed||L>(size_t)INT_MAX||(L&(L-1))||
       L>SIZE_MAX/sizeof(double))return CHEAP_EINVAL;
    bins=L/2+1;
    if(bins>SIZE_MAX/sizeof(fftw_complex)||
       bins>SIZE_MAX/sizeof(double))return CHEAP_EINVAL;
    for(k=0;k<n;++k)if(!isfinite(lags[k]))return CHEAP_EDOM;
    if(lags[0]<0)return CHEAP_EDOM;
    t.info.n=n;t.info.L=L;t.info.tol=tol;
    t.time=(double*)fftw_malloc(L*sizeof(double));
    t.freq=(fftw_complex*)fftw_malloc(bins*sizeof(fftw_complex));
    t.sqrt_eigs=(double*)malloc(bins*sizeof(double));
    if(!t.time||!t.freq||!t.sqrt_eigs)goto oom;
    t.forward=fftw_plan_dft_r2c_1d((int)L,t.time,t.freq,FFTW_ESTIMATE);
    if(!t.forward)goto oom;
    t.inverse=fftw_plan_dft_c2r_1d((int)L,t.freq,t.time,FFTW_ESTIMATE);
    if(!t.inverse)goto oom;
    memset(t.time,0,L*sizeof(double));
    t.time[0]=lags[0];
    for(k=1;k<n;++k){t.time[k]=lags[k];t.time[L-k]=lags[k];}
    fftw_execute(t.forward);
    t.info.min_eig=INFINITY;t.info.max_eig=-INFINITY;
    for(k=0;k<bins;++k){
        double v=t.freq[k][0];
        if(!isfinite(v)||!isfinite(t.freq[k][1]))goto domain;
        if(v<t.info.min_eig)t.info.min_eig=v;
        if(v>t.info.max_eig)t.info.max_eig=v;
        if(v<0)++t.info.neg_count;
    }
    threshold=tol*fmax(fabs(t.info.min_eig),fabs(t.info.max_eig));
    if(!isfinite(threshold))goto domain;
    t.info.status=CHEAP_EMBED_EXACT;
    if(t.info.min_eig < -threshold) {
        t.info.status=allow_approx?CHEAP_EMBED_APPROX_CLIP:CHEAP_EMBED_NOT_PSD;
        if(!allow_approx){*s=t;cheap_circulant_sampler_destroy(s);return CHEAP_EDOM;}
    }
    for(k=0;k<bins;++k){
        double v=fmax(0.,t.freq[k][0]);
        t.sqrt_eigs[k]=sqrt(v)*(k==0||(L%2==0&&k==L/2)?1.:M_SQRT1_2);
    }
    *s=t;return CHEAP_OK;
domain:
    cheap_circulant_sampler_destroy(&t);return CHEAP_EDOM;
oom:
    cheap_circulant_sampler_destroy(&t);return CHEAP_ENOMEM;
}
static inline int cheap_circulant_sampler_init(
    cheap_circulant_sampler *s,const double *lags,size_t n,size_t L,
    double tol,int backend)
{return cheap_circulant_sampler_init_mode(s,lags,n,L,tol,backend,0);}
static inline int cheap_circulant_sampler_init_approx_clip(
    cheap_circulant_sampler *s,const double *lags,size_t n,size_t L,
    double tol,int backend)
{return cheap_circulant_sampler_init_mode(s,lags,n,L,tol,backend,1);}
/* z: L independent standard normals, with packed frequency indexing:
 * z[0] for DC; z[2k-1],z[2k] for interior real/imag components;
 * z[L-1] for Nyquist when L even. No hidden RNG or hot-path allocation.
 * FFTW C2R is unnormalized: divide its output by sqrt(L).
 */
static inline int cheap_circulant_sampler_draw(
    cheap_circulant_sampler *s,const double *z,double *out)
{
    size_t L,n,k,bins;
    if(!s||!z||!out||!s->inverse||!s->freq||!s->time||
       !s->sqrt_eigs)return CHEAP_EUNINIT;
    L=s->info.L;n=s->info.n;bins=L/2+1;
    for(k=0;k<L;++k)if(!isfinite(z[k]))return CHEAP_EDOM;
    s->freq[0][0]=s->sqrt_eigs[0]*z[0];
    s->freq[0][1]=0;
    for(k=1;k<bins;++k){
        if(L%2==0&&k==L/2){
            s->freq[k][0]=s->sqrt_eigs[k]*z[L-1];
            s->freq[k][1]=0;
        }else{
            s->freq[k][0]=s->sqrt_eigs[k]*z[2*k-1];
            s->freq[k][1]=s->sqrt_eigs[k]*z[2*k];
        }
    }
    fftw_execute(s->inverse);
    for(k=0;k<n;++k)
        if(!isfinite(s->time[k]/sqrt((double)L)))return CHEAP_EDOM;
    for(k=0;k<n;++k)out[k]=s->time[k]/sqrt((double)L);
    return CHEAP_OK;
}
#endif
