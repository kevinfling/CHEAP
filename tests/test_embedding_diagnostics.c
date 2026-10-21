#include "cheap_circulant_sampler.h"
#include "cheap_spectral_diagnostics.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static uint64_t rng=UINT64_C(0x123456789abcdef);
static double uniform(void) {
    rng^=rng>>12;rng^=rng<<25;rng^=rng>>27;
    return ((rng*UINT64_C(2685821657736338717))>>11)*0x1.0p-53;
}
static double normal(void) {
    double u=uniform(),v=uniform();
    if(u<1e-15)u=1e-15;
    return sqrt(-2*log(u))*cos(2*M_PI*v);
}
static int sample(int n,double decay) {
    cheap_circulant_sampler s={0};
    double *lags=malloc((size_t)n*sizeof(double));
    double *z=malloc((size_t)(4*n+4)*sizeof(double));
    double *out=malloc((size_t)n*sizeof(double));
    int fail=1;
    if(!lags||!z||!out)goto done;
    for(int i=0;i<n;++i)lags[i]=pow(decay,(double)i);
    if(cheap_circulant_sampler_init(&s,lags,(size_t)n,0,1e-12,0)!=CHEAP_OK){
        fprintf(stderr,"PSD init failed n=%d decay=%g\n",n,decay);goto done;
    }
    if(s.info.status!=CHEAP_EMBED_EXACT)goto done;
    long double cov0=0,cov1=0,mean=0;
    int draws=12000;
    for(int j=0;j<draws;++j) {
        for(size_t k=0;k<s.info.L;++k)z[k]=normal();
        if(cheap_circulant_sampler_draw(&s,z,out)!=CHEAP_OK)goto done;
        mean+=out[0];
        cov0+=(long double)out[0]*out[0];
        if(n>1)cov1+=(long double)out[0]*out[1];
    }
    if(fabsl(mean/draws)>.07L ||
       fabsl(cov0/draws-1.)>.09L ||
       (n>1&&fabsl(cov1/draws-decay)>.09L)) {
        fprintf(stderr,"covariance recovery n=%d got %g,%g expected 1,%g\n",
                n,(double)(cov0/draws),(double)(cov1/draws),decay);goto done;
    }
    fail=0;
done:
    cheap_circulant_sampler_destroy(&s);
    cheap_circulant_sampler_destroy(&s);
    free(lags);free(z);free(out);
    return fail;
}
typedef struct {int n;double multiplier;} diag_ctx;
static int diag_apply(void *p,const double *x,double *y) {
    diag_ctx *d=(diag_ctx*)p;
    for(int i=0;i<d->n;++i)y[i]=x[i]*d->multiplier;
    return CHEAP_OK;
}
int main(void) {
    int sizes[]={1,2,3,5,8,17};
    for(int k=0;k<6;++k)
        if(sample(sizes[k],.3)||(sizes[k]>=3&&sample(sizes[k],.7)))return 1;
    cheap_circulant_sampler s={0};
    double bad[]={1.,2.},z[]={1.,0.,0.,0.},out[2];
    if(cheap_circulant_sampler_init(&s,bad,2,0,1e-12,0)!=CHEAP_EDOM)return 2;
    if(s.time||s.inverse)return 3;
    if(cheap_circulant_sampler_init_approx_clip(&s,bad,2,0,1e-12,0)!=CHEAP_OK ||
       s.info.status!=CHEAP_EMBED_APPROX_CLIP ||s.info.neg_count==0)return 4;
    if(cheap_circulant_sampler_draw(&s,z,out)!=CHEAP_OK)return 5;
    cheap_circulant_sampler_destroy(&s);
    /* Doubling a zero-padded embedding retains the old Fourier nodes:
     * a negative eigenvalue cannot disappear through zero-padding alone. */
    if(cheap_circulant_sampler_init_search(&s,bad,2,16,1e-12,0)!=CHEAP_EDOM ||
       s.info.status!=CHEAP_EMBED_NOT_PSD ||s.info.L!=16 ||
       s.time||s.inverse)return 9;
    memset(&s,0,sizeof(s));
    if(cheap_circulant_sampler_init_search(&s,bad,2,2,1e-12,0)!=CHEAP_EINVAL)
        return 10;
    {double nan_lags[]={1.,NAN};
     if(cheap_circulant_sampler_init(&s,nan_lags,2,0,1e-12,0)!=CHEAP_EDOM)
         return 11;
     if(cheap_circulant_sampler_init(&s,bad,(size_t)INT_MAX,0,1e-12,0)
        !=CHEAP_EINVAL)return 12;
    }
    double weights[]={1.,2.,4.};
    cheap_spectral_diag d;
    if(cheap_diag_from_weights(weights,3,&d)!=CHEAP_OK ||
       d.min_weight!=1 ||d.max_weight!=4 ||d.cond_est!=4 ||
       d.neg_count||d.zero_count||!isnan(d.proj_err))return 6;
    double col[]={1.,0.,0.};
    if(cheap_diag_projection_error(col,3,weights,&d)!=CHEAP_OK ||
       !isfinite(d.proj_err))return 7;
    diag_ctx a={3,2.},b={3,1.};
    if(cheap_diag_stochastic_probe(diag_apply,&a,diag_apply,&b,3,100,42,&d)
       !=CHEAP_OK ||fabs(d.probe_err-.5)>1e-12 ||
       d.seed!=42||d.n_probes!=100)return 8;
    puts("PASS: PSD embedding sampling, explicit clipping and spectral diagnostics");
    return 0;
}
