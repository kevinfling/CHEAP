/* Reproducible C99 timing harness for original-system Toeplitz PCG,
 * Toeplitz matvec and PSD circulant draws. No numerical implementation
 * changes. Times include only operations identified by the CSV row.
 */
#define _POSIX_C_SOURCE 200809L
#include "cheap_pcg.h"
#include "cheap_circulant_sampler.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#ifndef CHEAP_BENCH_BACKEND
#define CHEAP_BENCH_BACKEND "FFTW3"
#endif
#define RUNS 40
static volatile double sink=0;
static double clock_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec*1000.+(double)ts.tv_nsec*1e-6;
}
static int cmp(const void *a,const void *b) {
    double x=*(const double*)a,y=*(const double*)b;
    return (x>y)-(x<y);
}
static void stats(double *a,double *med,double *p95,double *iqr) {
    qsort(a,RUNS,sizeof(double),cmp);
    *med=.5*(a[19]+a[20]);
    *p95=a[37];
    *iqr=a[29]-a[9];
}
static long peak_bytes(void) {
    struct rusage ru;
    if(getrusage(RUSAGE_SELF,&ru)!=0)return -1;
#if defined(__APPLE__)
    return (long)ru.ru_maxrss;
#else
    return (long)ru.ru_maxrss*1024L;
#endif
}
static void emit(const char *backend,const char *op,int n,double setup,
    double first,double *samples,int iterations,double residual,
    double covariance_error,int status,const char *policy) {
    double med,p95,iqr;
    stats(samples,&med,&p95,&iqr);
    printf("%s,%s,%d,1d,%s,exponential_rho_0.3,1e-8,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%.12g,%.12g,%ld,%s,%d\n",
        backend,CHEAP_BENCH_BACKEND,n,op,setup,first,med,p95,iqr,
        iterations,residual,covariance_error,peak_bytes(),policy,status);
}
int main(int argc,char **argv) {
    int n=argc>1?atoi(argv[1]):257;
    if(n<2||n>65536)return 2;
    double *col=malloc((size_t)n*sizeof(double));
    double *b=malloc((size_t)n*sizeof(double));
    double *x=malloc((size_t)n*sizeof(double));
    double *y=malloc((size_t)n*sizeof(double));
    if(!col||!b||!x||!y)return 3;
    for(int i=0;i<n;++i) {
        col[i]=pow(.3,(double)i);
        b[i]=sin(.1*(double)i)+.5;
    }
    puts("backend,configured_backend,n,dims,operator,params,tolerance,setup_ms,first_ms,median_ms,p95_ms,iqr_ms,iterations,true_residual,covariance_error,peak_bytes,replacement_policy,result");
    cheap_toeplitz_operator op={0};
    double t=clock_ms();
    if(cheap_toeplitz_operator_init(&op,n,col)!=CHEAP_OK)return 4;
    double setup=clock_ms()-t;
    cheap_linear_operator A=cheap_toeplitz_as_linear_operator(&op);
    double samples[RUNS];
    t=clock_ms();
    if(cheap_linear_operator_apply(&A,b,y)!=CHEAP_OK)return 5;
    double first=clock_ms()-t;
    for(int k=0;k<RUNS;++k) {
        t=clock_ms();
        if(cheap_linear_operator_apply(&A,b,y)!=CHEAP_OK)return 6;
        samples[k]=clock_ms()-t;
        sink+=y[k%n];
    }
    emit("toeplitz_fft","toeplitz_fft_apply",n,setup,first,samples,0,NAN,NAN,0,"none");
    cheap_pcg_workspace w={0};
    t=clock_ms();
    if(cheap_pcg_workspace_init(&w,n)!=CHEAP_OK)return 7;
    double wsetup=clock_ms()-t;
    cheap_pcg_options options={4*n+40,1e-10,1e-8,0,1,CHEAP_PCG_REFRESH_CONTINUE};
    cheap_pcg_result r;
    t=clock_ms();
    r=cheap_pcg_solve_opts(&A,NULL,b,x,&w,&options);
    first=clock_ms()-t;
    if(r.reason!=CHEAP_PCG_CONVERGED||!r.residual_valid)return 8;
    double residual=r.true_residual;
    for(int k=0;k<RUNS;++k) {
        t=clock_ms();
        r=cheap_pcg_solve_opts(&A,NULL,b,x,&w,&options);
        samples[k]=clock_ms()-t;
        if(r.reason!=CHEAP_PCG_CONVERGED||!r.residual_valid)return 9;
        sink+=x[k%n];
    }
    emit("pcg","original_toeplitz_cg",n,setup+wsetup,first,samples,
         r.iterations,residual,NAN,0,"continue");
    cheap_circulant_sampler s={0};
    t=clock_ms();
    if(cheap_circulant_sampler_init(&s,col,(size_t)n,0,1e-12,0)!=CHEAP_OK)return 10;
    setup=clock_ms()-t;
    double *z=malloc(s.info.L*sizeof(double));
    if(!z)return 11;
    /* deterministic input stream for timing, not a covariance validation */
    for(size_t i=0;i<s.info.L;++i)z[i]=sin((double)i*1.234);
    t=clock_ms();
    if(cheap_circulant_sampler_draw(&s,z,y)!=CHEAP_OK)return 12;
    first=clock_ms()-t;
    for(int k=0;k<RUNS;++k) {
        t=clock_ms();
        if(cheap_circulant_sampler_draw(&s,z,y)!=CHEAP_OK)return 13;
        samples[k]=clock_ms()-t;
        sink+=y[k%n];
    }
    emit("circulant","psd_embedding_draw_fft_only",n,setup,first,samples,
         0,NAN,NAN,0,"none");
    cheap_circulant_sampler_destroy(&s);
    cheap_pcg_workspace_destroy(&w);
    cheap_toeplitz_operator_destroy(&op);
    free(z);free(col);free(b);free(x);free(y);
    return sink==HUGE_VAL?14:0;
}
