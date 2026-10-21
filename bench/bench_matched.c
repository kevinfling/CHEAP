/* Reproducible C99 timing harness for original-system Toeplitz PCG,
 * Toeplitz matvec and PSD circulant draws. No numerical implementation
 * changes. Times include only operations identified by the CSV row.
 */
#define _POSIX_C_SOURCE 200809L
/* Keep the benchmark's planner policy local to this translation unit. The
 * production DCT initializer requests PATIENT; this experiment consistently
 * uses ESTIMATE, including DCT plans, without changing the library headers. */
#if defined(CHEAP_USE_NOTORIOUS_FFT)
#include "notorious_fft.h"
#include "notorious_fft_fftw.h"
#else
#include <fftw3.h>
#endif
static fftw_plan bench_plan_r2r_1d(int n,double *in,double *out,
                                  fftw_r2r_kind kind,unsigned flags) {
    (void)flags;
    return fftw_plan_r2r_1d(n,in,out,kind,FFTW_ESTIMATE);
}
#ifdef fftw_plan_r2r_1d
#undef fftw_plan_r2r_1d
#endif
#define fftw_plan_r2r_1d bench_plan_r2r_1d
#include "cheap_pcg.h"
#include "cheap_circulant_sampler.h"
#undef fftw_plan_r2r_1d
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/resource.h>
#include <errno.h>
#ifndef CHEAP_BENCH_BACKEND
#define CHEAP_BENCH_BACKEND "FFTW3"
#endif
#define RUNS 40
#define COV_DRAWS 12000
static volatile double sink=0;
static unsigned long long rng_state=0x9e3779b97f4a7c15ULL;
static double normal_sample(void) {
    rng_state^=rng_state>>12;rng_state^=rng_state<<25;rng_state^=rng_state>>27;
    unsigned long long a=rng_state*2685821657736338717ULL;
    double u=((a>>11)+0.5)*0x1.0p-53;
    rng_state^=rng_state>>12;rng_state^=rng_state<<25;rng_state^=rng_state>>27;
    unsigned long long b=rng_state*2685821657736338717ULL;
    double v=((b>>11)+0.5)*0x1.0p-53;
    return sqrt(-2*log(u))*cos(6.2831853071795864769*v);
}
static double dense_error(int n,const double *c,const double *x,const double *y) {
    long double num=0,den=0;
    for(int i=0;i<n;++i) {
        long double expected=0;
        for(int j=0;j<n;++j)expected+=(long double)c[abs(i-j)]*x[j];
        long double e=expected-y[i];
        num+=e*e;den+=expected*expected;
    }
    return (double)sqrtl(num/(den>0?den:1));
}

/* Independent O(n) AR(1) oracle, evaluated in long double outside timing.
 * Left/right geometric sums evaluate sum_j rho^|i-j| x[j] without FFT or DCT.
 * At small sizes, cross-check this oracle against the existing dense oracle. */
static void ar1_apply(int n,double rho,const double *x,long double *y) {
    long double left=0,right=0;
    for(int i=0;i<n;++i) {left=x[i]+rho*left;y[i]=left;}
    for(int i=n-1;i>=0;--i) {y[i]+=rho*right;right=x[i]+rho*right;}
}
static double ar1_residual(int n,double rho,const double *b,const double *x,
                           long double *scratch,double bnorm) {
    ar1_apply(n,rho,x,scratch);
    long double sum=0;
    for(int i=0;i<n;++i) {
        long double e=(long double)b[i]-scratch[i];
        sum+=e*e;
    }
    return (double)(sqrtl(sum)/bnorm);
}

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
typedef struct {
    cheap_pcg_result first,last;
    double first_ms,first_independent,max_relative,max_independent;
    int failed,status,min_iterations,max_iterations;
} solve_summary;

static void emit(const char *backend,const char *op,int n,double rho,double setup,
    double first,double *samples,int iterations,double residual,
    double covariance_error,int status,const char *policy,double common_setup,
    double preconditioner_setup,int fft_length,const solve_summary *sol) {
    double med,p95,iqr;
    stats(samples,&med,&p95,&iqr);
    printf("%s,%s,%d,1d,%s,exponential_rho_%.17g,1e-8,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%.12g,%.12g,%ld,%s,%d",
        backend,CHEAP_BENCH_BACKEND,n,op,rho,setup,first,med,p95,iqr,
        iterations,residual,covariance_error,peak_bytes(),policy,status);
    printf(",%.17g,1e-10,%.9g,%.9g,%.12g,%.12g,%.12g,%.12g,%d,%d,%d,%d,%d,%d,%d,%d,FFTW_ESTIMATE\n",
        rho,common_setup,preconditioner_setup,
        sol?sol->first.relative_residual:NAN,sol?sol->first_independent:NAN,
        sol?sol->max_relative:NAN,sol?sol->max_independent:NAN,
        sol?sol->last.operator_calls:0,sol?sol->last.preconditioner_calls:0,
        sol?sol->last.residual_valid:0,sol?sol->first.reason:0,
        sol?sol->failed:0,sol?sol->min_iterations:0,sol?sol->max_iterations:0,
        fft_length);
}

static solve_summary time_solves(const cheap_linear_operator *A,
    const cheap_linear_operator *M,const double *b,double *x,
    cheap_pcg_workspace *w,const cheap_pcg_options *options,double rho,
    long double *scratch,double bnorm,const char *name,FILE *log,double *samples) {
    solve_summary s={0};
    s.min_iterations=INT_MAX;
    for(int k=-1;k<RUNS;++k) {
        double t=clock_ms();
        cheap_pcg_result r=cheap_pcg_solve_opts(A,M,b,x,w,options);
        double elapsed=clock_ms()-t;
        double independent=ar1_residual(A->n,rho,b,x,scratch,bnorm);
        double limit=options->atol+options->rtol*bnorm;
        int accepted=r.reason==CHEAP_PCG_CONVERGED && r.residual_valid &&
            isfinite(r.true_residual) && r.true_residual<=limit &&
            isfinite(r.relative_residual) && isfinite(independent) &&
            independent*bnorm<=limit;
        if(k<0) {s.first=r;s.first_ms=elapsed;s.first_independent=independent;}
        else samples[k]=elapsed;
        s.last=r;
        if(r.iterations<s.min_iterations)s.min_iterations=r.iterations;
        if(r.iterations>s.max_iterations)s.max_iterations=r.iterations;
        double rel=isfinite(r.relative_residual)?r.relative_residual:INFINITY;
        if(rel>s.max_relative)s.max_relative=rel;
        if(!isfinite(independent))independent=INFINITY;
        if(independent>s.max_independent)s.max_independent=independent;
        if(!accepted) {
            ++s.failed;
            if(!s.status)s.status=r.reason?(int)r.reason:CHEAP_EDOM;
            fprintf(stderr,"%s n=%d rho=%.17g repetition=%d reason=%d residual=%.12g independent_relative=%.12g\n",
                name,A->n,rho,k,r.reason,r.true_residual,independent);
        }
        if(log)fprintf(log,"%s,%d,%.17g,%s,%s,%d,%.9g,%d,%d,%d,%.17g,%.17g,%.17g,%d,%d,%d\n",
            CHEAP_BENCH_BACKEND,A->n,rho,name,k<0?"first":"steady",k,
            elapsed,r.iterations,r.operator_calls,r.preconditioner_calls,
            r.true_residual,r.relative_residual,independent,r.residual_valid,
            r.reason,accepted);
        sink+=x[(k+1)%A->n];
    }
    return s;
}
int main(int argc,char **argv) {
    int n=257,pcg_only=0,exit_status=0;
    double rho=.3;
    char *end;
    if(argc>5)goto usage;
    if(argc>1) {
        errno=0;
        long value=strtol(argv[1],&end,10);
        if(errno||end==argv[1]||*end||value<2||value>65536)goto usage;
        n=(int)value;
    }
    if(argc>2) {
        errno=0;rho=strtod(argv[2],&end);
        if(errno||end==argv[2]||*end||!isfinite(rho)||rho<0||rho>=1)goto usage;
    }
    if(argc>3) {
        if(strcmp(argv[3],"pcg")==0)pcg_only=1;
        else if(strcmp(argv[3],"all")!=0)goto usage;
    }
    FILE *log=argc>4?fopen(argv[4],"w"):NULL;
    if(argc>4&&!log){perror(argv[4]);return 2;}
    if(log)fputs("configured_backend,n,rho,operator,phase,repetition,elapsed_ms,iterations,operator_calls,preconditioner_calls,true_residual,relative_residual,independent_relative_residual,residual_valid,reason,accepted\n",log);
    cheap_toeplitz_operator op={0};
    cheap_pcg_workspace w={0};
    cheap_ctx dct={0};
    cheap_pcg_dct_preconditioner pre={0};
    cheap_circulant_sampler s={0};
    double *z=NULL;
    double *col=malloc((size_t)n*sizeof(double));
    double *b=malloc((size_t)n*sizeof(double));
    double *x=malloc((size_t)n*sizeof(double));
    double *y=malloc((size_t)n*sizeof(double));
    long double *scratch=malloc((size_t)n*sizeof(long double));
    const char *stage="input allocation";
    int rc=CHEAP_ENOMEM;
    if(!col||!b||!x||!y||!scratch)goto fail;
    for(int i=0;i<n;++i) {
        col[i]=pow(rho,(double)i);
        b[i]=sin(.1*(double)i)+.5;
    }
    double bnorm=cheap_pcg_norm(n,b);
    /* Validate the recurrence oracle independently for all small cases. */
    if(n<=1024) {
        ar1_apply(n,rho,b,scratch);
        for(int i=0;i<n;++i)y[i]=(double)scratch[i];
        stage="AR(1) oracle dense cross-check";rc=CHEAP_EDOM;
        double error=dense_error(n,col,b,y);
        if(!isfinite(error)||error>1e-12)goto fail;
    }
    puts("backend,configured_backend,n,dims,operator,params,tolerance,setup_ms,first_ms,median_ms,p95_ms,iqr_ms,iterations,true_residual,covariance_error,peak_bytes,replacement_policy,result,rho,atol,common_setup_ms,preconditioner_setup_ms,relative_residual,independent_relative_residual,max_relative_residual,max_independent_relative_residual,operator_calls,preconditioner_calls,residual_valid,first_result,failed_solves,min_iterations,max_iterations,fft_length,planner");
    double t=clock_ms();
    stage="Toeplitz setup";
    if((rc=cheap_toeplitz_operator_init(&op,n,col))!=CHEAP_OK)goto fail;
    double setup=clock_ms()-t;
    cheap_linear_operator A=cheap_toeplitz_as_linear_operator(&op);
    double samples[RUNS];
    double first=0;
    if(!pcg_only) {
        t=clock_ms();
        stage="Toeplitz apply";
        if((rc=cheap_linear_operator_apply(&A,b,y))!=CHEAP_OK)goto fail;
        first=clock_ms()-t;
        for(int k=0;k<RUNS;++k) {
            t=clock_ms();
            if((rc=cheap_linear_operator_apply(&A,b,y))!=CHEAP_OK)goto fail;
            samples[k]=clock_ms()-t;
            sink+=y[k%n];
        }
        double matvec_error=n<=1024?dense_error(n,col,b,y):NAN;
        stage="Toeplitz dense cross-check";rc=CHEAP_EDOM;
        if(n<=1024 && (!isfinite(matvec_error)||matvec_error>1e-10))goto fail;
        emit("toeplitz_fft","toeplitz_fft_apply",n,rho,setup,first,samples,
             0,matvec_error,NAN,0,"none",setup,0,op.fft_length,NULL);
    }
    t=clock_ms();
    stage="PCG workspace setup";
    if((rc=cheap_pcg_workspace_init(&w,n))!=CHEAP_OK)goto fail;
    double wsetup=clock_ms()-t;
    cheap_pcg_options options={4*n+40,1e-10,1e-8,0,1,CHEAP_PCG_REFRESH_CONTINUE};
    double common_setup=setup+wsetup;
    solve_summary cg=time_solves(&A,NULL,b,x,&w,&options,rho,scratch,bnorm,
                                 "original_toeplitz_cg",log,samples);
    emit("pcg","original_toeplitz_cg",n,rho,common_setup,cg.first_ms,samples,
         cg.last.iterations,cg.first.true_residual,NAN,cg.status,"continue",
         common_setup,0,op.fft_length,&cg);
    t=clock_ms();
    stage="DCT projection setup";
    if((rc=cheap_init_from_toeplitz_projection(&dct,n,col))!=CHEAP_OK)goto fail;
    stage="DCT inverse preconditioner setup";
    if((rc=cheap_pcg_dct_init(&pre,&dct,dct.lambda))!=CHEAP_OK)goto fail;
    double pre_setup=clock_ms()-t;
    cheap_linear_operator M=cheap_pcg_dct_as_operator(&pre);
    solve_summary pcg=time_solves(&A,&M,b,x,&w,&options,rho,scratch,bnorm,
                                  "original_toeplitz_dct_pcg",log,samples);
    emit("dct_pcg","original_toeplitz_dct_pcg",n,rho,common_setup+pre_setup,
         pcg.first_ms,samples,pcg.last.iterations,pcg.first.true_residual,NAN,
         pcg.status,"continue",common_setup,pre_setup,op.fft_length,&pcg);
    if(cg.failed||pcg.failed)exit_status=1;
    if(pcg_only)goto cleanup;
    t=clock_ms();
    stage="circulant sampler setup";
    if((rc=cheap_circulant_sampler_init(&s,col,(size_t)n,0,1e-12,0))!=CHEAP_OK)goto fail;
    setup=clock_ms()-t;
    z=malloc(s.info.L*sizeof(double));
    stage="sampler input allocation";rc=CHEAP_ENOMEM;
    if(!z)goto fail;
    /* deterministic input stream for timing, not a covariance validation */
    for(size_t i=0;i<s.info.L;++i)z[i]=sin((double)i*1.234);
    t=clock_ms();
    stage="sampler draw";
    if((rc=cheap_circulant_sampler_draw(&s,z,y))!=CHEAP_OK)goto fail;
    first=clock_ms()-t;
    for(int k=0;k<RUNS;++k) {
        t=clock_ms();
        if((rc=cheap_circulant_sampler_draw(&s,z,y))!=CHEAP_OK)goto fail;
        samples[k]=clock_ms()-t;
        sink+=y[k%n];
    }
    /* Separate seeded covariance check; timed FFT calls exclude RNG. */
    long double m0=0,m1=0,c0=0,c1=0;
    for(int j=0;j<COV_DRAWS;++j) {
        for(size_t i=0;i<s.info.L;++i)z[i]=normal_sample();
        if((rc=cheap_circulant_sampler_draw(&s,z,y))!=CHEAP_OK)goto fail;
        m0+=y[0];m1+=y[1];c0+=(long double)y[0]*y[0];
        c1+=(long double)y[0]*y[1];
    }
    long double e0=c0/COV_DRAWS-(m0/COV_DRAWS)*(m0/COV_DRAWS)-col[0];
    long double e1=c1/COV_DRAWS-(m0/COV_DRAWS)*(m1/COV_DRAWS)-col[1];
    double covariance_error=(double)sqrtl((e0*e0+e1*e1)/
                                           (col[0]*col[0]+col[1]*col[1]));
    stage="sampler covariance check";rc=CHEAP_EDOM;
    if(!isfinite(covariance_error)||covariance_error>0.12)goto fail;
    emit("circulant","psd_embedding_draw_fft_only",n,rho,setup,first,samples,
         0,NAN,covariance_error,0,"none",setup,0,(int)s.info.L,NULL);
    goto cleanup;
fail:
    fprintf(stderr,"%s failed: n=%d rho=%.17g status=%d\n",stage,n,rho,rc);
    exit_status=1;
cleanup:
    cheap_circulant_sampler_destroy(&s);
    cheap_pcg_dct_destroy(&pre);
    cheap_destroy(&dct);
    cheap_pcg_workspace_destroy(&w);
    cheap_toeplitz_operator_destroy(&op);
    free(z);free(col);free(b);free(x);free(y);free(scratch);
    if(log&&fclose(log)!=0){perror("solve sample log");exit_status=1;}
    if(fflush(stdout)!=0)exit_status=1;
    return sink==HUGE_VAL?14:exit_status;
usage:
    fprintf(stderr,"usage: %s [n [rho [all|pcg [solve_samples.csv]]]]\n"
                   "n: 2..65536; rho: finite, 0 <= rho < 1; defaults: 257 0.3 all\n",argv[0]);
    return 2;
}
