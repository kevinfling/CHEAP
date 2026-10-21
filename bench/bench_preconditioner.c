/* Isolate the existing DCT preconditioner and its component costs.
 * No numerical library/backend modifications; all planning uses ESTIMATE. */
#define _POSIX_C_SOURCE 200809L
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
#undef fftw_plan_r2r_1d
#include <errno.h>
#include <stdio.h>
#include <sys/resource.h>
#include <time.h>

#ifndef CHEAP_BENCH_BACKEND
#define CHEAP_BENCH_BACKEND "FFTW3"
#endif
#define RUNS 40
#define ERROR_LIMIT 1e-9
static volatile double sink;
static const char *names[]={"preconditioner_apply","cheap_apply",
    "validate_copy","dct2_execute","weight_multiply","dct3_execute","scale_copy"};

static double clock_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec*1000.+(double)ts.tv_nsec*1e-6;
}
static int cmp(const void *a,const void *b) {
    double x=*(const double*)a,y=*(const double*)b;
    return (x>y)-(x<y);
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
static double relative_error(int n,const double *actual,const long double *expected) {
    long double num=0,den=0;
    for(int i=0;i<n;++i) {
        if(!isfinite(actual[i]))return INFINITY;
        long double e=actual[i]-expected[i];
        num+=e*e;den+=expected[i]*expected[i];
    }
    return (double)sqrtl(num/(den>0?den:1));
}

/* Direct cosine sums are an untimed oracle, independent of backend transforms.
 * References cover the complete inverse application and each isolated stage.
 * Component transforms take fixed b as input, not repeatedly transformed data. */
static void references(int n,const double *b,const double *weights,
                        long double *expected,long double *coeff) {
    for(int k=0;k<n;++k) {
        long double sum=0;
        for(int i=0;i<n;++i)sum+=b[i]*cos(M_PI*k*(i+.5)/n);
        expected[3*n+k]=2*sum;
        coeff[k]=sum*weights[k]*(k==0?1.L:2.L)/n;
    }
    for(int i=0;i<n;++i) {
        long double full=0,inv=b[0];
        for(int k=0;k<n;++k) {
            double c=cos(M_PI*k*(i+.5)/n);
            full+=coeff[k]*c;
            if(k)inv+=2.L*b[k]*c;
        }
        expected[i]=expected[n+i]=full;
        expected[2*n+i]=b[i];
        expected[4*n+i]=(long double)b[i]*weights[i];
        expected[5*n+i]=inv;
        expected[6*n+i]=(long double)b[i]/(2*n);
    }
}

static int execute(int operation,cheap_ctx *ctx,cheap_pcg_dct_preconditioner *pre,
                    const double *b,double *out) {
    switch(operation) {
    case 0: return cheap_pcg_dct_apply(pre,b,out);
    case 1: return cheap_apply(ctx,b,pre->inverse_weights,out);
    case 2:
        for(int i=0;i<ctx->n;++i)if(!isfinite(b[i]))return CHEAP_EDOM;
        memcpy(ctx->workspace,b,(size_t)ctx->n*sizeof(double));break;
    case 3: fftw_execute(ctx->plan_fwd);break;
    case 4: cheap__mul_inplace(ctx->workspace,pre->inverse_weights,ctx->n);break;
    case 5: fftw_execute(ctx->plan_inv);break;
    case 6: cheap__scale_copy(out,ctx->workspace,1./(2*ctx->n),ctx->n);break;
    default:return CHEAP_EINVAL;
    }
    return CHEAP_OK;
}

static void dump_plan(FILE *f,const char *name,fftw_plan plan) {
    fprintf(f,"%s: ",name);
#if defined(CHEAP_USE_NOTORIOUS_FFT)
    /* Pinned backend exposes plan structs, but has no FFTW plan printer. */
    notorious_fft_aux *axis=plan->axis_aux[0];
    notorious_fft_plan *fft=axis?axis->plan:NULL;
    fprintf(f,"rank=%d n=%d flags=%u r2r_kind=%d in_place=%d "
        "axis_N=%d direct_r2r=%d fft_present=%d",
        plan->rank,plan->n[0],plan->flags,(int)plan->r2r_kind[0],
        plan->in==plan->out,axis?axis->N:0,axis?axis->direct_r2r:-1,fft!=NULL);
#if defined(NOTORIOUS_R2R_ALGO_DCT_FFT_N)
    if(axis)fprintf(f," r2r_algo=%d",axis->r2r_algo);
#endif
    if(fft)fprintf(f," fft_n=%zu mixed_radix=%d bluestein_n=%zu rader_active=%d",
        fft->n,fft->mixed_radix,fft->bluestein_n,fft->rader_sub!=NULL);
    fputc('\n',f);
#else
    fftw_fprint_plan(plan,f);
    fputc('\n',f);
#endif
}

int main(int argc,char **argv) {
    char *end;
    if(argc<3||argc>4)goto usage;
    errno=0;
    long value=strtol(argv[1],&end,10);
    if(errno||end==argv[1]||*end||value<2||value>4096)goto usage;
    int n=(int)value;
    double rho=.99;
    if(argc==4) {
        errno=0;rho=strtod(argv[3],&end);
        if(errno||end==argv[3]||*end||!isfinite(rho)||rho<0||rho>=1)goto usage;
    }
    char path[4096];
    int length=snprintf(path,sizeof(path),"%s-samples.csv",argv[2]);
    if(length<0||(size_t)length>=sizeof(path))goto usage;
    FILE *log=fopen(path,"w");
    if(!log){perror(path);return 2;}
    length=snprintf(path,sizeof(path),"%s-plans.txt",argv[2]);
    if(length<0||(size_t)length>=sizeof(path)){fclose(log);goto usage;}
    FILE *plans=fopen(path,"w");
    if(!plans){perror(path);fclose(log);return 2;}
    cheap_ctx ctx={0};
    cheap_pcg_dct_preconditioner pre={0};
    double *col=malloc((size_t)n*sizeof(double));
    double *b=malloc((size_t)n*sizeof(double));
    double *out=malloc((size_t)n*sizeof(double));
    long double *ref=malloc((size_t)7*n*sizeof(long double));
    long double *coeff=malloc((size_t)n*sizeof(long double));
    int status=1,rc=CHEAP_ENOMEM;
    const char *stage="allocation";
    if(!col||!b||!out||!ref||!coeff)goto fail;
    for(int i=0;i<n;++i){col[i]=pow(rho,i);b[i]=sin(.1*i)+.5;}
    stage="DCT setup";
    double t=clock_ms();
    if((rc=cheap_init_from_toeplitz_projection(&ctx,n,col))!=CHEAP_OK)goto fail;
    if((rc=cheap_pcg_dct_init(&pre,&ctx,ctx.lambda))!=CHEAP_OK)goto fail;
    double setup=clock_ms()-t;
    references(n,b,pre.inverse_weights,ref,coeff);
    fputs("backend,n,rho,operation,phase,repetition,elapsed_ms,relative_error,result\n",log);
    puts("backend,n,rho,operation,planner,repetitions,setup_ms,first_ms,median_ms,p95_ms,iqr_ms,max_relative_error,peak_bytes,result");
    status=0;
    for(int op=0;op<7;++op) {
        double times[RUNS],first=0,max_error=0;
        int failed=0;
        for(int k=-1;k<RUNS;++k) {
            /* Reset is untimed and prevents repeated transform overflow. */
            if(op>=2)memcpy(ctx.workspace,b,(size_t)n*sizeof(double));
            t=clock_ms();
            rc=execute(op,&ctx,&pre,b,out);
            double elapsed=clock_ms()-t;
            double error=relative_error(n,(op>=2&&op<=5)?ctx.workspace:out,ref+op*n);
            int accepted=rc==CHEAP_OK&&isfinite(error)&&error<=ERROR_LIMIT;
            if(!accepted){failed=1;status=1;}
            if(error>max_error)max_error=error;
            if(k<0)first=elapsed;else times[k]=elapsed;
            fprintf(log,"%s,%d,%.17g,%s,%s,%d,%.12g,%.12g,%d\n",
                CHEAP_BENCH_BACKEND,n,rho,names[op],k<0?"first":"steady",k,
                elapsed,error,accepted?0:1);
            sink+=(op>=2&&op<=5)?ctx.workspace[(k+1)%n]:out[(k+1)%n];
        }
        qsort(times,RUNS,sizeof(double),cmp);
        printf("%s,%d,%.17g,%s,FFTW_ESTIMATE,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%ld,%d\n",
            CHEAP_BENCH_BACKEND,n,rho,names[op],RUNS,setup,first,
            .5*(times[19]+times[20]),times[37],times[29]-times[9],max_error,peak_bytes(),failed);
    }
    /* Inspect selected plans after timing; never print or plan in execute. */
    fprintf(plans,"backend=%s n=%d rho=%.17g planner=FFTW_ESTIMATE\n",
            CHEAP_BENCH_BACKEND,n,rho);
#if defined(CHEAP_USE_NOTORIOUS_FFT)
    fprintf(plans,"backend_version=%s\n",NOTORIOUS_FFT_VERSION_STRING);
#endif
    dump_plan(plans,"DCT-II/REDFT10",ctx.plan_fwd);
    dump_plan(plans,"DCT-III/REDFT01",ctx.plan_inv);
    goto cleanup;
fail:
    fprintf(stderr,"%s failed: n=%d rho=%.17g status=%d\n",stage,n,rho,rc);
cleanup:
    cheap_pcg_dct_destroy(&pre);cheap_destroy(&ctx);
    free(col);free(b);free(out);free(ref);free(coeff);
    if(fclose(log)!=0){perror("sample log");status=1;}
    if(fclose(plans)!=0){perror("plan log");status=1;}
    if(fflush(stdout)!=0)status=1;
    return status;
usage:
    fprintf(stderr,"usage: %s N OUTPUT_PREFIX [rho]\nN: 2..4096; rho: finite 0 <= rho < 1, default 0.99\n",argv[0]);
    return 2;
}
