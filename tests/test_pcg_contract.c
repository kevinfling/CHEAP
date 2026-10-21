#include "cheap_pcg.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
typedef struct {int n,kind,calls;} op_ctx;
static int apply(void *v,const double *x,double *y) {
    op_ctx *o=(op_ctx*)v;++o->calls;
    if(o->kind==4)return CHEAP_EDOM;
    for(int i=0;i<o->n;++i) {
        if(o->kind==1)y[i]=0;
        else if(o->kind==2)y[i]=NAN;
        else if(o->kind==3)y[i]=-x[i];
        else {
            y[i]=2*x[i];
            if(i>0)y[i]-=x[i-1];
            if(i+1<o->n)y[i]-=x[i+1];
        }
    }
    return CHEAP_OK;
}
static int bad_precond(void *v,const double *x,double *y) {
    (void)v;(void)x;(void)y;return CHEAP_EDOM;
}
static int check(int n) {
    cheap_pcg_workspace w={0};
    op_ctx ctx={n,0,0};
    cheap_linear_operator A={n,&ctx,apply},M={n,&ctx,bad_precond};
    double *b=calloc((size_t)n,sizeof(double));
    double *x=calloc((size_t)n,sizeof(double));
    double *truth=calloc((size_t)n,sizeof(double));
    cheap_pcg_options opts={6*n+100,1e-10,1e-10,0,1,CHEAP_PCG_REFRESH_CONTINUE};
    int fail=1;
    if(!b||!x||!truth||cheap_pcg_workspace_init(&w,n)!=CHEAP_OK)goto done;
    for(int i=0;i<n;++i)truth[i]=sin((double)i*.1)+1;
    if(apply(&ctx,truth,b)!=CHEAP_OK)goto done;
    cheap_pcg_result r=cheap_pcg_solve_opts(&A,NULL,b,x,&w,&opts);
    if(r.reason!=CHEAP_PCG_CONVERGED||!r.residual_valid||
       r.replacement_policy!=CHEAP_PCG_REFRESH_CONTINUE) {
        fprintf(stderr,"n=%d solve reason=%d iter=%d residual=%g\n",
                n,r.reason,r.iterations,r.true_residual);goto done;
    }
    for(int i=0;i<n;++i)
        if(fabs(x[i]-truth[i])>3e-7)goto done;
    double *fresh=calloc((size_t)n,sizeof(double));
    if(!fresh)goto done;
    if(apply(&ctx,x,fresh)!=CHEAP_OK){free(fresh);goto done;}
    long double sum=0;
    for(int i=0;i<n;++i) {
        long double diff=(long double)b[i]-fresh[i];
        sum+=diff*diff;
    }
    free(fresh);
    if(fabs(r.true_residual-(double)sqrtl(sum))>1e-9)goto done;
    opts.zero_initial_guess=0;
    r=cheap_pcg_solve_opts(&A,NULL,b,x,&w,&opts);
    if(r.reason!=CHEAP_PCG_CONVERGED||r.iterations!=0)goto done;
    opts.zero_initial_guess=1;opts.max_iterations=0;
    r=cheap_pcg_solve_opts(&A,NULL,b,x,&w,&opts);
    if(r.reason!=CHEAP_PCG_MAX_ITER||!r.residual_valid)goto done;
    opts.max_iterations=6*n+100;
    r=cheap_pcg_solve_opts(&A,&M,b,x,&w,&opts);
    if(r.reason!=CHEAP_PCG_PRECOND_INVALID||!r.residual_valid)goto done;
    ctx.kind=1;
    r=cheap_pcg_solve_opts(&A,NULL,b,x,&w,&opts);
    if(r.reason!=CHEAP_PCG_BREAKDOWN_CURVATURE||!r.residual_valid)goto done;
    ctx.kind=2;
    r=cheap_pcg_solve_opts(&A,NULL,b,x,&w,&opts);
    if(r.reason!=CHEAP_PCG_BREAKDOWN_NONFINITE||r.residual_valid||
       !isnan(r.true_residual))goto done;
    ctx.kind=4;
    r=cheap_pcg_solve_opts(&A,NULL,b,x,&w,&opts);
    if(r.reason!=CHEAP_PCG_CALLBACK_FAILED||r.residual_valid||
       !isnan(r.true_residual))goto done;
    ctx.kind=0;
    memset(b,0,(size_t)n*sizeof(double));
    r=cheap_pcg_solve_opts(&A,NULL,b,x,&w,&opts);
    if(r.reason!=CHEAP_PCG_CONVERGED||r.iterations!=0||
       r.relative_residual!=0)goto done;
    fail=0;
done:
    cheap_pcg_workspace_destroy(&w);
    cheap_pcg_workspace_destroy(&w);
    free(b);free(x);free(truth);
    return fail;
}
int main(void) {
    const int sizes[]={1,2,3,5,17,64,257,512};
    for(int i=0;i<8;++i)
        if(check(sizes[i])) {
            fprintf(stderr,"PCG contract failed n=%d\n",sizes[i]);return 1;
        }
    puts("PASS: PCG contract statuses, fresh residuals and awkward sizes");
    return 0;
}
