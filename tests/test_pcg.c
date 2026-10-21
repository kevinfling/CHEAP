#include "cheap_pcg.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
typedef struct {int n;double *a;} dense_ctx;
static int dense_apply(void *v,const double *x,double *y) {
    dense_ctx *d=(dense_ctx*)v;
    for(int i=0;i<d->n;++i) {
        double sum=0;
        for(int j=0;j<d->n;++j)sum+=d->a[i*d->n+j]*x[j];
        y[i]=sum;
    }
    return CHEAP_OK;
}
static int direct(int n,double *a,const double *b,double *x) {
    double *m=(double*)malloc((size_t)n*(size_t)n*sizeof(double));
    double *rhs=(double*)malloc((size_t)n*sizeof(double));
    if(!m||!rhs){free(m);free(rhs);return 1;}
    memcpy(m,a,(size_t)n*(size_t)n*sizeof(double));
    memcpy(rhs,b,(size_t)n*sizeof(double));
    for(int k=0;k<n;++k) {
        int pivot=k;
        for(int i=k+1;i<n;++i)
            if(fabs(m[i*n+k])>fabs(m[pivot*n+k]))pivot=i;
        if(fabs(m[pivot*n+k])<1e-14){free(m);free(rhs);return 1;}
        for(int j=k;j<n;++j) {
            double t=m[k*n+j];m[k*n+j]=m[pivot*n+j];m[pivot*n+j]=t;
        }
        double t=rhs[k];rhs[k]=rhs[pivot];rhs[pivot]=t;
        for(int i=k+1;i<n;++i) {
            double f=m[i*n+k]/m[k*n+k];
            for(int j=k+1;j<n;++j)m[i*n+j]-=f*m[k*n+j];
            rhs[i]-=f*rhs[k];
        }
    }
    for(int i=n-1;i>=0;--i) {
        double t=rhs[i];
        for(int j=i+1;j<n;++j)t-=m[i*n+j]*x[j];
        x[i]=t/m[i*n+i];
    }
    free(m);free(rhs);return 0;
}
static int one(int n,int toeplitz,int preconditioned) {
    double *a=calloc((size_t)n*(size_t)n,sizeof(double));
    double *c=calloc((size_t)n,sizeof(double));
    double *b=calloc((size_t)n,sizeof(double));
    double *x=calloc((size_t)n,sizeof(double));
    double *expected=calloc((size_t)n,sizeof(double));
    cheap_pcg_workspace w={0};
    cheap_toeplitz_operator t={0};
    cheap_ctx dct={0};
    cheap_pcg_dct_preconditioner pre={0};
    cheap_linear_operator A={0},M={0};
    dense_ctx d={n,a};
    int fail=1;
    if(!a||!c||!b||!x||!expected)goto end;
    for(int i=0;i<n;++i) c[i]=pow(.15,(double)i);
    for(int i=0;i<n;++i) {
        b[i]=sin(i+1.)+0.5;
        for(int j=0;j<n;++j)
            a[i*n+j]=toeplitz?c[abs(i-j)]:(i==j?2.+i*.1:0.);
    }
    if(cheap_pcg_workspace_init(&w,n)!=CHEAP_OK)goto end;
    if(toeplitz) {
        if(cheap_toeplitz_operator_init(&t,n,c)!=CHEAP_OK)goto end;
        A=cheap_toeplitz_as_linear_operator(&t);
    }else {A.n=n;A.context=&d;A.apply=dense_apply;}
    if(preconditioned) {
        double *weights=calloc((size_t)n,sizeof(double));
        if(!weights)goto end;
        for(int i=0;i<n;++i)weights[i]=1.;
        /* A positive DCT diagonal is SPD, even if it is a crude M. */
        int rc=cheap_init(&dct,n,0.5);
        if(rc==CHEAP_OK)rc=cheap_pcg_dct_init(&pre,&dct,weights);
        free(weights);
        if(rc!=CHEAP_OK)goto end;
        M=cheap_pcg_dct_as_operator(&pre);
    }
    if(direct(n,a,b,expected))goto end;
    cheap_pcg_result result=cheap_pcg_solve(&A,preconditioned?&M:NULL,b,x,&w,
                                            1e-11,1e-10,4*n+20,7);
    if(result.reason!=CHEAP_PCG_CONVERGED) {
        fprintf(stderr,"PCG n=%d toeplitz=%d pre=%d reason=%d iter=%d residual=%g\n",
                n,toeplitz,preconditioned,result.reason,result.iterations,result.true_residual);
        goto end;
    }
    double err=0;
    for(int i=0;i<n;++i) {
        double e=fabs(x[i]-expected[i]);
        if(e>err)err=e;
    }
    if(err>2e-8 || result.operator_calls<2 || !isfinite(result.true_residual)) {
        fprintf(stderr,"PCG dense mismatch n=%d err=%g\n",n,err);goto end;
    }
    memset(b,0,(size_t)n*sizeof(double));
    memset(x,0,(size_t)n*sizeof(double));
    result=cheap_pcg_solve(&A,NULL,b,x,&w,0,1e-10,5,2);
    if(result.reason!=CHEAP_PCG_CONVERGED || result.true_residual!=0 ||
       result.relative_residual!=0)goto end;
    fail=0;
end:
    cheap_pcg_dct_destroy(&pre);
    cheap_destroy(&dct);
    cheap_toeplitz_operator_destroy(&t);
    cheap_pcg_workspace_destroy(&w);
    free(a);free(c);free(b);free(x);free(expected);
    return fail;
}
int main(void) {
    static const int sizes[]={1,2,3,5,8,17,64};
    for(int k=0;k<(int)(sizeof(sizes)/sizeof(sizes[0]));++k)
        for(int t=0;t<2;++t)
            for(int p=0;p<(sizes[k]==1?1:2);++p)
                if(one(sizes[k],t,p))return 1;
    puts("PASS: true-residual PCG against dense direct reference");
    return 0;
}
