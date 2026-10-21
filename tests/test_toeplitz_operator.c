/* Dense independent reference for exact Toeplitz matvec, C99. */
#include "cheap_toeplitz_operator.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void dense(int n, const double *c, const double *x, double *y) {
    int i, j;
    for (i=0;i<n;++i) {
        long double sum=0;
        for (j=0;j<n;++j) {
            int d=i>j?i-j:j-i;
            sum+=(long double)c[d]*(long double)x[j];
        }
        y[i]=(double)sum;
    }
}
static int check(int n, int variant) {
    double *c=calloc((size_t)n,sizeof(double)), *x=calloc((size_t)n,sizeof(double));
    double *y=calloc((size_t)n,sizeof(double)), *reference=calloc((size_t)n,sizeof(double));
    cheap_toeplitz_operator op={0};
    cheap_linear_operator linear;
    int i, rc, fail=0;
    if(!c||!x||!y||!reference) { fail=1; goto end; }
    for(i=0;i<n;++i) c[i]=(i%2?-1.:1.)/(1.+i*.125);
    for(i=0;i<n;++i) {
        if(variant==0) x[i]=(i==n/2)?1.:0.;
        else if(variant==1) x[i]=1.;
        else if(variant==2) x[i]=i%2?-1.:1.;
        else x[i]=(i%2?-1.:1.)*(1.+1e-9*i);
    }
    rc=cheap_toeplitz_operator_init(&op,n,c);
    if(rc!=CHEAP_OK) { fprintf(stderr,"init n=%d rc=%d\n",n,rc);fail=1;goto end; }
    if(op.fft_length<2*n-1 || (op.fft_length&(op.fft_length-1))) {
        fprintf(stderr,"embedding length invalid n=%d\n",n);fail=1;goto end;
    }
    linear=cheap_toeplitz_as_linear_operator(&op);
    dense(n,c,x,reference);
    for(int round=0;round<3;++round) {
        double scale=0;
        for(i=0;i<n;++i) scale+=fabs(c[i]);
        for(i=0;i<n;++i) {
            double tol=2e-11*(1.+log2((double)op.fft_length))*(
                fabs(reference[i])+scale*(double)n+1.);
            (void)tol;
        }
        rc=cheap_linear_operator_apply(&linear,x,y);
        if(rc!=CHEAP_OK) {fprintf(stderr,"apply n=%d rc=%d\n",n,rc);fail=1;goto end;}
        for(i=0;i<n;++i) {
            /* Absolute term handles cancellation; relative term handles scale. */
            double absbound=0;
            for(int j=0;j<n;++j) absbound+=fabs(c[i>j?i-j:j-i]*x[j]);
            double tol=5e-12*(1.+log2((double)op.fft_length))*
                       (absbound+fabs(reference[i])+1e-15);
            if(fabs(y[i]-reference[i])>tol) {
                fprintf(stderr,"n=%d variant=%d i=%d actual=%.17g expected=%.17g tol=%g\n",
                        n,variant,i,y[i],reference[i],tol);
                fail=1;goto end;
            }
        }
    }
    /* In-place application is supported and uses the internal scratch. */
    memcpy(y,x,(size_t)n*sizeof(double));
    if(cheap_toeplitz_operator_apply(&op,y,y)!=CHEAP_OK) {fail=1;goto end;}
    for(i=0;i<n;++i) {
        double absbound=0;
        for(int j=0;j<n;++j) absbound+=fabs(c[i>j?i-j:j-i]*x[j]);
        if(fabs(y[i]-reference[i])>5e-12*(1.+log2((double)op.fft_length))*
           (absbound+fabs(reference[i])+1e-15)) {fail=1;goto end;}
    }
    /* Reject invalid input without touching destination. */
    x[0]=NAN;
    y[0]=12345.;
    if(cheap_toeplitz_operator_apply(&op,x,y)!=CHEAP_EDOM || y[0]!=12345.) {
        fail=1;goto end;
    }
    if(n>1) {
        x[0]=1.;
        if(cheap_toeplitz_operator_apply(&op,x,x+1)!=CHEAP_EINVAL) {
            fail=1;goto end;
        }
    }
end:
    cheap_toeplitz_operator_destroy(&op);
    cheap_toeplitz_operator_destroy(&op);
    free(c);free(x);free(y);free(reference);
    return fail;
}
int main(void) {
    static const int sizes[]={1,2,3,5,8,17,64,257};
    int i,j;
    cheap_toeplitz_operator a={0},b={0},bad={0};
    double c[3]={2.,-.25,.125}, x[3]={1.,2.,3.},ya[3],yb[3];
    double nan_col[3]={2.,NAN,1.};
    for(i=0;i<(int)(sizeof(sizes)/sizeof(sizes[0]));++i)
        for(j=0;j<4;++j) if(check(sizes[i],j)) return 1;
    if(cheap_toeplitz_operator_init(&bad,3,nan_col)!=CHEAP_EDOM ||
       bad.time || bad.forward) return 2;
    if(cheap_toeplitz_operator_init(&bad,INT_MAX,c)!=CHEAP_EINVAL) return 3;
    if(cheap_toeplitz_operator_init(&a,3,c)!=CHEAP_OK ||
       cheap_toeplitz_operator_init(&b,3,c)!=CHEAP_OK) return 4;
    if(cheap_toeplitz_operator_apply(&a,x,ya)!=CHEAP_OK ||
       cheap_toeplitz_operator_apply(&b,x,yb)!=CHEAP_OK) return 5;
    for(i=0;i<3;++i) if(fabs(ya[i]-yb[i])>1e-13) return 6;
    cheap_toeplitz_operator_destroy(&a);
    if(cheap_toeplitz_operator_apply(&b,x,yb)!=CHEAP_OK) return 7;
    cheap_toeplitz_operator_destroy(&b);
    puts("PASS: exact Toeplitz FFT matvec matches independent dense reference");
    return 0;
}
