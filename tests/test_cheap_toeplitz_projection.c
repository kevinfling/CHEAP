/* Dense-reference and error-contract tests for the corrected Toeplitz
 * projection initializer. C99; link FFTW3 and libm.
 */
#include "cheap.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
static int check(int n) {
    double t[64]; cheap_ctx ctx;
    for (int d=0; d<n; ++d) t[d]=pow(.8,(double)d);
    int rc=cheap_init_from_toeplitz_projection(&ctx,n,t);
    if(rc!=CHEAP_OK) {fprintf(stderr,"init rc=%d\n",rc);return 1;}
    for(int k=0;k<n;k++) {
        double sum=0,scale=sqrt((k==0?1.:2.)/n);
        for(int i=0;i<n;i++)for(int j=0;j<n;j++)
            sum+=scale*scale*cos(M_PI*k*(i+.5)/n)*
                t[i>j?i-j:j-i]*cos(M_PI*k*(j+.5)/n);
        if(fabs(ctx.lambda[k]-sum)>1e-9*fmax(1.,fabs(sum))) {
            fprintf(stderr,"n=%d k=%d projected=%g dense=%g\n",
                    n,k,ctx.lambda[k],sum);
            cheap_destroy(&ctx);return 2;
        }
    }
    cheap_destroy(&ctx);return 0;
}
int main(void) {
    for(int n=2;n<=64;n++)if(check(n))return 1;
    cheap_toeplitz_projection_workspace ws;
    if (cheap_toeplitz_projection_workspace_init(&ws, 32)!=CHEAP_OK) return 4;
    {
        double col[32], weights[32], bad[32];
        cheap_ctx reference;
        for (int d=0;d<32;d++) col[d]=pow(.8,(double)d);
        if (cheap_init_from_toeplitz_projection(&reference,32,col)!=CHEAP_OK) return 5;
        if (cheap_toeplitz_projection_with_workspace(&ws,col,weights)!=CHEAP_OK) return 6;
        for (int k=0;k<32;k++)
            if (fabs(weights[k]-reference.lambda[k])>1e-10) return 7;
        for (int round=0; round<12; ++round) {
            col[1]=pow(.8 + .001*round,1.);
            if (cheap_toeplitz_projection_with_workspace(&ws,col,weights)!=CHEAP_OK)
                return 10;
            cheap_ctx fresh;
            if (cheap_init_from_toeplitz_projection(&fresh,32,col)!=CHEAP_OK)
                return 11;
            for (int k=0;k<32;k++) {
                if (fabs(weights[k]-fresh.lambda[k])>1e-10) return 12;
            }
            cheap_destroy(&fresh);
        }
        memcpy(bad,col,sizeof(bad));
        bad[4]=NAN;
        weights[0]=12345.0;
        if (cheap_toeplitz_projection_with_workspace(&ws,bad,weights)!=CHEAP_EDOM)
            return 8;
        if (weights[0]!=12345.0) return 9;
        cheap_destroy(&reference);
    }
    cheap_toeplitz_projection_workspace_destroy(&ws);
    /* Explicitly test that a zeroed workspace can be safely destroyed. */
    cheap_toeplitz_projection_workspace ws_empty = {0};
    cheap_toeplitz_projection_workspace_destroy(&ws_empty);
    double invalid[4]={1.,.2,NAN,.1};cheap_ctx ctx;
    if(cheap_init_from_toeplitz_projection(&ctx,4,invalid)!=CHEAP_EDOM)return 2;
    if(cheap_init_from_toeplitz_projection(NULL,4,invalid)!=CHEAP_EINVAL)return 3;
    puts("PASS: Toeplitz DCT projection matches dense reference and rejects NaN");
    return 0;
}
