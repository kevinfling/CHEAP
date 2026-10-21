/* Independent dense verification of the Toeplitz DCT projection diagonal.
 * cc -std=c99 -O2 -Wall -Wextra -pedantic -Iinclude \
 *    tests/test_toeplitz_dct_diagonal.c -lm -o /tmp/test_proj
 */
#include <math.h>
#include <stdio.h>
#include "../include/cheap_toeplitz_dct_diagonal.h"
#define PI 3.14159265358979323846
int main(void) {
    for (size_t n=3;n<=128;n++) {
        double c[128],w[128];
        for(size_t d=0;d<n;d++) c[d]=pow(0.9,(double)d);
        if(cheap_toeplitz_dct_projection_diagonal(n,c,w)) return 1;
        for(size_t k=0;k<n;k++) {
            double scale=sqrt((k==0?1.0:2.0)/(double)n);
            double ref=0;
            for(size_t i=0;i<n;i++) for(size_t j=0;j<n;j++)
                ref+=scale*scale*cos(PI*(double)k*((double)i+0.5)/(double)n)
                    *c[i>j?i-j:j-i]
                    *cos(PI*(double)k*((double)j+0.5)/(double)n);
            double tol=1e-9*fmax(1.0,fabs(ref));
            if(fabs(w[k]-ref)>tol) {
                fprintf(stderr,"FAIL n=%zu k=%zu got=%.17g ref=%.17g\n",n,k,w[k],ref);
                return 2;
            }
        }
    }
    puts("PASS: analytic Toeplitz DCT projection diagonal matches dense reference");
    return 0;
}
