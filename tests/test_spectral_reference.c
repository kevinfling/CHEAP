/* Standalone C99 reference tests for CHEAP's transform-diagonal model.
 * No FFTW dependency; these tests deliberately use dense DCT matrices.
 * cc -std=c99 -O2 -Wall -Wextra -pedantic tests/test_spectral_reference.c -lm -o /tmp/cheap_ref && /tmp/cheap_ref
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define PI 3.14159265358979323846264338327950288
#define NMAX 32
static void dct(int n, double q[NMAX][NMAX]) {
    for(int k=0;k<n;k++) for(int j=0;j<n;j++)
        q[k][j]=sqrt((k==0?1.0:2.0)/n)*cos(PI*k*(j+0.5)/n);
}
static double normf(int n,double a[NMAX][NMAX]){
    double s=0;for(int i=0;i<n;i++)for(int j=0;j<n;j++)s+=a[i][j]*a[i][j];return sqrt(s);
}
int main(void) {
    for(int n=3;n<=16;n++){
        double q[NMAX][NMAX]={{0}},a[NMAX][NMAX]={{0}},s[NMAX][NMAX]={{0}};
        double b[NMAX][NMAX]={{0}},lap[NMAX][NMAX]={{0}};
        dct(n,q);
        for(int i=0;i<n;i++)for(int j=0;j<n;j++){
            double v=0;for(int k=0;k<n;k++)v+=q[k][i]*q[k][j];
            if(fabs(v-(i==j))>1e-12)return fprintf(stderr,"orthogonality N=%d\n",n),1;
        }
        for(int i=0;i<n;i++){
            lap[i][i]=(i==0||i==n-1)?1:2;
            if(i+1<n)lap[i][i+1]=lap[i+1][i]=-1;
        }
        for(int i=0;i<n;i++)for(int j=0;j<n;j++){
            double v=0;for(int k=0;k<n;k++){
                double mu=4*pow(sin(PI*k/(2*n)),2);
                v+=q[k][i]*mu*q[k][j];
            }
            if(fabs(v-lap[i][j])>1e-11)return fprintf(stderr,"laplacian N=%d\n",n),1;
        }
        /* Symmetric Toeplitz, deliberately not DCT diagonal in general. */
        for(int i=0;i<n;i++)for(int j=0;j<n;j++)
            a[i][j]=pow(0.67,abs(i-j))+0.1*(i==j);
        for(int k=0;k<n;k++)for(int l=0;l<n;l++)
            for(int i=0;i<n;i++)for(int j=0;j<n;j++)
                s[k][l]+=q[k][i]*a[i][j]*q[l][j];
        for(int i=0;i<n;i++)for(int j=0;j<n;j++)
            for(int k=0;k<n;k++)b[i][j]+=q[k][i]*s[k][k]*q[k][j];
        double e[NMAX][NMAX]={{0}},off=0;
        for(int i=0;i<n;i++)for(int j=0;j<n;j++){
            e[i][j]=a[i][j]-b[i][j];
            if(i!=j)off+=s[i][j]*s[i][j];
        }
        if(fabs(normf(n,e)*normf(n,e)-off)>1e-9)
            return fprintf(stderr,"projection identity N=%d\n",n),1;
        if(n>=4 && off<1e-12)
            return fprintf(stderr,"unexpected exact Toeplitz diagonalization N=%d\n",n),1;
        /* The direction of the implicit Euler inequality is important. */
        for(int k=0;k<n;k++){
            double x=4*pow(sin(PI*k/(2*n)),2);
            if(exp(-x)>1/(1+x)+1e-14)return 1;
        }
    }
    puts("PASS: orthonormal DCT, Neumann spectrum, projection identity, Toeplitz counterexample, heat inequality");
    return 0;
}
