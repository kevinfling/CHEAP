#ifndef CHEAP_TOEPLITZ_DCT_DIAGONAL_H
#define CHEAP_TOEPLITZ_DCT_DIAGONAL_H
/* Exact Frobenius-optimal orthonormal DCT-II diagonal of a real symmetric
 * Toeplitz matrix A[i,j]=column[abs(i-j)]. O(N^2) time, O(1) extra space.
 * The result is diag(Q A Q^T), NOT the eigenvalues of A.
 * Include <math.h> before use. Header-only C99.
 */
#include <math.h>
#include <stddef.h>
#ifndef CHEAP_TOEPLITZ_PI
#define CHEAP_TOEPLITZ_PI 3.14159265358979323846264338327950288
#endif
static inline int cheap_toeplitz_dct_projection_diagonal(
    size_t n, const double *column, double *weights)
{
    if (!column || !weights || n < 2) return -1;
    for (size_t d=0; d<n; ++d)
        if (!isfinite(column[d])) return -1;
    double dc=column[0];
    for (size_t d=1; d<n; ++d)
        dc += 2.0 * (double)(n-d)/(double)n * column[d];
    weights[0]=dc;
    for (size_t k=1; k<n; ++k) {
        double theta=CHEAP_TOEPLITZ_PI*(double)k/(double)n;
        double den=sin(theta);
        double sum=column[0];
        for (size_t d=1; d<n; ++d) {
            double angle=theta*(double)d;
            sum += 2.0/(double)n*column[d]*
                ((double)(n-d)*cos(angle)-sin(angle)/den);
        }
        weights[k]=sum;
    }
    return 0;
}
#endif
