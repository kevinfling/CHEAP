/* CHEAP dense entropic optimal transport: strict C99, no FFT assumptions.
 * Costs C[i*n+j], positive equal-mass marginals a,b. All values finite.
 * f,g are logarithms of scaling vectors, NOT epsilon-scaled dual potentials.
 * Caller provides n*n cost and n*n optional transport output; O(n^2) per sweep.
 * This is an exact Sinkhorn iteration for K_ij=exp(-C_ij/epsilon), not a
 * DCT operator exponential. MIT.
 */
#ifndef CHEAP_TRANSPORT_H
#define CHEAP_TRANSPORT_H
#include <math.h>
#include <stddef.h>
#include "cheap.h"
static inline int cheap_transport_sinkhorn(
    int n, const double *a, const double *b, const double *cost,
    double epsilon, int max_iter, double tol,
    double *f, double *g, double *plan_out)
{
    if (n < 1 || !a || !b || !cost || !f || !g ||
        !isfinite(epsilon) || epsilon <= 0 ||
        max_iter < 1 || !isfinite(tol) || tol <= 0) return CHEAP_EINVAL;
    double sa=0, sb=0;
    for (int i=0;i<n;++i) {
        if (!isfinite(a[i]) || !isfinite(b[i]) || a[i]<=0 || b[i]<=0)
            return CHEAP_EDOM;
        sa+=a[i]; sb+=b[i];
        for (int j=0;j<n;++j)
            if (!isfinite(cost[(size_t)i*n+j])) return CHEAP_EDOM;
    }
    if (!isfinite(sa) || !isfinite(sb) ||
        fabs(sa-sb)>1e-12*fmax(sa,sb)) return CHEAP_EINVAL;
    for (int i=0;i<n;++i) f[i]=g[i]=0;
    for (int iter=0;iter<max_iter;++iter) {
        for (int i=0;i<n;++i) {
            double mx=-INFINITY;
            for (int j=0;j<n;++j) {
                double z=-cost[(size_t)i*n+j]/epsilon+g[j];
                if (z>mx) mx=z;
            }
            if (!isfinite(mx)) return CHEAP_EDOM;
            double sum=0;
            for (int j=0;j<n;++j)
                sum+=exp(-cost[(size_t)i*n+j]/epsilon+g[j]-mx);
            f[i]=log(a[i])-mx-log(sum);
        }
        for (int j=0;j<n;++j) {
            double mx=-INFINITY;
            for (int i=0;i<n;++i) {
                double z=-cost[(size_t)i*n+j]/epsilon+f[i];
                if (z>mx) mx=z;
            }
            if (!isfinite(mx)) return CHEAP_EDOM;
            double sum=0;
            for (int i=0;i<n;++i)
                sum+=exp(-cost[(size_t)i*n+j]/epsilon+f[i]-mx);
            g[j]=log(b[j])-mx-log(sum);
        }
        /* Check both row and column marginal errors, not just dual drift. */
        double err=0;
        for (int i=0;i<n;++i) {
            double mx=-INFINITY;
            for (int j=0;j<n;++j) {
                double z=-cost[(size_t)i*n+j]/epsilon+f[i]+g[j];
                if (z>mx) mx=z;
            }
            double sum=0;
            for (int j=0;j<n;++j)
                sum+=exp(-cost[(size_t)i*n+j]/epsilon+f[i]+g[j]-mx);
            double row=exp(mx)*sum;
            if (!isfinite(row)) return CHEAP_EDOM;
            double e=fabs(row-a[i]); if(e>err)err=e;
        }
        for (int j=0;j<n;++j) {
            double mx=-INFINITY;
            for (int i=0;i<n;++i) {
                double z=-cost[(size_t)i*n+j]/epsilon+f[i]+g[j];
                if(z>mx)mx=z;
            }
            double sum=0;
            for (int i=0;i<n;++i)
                sum+=exp(-cost[(size_t)i*n+j]/epsilon+f[i]+g[j]-mx);
            double col=exp(mx)*sum;
            if (!isfinite(col)) return CHEAP_EDOM;
            double e=fabs(col-b[j]); if(e>err)err=e;
        }
        if(err<=tol*fmax(1.0,sa)) {
            if(plan_out) for(int i=0;i<n;++i)for(int j=0;j<n;++j)
                plan_out[(size_t)i*n+j]=exp(-cost[(size_t)i*n+j]/epsilon+f[i]+g[j]);
            return CHEAP_OK;
        }
    }
    return CHEAP_ENOCONV;
}
/* Regularized objective: <P,C> + eps*sum(P*(log P-1)).
 * This includes the -eps*mass constant; debiased divergence cancels it.
 */
static inline int cheap_transport_cost(
    int n,const double *cost,const double *plan,double eps,double *value)
{
    if(n<1||!cost||!plan||!value||!isfinite(eps)||eps<=0)return CHEAP_EINVAL;
    double acc=0;
    for(int i=0;i<n*n;++i) {
        double p=plan[i];
        if(!isfinite(cost[i])||!isfinite(p)||p<0)return CHEAP_EDOM;
        if(p>0)acc+=p*cost[i]+eps*p*(log(p)-1.0);
    }
    if(!isfinite(acc))return CHEAP_EDOM;
    *value=acc;return CHEAP_OK;
}
#endif
