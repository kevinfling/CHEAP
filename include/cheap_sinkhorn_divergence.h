/* Debiased entropic OT divergence on discrete measures.
 * Unlike the former 4-vector squared-distance proxy, all three
 * regularized transport problems are solved explicitly.
 * Costs: C_ab, C_aa, C_bb are row-major n*n; measures strictly positive,
 * equal total mass. Workspace: f[n], g[n], plan[n*n]. MIT.
 */
#ifndef CHEAP_SINKHORN_DIVERGENCE_H
#define CHEAP_SINKHORN_DIVERGENCE_H
#include "cheap_transport.h"
static inline int cheap_sinkhorn_divergence(
    int n, const double *a, const double *b,
    const double *cab, const double *caa, const double *cbb,
    double epsilon, int max_iter, double tol,
    double *f, double *g, double *plan, double *out)
{
    if(!out||!cab||!caa||!cbb||!plan)return CHEAP_EINVAL;
    double ab,aa,bb;
    int rc=cheap_transport_sinkhorn(n,a,b,cab,epsilon,max_iter,tol,f,g,plan);
    if(rc!=CHEAP_OK)return rc;
    rc=cheap_transport_cost(n,cab,plan,epsilon,&ab);
    if(rc!=CHEAP_OK)return rc;
    rc=cheap_transport_sinkhorn(n,a,a,caa,epsilon,max_iter,tol,f,g,plan);
    if(rc!=CHEAP_OK)return rc;
    rc=cheap_transport_cost(n,caa,plan,epsilon,&aa);
    if(rc!=CHEAP_OK)return rc;
    rc=cheap_transport_sinkhorn(n,b,b,cbb,epsilon,max_iter,tol,f,g,plan);
    if(rc!=CHEAP_OK)return rc;
    rc=cheap_transport_cost(n,cbb,plan,epsilon,&bb);
    if(rc!=CHEAP_OK)return rc;
    *out=ab-0.5*(aa+bb);
    return isfinite(*out)?CHEAP_OK:CHEAP_EDOM;
}
/* Richardson extrapolation is heuristic unless a first-order
 * regularization-error expansion has been established for the problem.
 */
static inline int cheap_sinkhorn_richardson(
    int n,const double *a,const double *b,
    const double *cab,const double *caa,const double *cbb,
    double epsilon,int max_iter,double tol,
    double *f,double *g,double *plan,double *out)
{
    if(!out || !isfinite(epsilon) || epsilon<=0)return CHEAP_EINVAL;
    double full,half;
    int rc=cheap_sinkhorn_divergence(n,a,b,cab,caa,cbb,epsilon,max_iter,
                                     tol,f,g,plan,&full);
    if(rc!=CHEAP_OK)return rc;
    rc=cheap_sinkhorn_divergence(n,a,b,cab,caa,cbb,0.5*epsilon,max_iter,
                                 tol,f,g,plan,&half);
    if(rc!=CHEAP_OK)return rc;
    *out=2.0*half-full;
    return isfinite(*out)?CHEAP_OK:CHEAP_EDOM;
}
#endif
