/* Gumbel-perturbed entropic transport, strict C99.
 * Perturbs every cost entry independently, then solves the perturbed
 * entropic OT problem. This is a stochastic relaxation, NOT an exact
 * Gumbel-matching sampler and not a Gaussian-copula field.
 * Caller provides n*n scratch; no allocation in the hot path.
 * MIT.
 */
#ifndef CHEAP_GUMBEL_TRANSPORT_H
#define CHEAP_GUMBEL_TRANSPORT_H
#include <stdint.h>
#include <math.h>
#include <stddef.h>
#include "cheap_transport.h"
static inline uint64_t cheap_gumbel_next(uint64_t *state)
{
    uint64_t z=(*state+=UINT64_C(0x9e3779b97f4a7c15));
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    return z^(z>>31);
}
static inline double cheap_gumbel_standard(uint64_t *state)
{
    /* Midpoint uniform in (0,1): no log(0) or ad hoc clipping. */
    double u=((double)(cheap_gumbel_next(state)>>11)+0.5)
             * (1.0/9007199254740992.0);
    return -log(-log(u));
}
static inline int cheap_gumbel_transport(
    int n, const double *a, const double *b, const double *cost,
    double epsilon, double perturbation_scale, uint64_t seed,
    int max_iter, double tol, double *f, double *g,
    double *cost_scratch, double *plan_out)
{
    if(n<1 || !cost || !cost_scratch || !isfinite(perturbation_scale) ||
       perturbation_scale<0 || cost==cost_scratch)return CHEAP_EINVAL;
    uint64_t state=seed;
    for(size_t i=0;i<(size_t)n*(size_t)n;++i) {
        if(!isfinite(cost[i]))return CHEAP_EDOM;
        double z=cost[i]-perturbation_scale*cheap_gumbel_standard(&state);
        if(!isfinite(z))return CHEAP_EDOM;
        cost_scratch[i]=z;
    }
    return cheap_transport_sinkhorn(n,a,b,cost_scratch,epsilon,max_iter,tol,
                                    f,g,plan_out);
}
#endif
