#ifndef CHEAP_PCG_H
#define CHEAP_PCG_H
/* Residual-controlled left-preconditioned CG for an original SPD operator.
 * C99; exact Toeplitz A and DCT-diagonal M^{-1} are optional adapters.
 * SPDX-License-Identifier: MIT
 */
#include "cheap_toeplitz_operator.h"
#include <float.h>
#include <limits.h>

typedef enum {
    CHEAP_PCG_CONVERGED = 0,
    CHEAP_PCG_MAX_ITER = 1,
    CHEAP_PCG_INVALID = 2,
    CHEAP_PCG_BREAKDOWN = 3,
    CHEAP_PCG_OPERATOR_ERROR = 4
} cheap_pcg_reason;

typedef struct {
    int iterations;
    double true_residual;
    double relative_residual; /* absolute residual if ||b||=0 */
    cheap_pcg_reason reason;
    int operator_calls;
    int preconditioner_calls;
    int callback_status;
} cheap_pcg_result;

typedef struct {
    int n;
    double *r, *z, *p, *Ap;
} cheap_pcg_workspace;

static inline void cheap_pcg_workspace_destroy(cheap_pcg_workspace *w) {
    if (!w) return;
    free(w->r);
    free(w->z);
    free(w->p);
    free(w->Ap);
    memset(w,0,sizeof(*w));
}
static inline int cheap_pcg_workspace_init(cheap_pcg_workspace *w,int n) {
    cheap_pcg_workspace t={0};
    if (!w || n<1 || (size_t)n>SIZE_MAX/sizeof(double) ||
        w->r || w->z || w->p || w->Ap) return CHEAP_EINVAL;
    t.n=n;
    t.r=(double*)malloc((size_t)n*sizeof(double));
    t.z=(double*)malloc((size_t)n*sizeof(double));
    t.p=(double*)malloc((size_t)n*sizeof(double));
    t.Ap=(double*)malloc((size_t)n*sizeof(double));
    if (!t.r || !t.z || !t.p || !t.Ap) {
        cheap_pcg_workspace_destroy(&t);return CHEAP_ENOMEM;
    }
    *w=t;return CHEAP_OK;
}
static inline double cheap_pcg_dot(int n,const double *a,const double *b) {
    double sum=0;
    for(int i=0;i<n;++i) sum+=a[i]*b[i];
    return sum;
}
static inline double cheap_pcg_norm(int n,const double *x) {
    double scale=0, ssq=1;
    for(int i=0;i<n;++i) {
        double a=fabs(x[i]);
        if(!isfinite(a)) return NAN;
        if(a!=0) {
            if(scale<a) {double q=scale/a;ssq=1+ssq*q*q;scale=a;}
            else {double q=a/scale;ssq+=q*q;}
        }
    }
    return scale==0?0:scale*sqrt(ssq);
}
static inline int cheap_pcg_recompute(
    const cheap_linear_operator *A,const double *b,const double *x,
    cheap_pcg_workspace *w,cheap_pcg_result *result,double *norm)
{
    int rc=cheap_linear_operator_apply(A,x,w->Ap);
    ++result->operator_calls;
    if(rc!=CHEAP_OK) {result->callback_status=rc;return CHEAP_PCG_OPERATOR_ERROR;}
    for(int i=0;i<A->n;++i) w->r[i]=b[i]-w->Ap[i];
    *norm=cheap_pcg_norm(A->n,w->r);
    return isfinite(*norm)?CHEAP_PCG_CONVERGED:CHEAP_PCG_BREAKDOWN;
}
/* Recompute the true residual at entry, periodically, and at exit.
 * refresh_every >= 1. An iteration with a refreshed residual restarts
 * the Krylov recurrence, avoiding an inconsistent (r,p) pair.
 * x and b must not overlap; workspace buffers must not overlap user arrays.
 * Caller supplies an SPD A, and optionally SPD M^{-1}; the implementation
 * checks observed curvature but cannot certify global SPD.
 * The output x may be partially updated on nonconvergence/breakdown.
 */
static inline cheap_pcg_result cheap_pcg_solve(
    const cheap_linear_operator *A,const cheap_linear_operator *Minv,
    const double *b,double *x,cheap_pcg_workspace *w,
    double atol,double rtol,int max_iterations,int refresh_every)
{
    cheap_pcg_result out={0, NAN,NAN,CHEAP_PCG_INVALID,0,0,0};
    double bn=0,rn=0,limit=0,rho=0;
    int n,rc,iteration,restart=1;
    if(!A || !A->apply || !A->context || !w || !b || !x ||
       A->n<1 || w->n!=A->n || !w->r || !w->z || !w->p || !w->Ap ||
       (Minv && (!Minv->apply || !Minv->context || Minv->n!=A->n)) ||
       !isfinite(atol) || !isfinite(rtol) || atol<0 || rtol<0 ||
       max_iterations<0 || refresh_every<1) return out;
    n=A->n;
    {uintptr_t ba=(uintptr_t)b,xa=(uintptr_t)x;
     size_t bytes=(size_t)n*sizeof(double);
     if((ba<=xa && xa-ba<bytes)||(xa<ba && ba-xa<bytes))return out;
     const double *bufs[]={w->r,w->z,w->p,w->Ap};
     for(int k=0;k<4;++k) {
         uintptr_t v=(uintptr_t)bufs[k];
         if((ba<=v && v-ba<bytes)||(v<ba && ba-v<bytes)||
            (xa<=v && v-xa<bytes)||(v<xa && xa-v<bytes))return out;
         for(int j=0;j<k;++j) {
             uintptr_t u=(uintptr_t)bufs[j];
             if((u<=v && v-u<bytes)||(v<u && u-v<bytes))return out;
         }
     }
    }
    for(int i=0;i<n;++i)
        if(!isfinite(b[i]) || !isfinite(x[i])) return out;
    bn=cheap_pcg_norm(n,b);
    if(!isfinite(bn))return out;
    limit=atol+rtol*bn;
    if(!isfinite(limit))return out;
    rc=cheap_pcg_recompute(A,b,x,w,&out,&rn);
    if(rc!=CHEAP_PCG_CONVERGED) {out.reason=(cheap_pcg_reason)rc;return out;}
    out.true_residual=rn;out.relative_residual=bn>0?rn/bn:rn;
    if(rn<=limit) {out.reason=CHEAP_PCG_CONVERGED;return out;}
    for(iteration=0;iteration<max_iterations;++iteration) {
        if(Minv) {
            rc=cheap_linear_operator_apply(Minv,w->r,w->z);
            ++out.preconditioner_calls;
            if(rc!=CHEAP_OK) {out.callback_status=rc;out.reason=CHEAP_PCG_OPERATOR_ERROR;break;}
        } else memcpy(w->z,w->r,(size_t)n*sizeof(double));
        double next_rho=cheap_pcg_dot(n,w->r,w->z);
        if(!isfinite(next_rho) || next_rho<=0) {out.reason=CHEAP_PCG_BREAKDOWN;break;}
        if(restart) {
            memcpy(w->p,w->z,(size_t)n*sizeof(double));
        } else {
            double beta=next_rho/rho;
            if(!isfinite(beta)) {out.reason=CHEAP_PCG_BREAKDOWN;break;}
            for(int i=0;i<n;++i)w->p[i]=w->z[i]+beta*w->p[i];
        }
        restart=0;
        rho=next_rho;
        rc=cheap_linear_operator_apply(A,w->p,w->Ap);
        ++out.operator_calls;
        if(rc!=CHEAP_OK) {out.callback_status=rc;out.reason=CHEAP_PCG_OPERATOR_ERROR;break;}
        double curvature=cheap_pcg_dot(n,w->p,w->Ap);
        if(!isfinite(curvature) || curvature<=0) {out.reason=CHEAP_PCG_BREAKDOWN;break;}
        double alpha=rho/curvature;
        if(!isfinite(alpha)) {out.reason=CHEAP_PCG_BREAKDOWN;break;}
        for(int i=0;i<n;++i) {
            x[i]+=alpha*w->p[i];
            w->r[i]-=alpha*w->Ap[i];
        }
        out.iterations=iteration+1;
        if(out.iterations%refresh_every==0) {
            rc=cheap_pcg_recompute(A,b,x,w,&out,&rn);
            if(rc!=CHEAP_PCG_CONVERGED) {out.reason=(cheap_pcg_reason)rc;break;}
            restart=1;
        } else rn=cheap_pcg_norm(n,w->r);
        if(!isfinite(rn)) {out.reason=CHEAP_PCG_BREAKDOWN;break;}
        if(rn<=limit) {
            rc=cheap_pcg_recompute(A,b,x,w,&out,&rn);
            if(rc!=CHEAP_PCG_CONVERGED) {out.reason=(cheap_pcg_reason)rc;break;}
            if(rn<=limit) {out.reason=CHEAP_PCG_CONVERGED;break;}
            /* Recursive estimate was optimistic: restart with true r. */
            restart=1;
        }
    }
    /* Termination always reports a newly evaluated true residual. */
    rc=cheap_pcg_recompute(A,b,x,w,&out,&rn);
    if(rc!=CHEAP_PCG_CONVERGED)out.reason=(cheap_pcg_reason)rc;
    else {
        out.true_residual=rn;
        out.relative_residual=bn>0?rn/bn:rn;
        if(out.reason==CHEAP_PCG_INVALID)
            out.reason=rn<=limit?CHEAP_PCG_CONVERGED:CHEAP_PCG_MAX_ITER;
        if(out.reason==CHEAP_PCG_CONVERGED && rn>limit)
            out.reason=CHEAP_PCG_MAX_ITER;
    }
    return out;
}

/* DCT-diagonal surrogate inverse used ONLY as a preconditioner, never
 * substituted for A. This adapter does not own the cheap_ctx.
 * User must keep the context alive and must not mutate weights concurrently.
 */
typedef struct {
    cheap_ctx *ctx;
    double *inverse_weights;
    int n;
} cheap_pcg_dct_preconditioner;
static inline void cheap_pcg_dct_destroy(cheap_pcg_dct_preconditioner *p) {
    if(!p)return;
    free(p->inverse_weights);
    memset(p,0,sizeof(*p));
}
static inline int cheap_pcg_dct_init(
    cheap_pcg_dct_preconditioner *p,cheap_ctx *ctx,const double *weights)
{
    if(!p || !ctx || !ctx->is_initialized || !weights ||
       p->inverse_weights || ctx->n<1)return CHEAP_EINVAL;
    double *v=(double*)malloc((size_t)ctx->n*sizeof(double));
    if(!v)return CHEAP_ENOMEM;
    for(int i=0;i<ctx->n;++i) {
        if(!isfinite(weights[i]) || weights[i]<=0) {
            free(v);return CHEAP_EDOM;
        }
        v[i]=1.0/weights[i];
        if(!isfinite(v[i]) || v[i]<=0) {free(v);return CHEAP_EDOM;}
    }
    p->ctx=ctx;p->inverse_weights=v;p->n=ctx->n;
    return CHEAP_OK;
}
static inline int cheap_pcg_dct_apply(void *context,const double *r,double *z) {
    cheap_pcg_dct_preconditioner *p=(cheap_pcg_dct_preconditioner*)context;
    if(!p || !p->ctx || !p->inverse_weights || !p->ctx->is_initialized ||
       p->ctx->n!=p->n)return CHEAP_EUNINIT;
    return cheap_apply(p->ctx,r,p->inverse_weights,z);
}
static inline cheap_linear_operator cheap_pcg_dct_as_operator(
    cheap_pcg_dct_preconditioner *p)
{
    cheap_linear_operator op={p?p->n:0,p,cheap_pcg_dct_apply};
    return op;
}
#endif
