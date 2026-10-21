#include <math.h>
#include <stdio.h>
#include "cheap_sinkhorn_divergence.h"
#include "cheap_gumbel_transport.h"
int main(void) {
    const double a[2]={0.5,0.5};
    const double c[4]={0,1,1,0};
    double f[2],g[2],p[4],scratch[4],d,v;
    int rc=cheap_sinkhorn_divergence(2,a,a,c,c,c,0.5,1000,1e-12,
                                      f,g,p,&d);
    if(rc!=CHEAP_OK || fabs(d)>1e-10)return 1;
    rc=cheap_sinkhorn_richardson(2,a,a,c,c,c,0.5,1000,1e-12,
                                 f,g,p,&d);
    if(rc!=CHEAP_OK || fabs(d)>1e-10)return 2;
    rc=cheap_gumbel_transport(2,a,a,c,0.5,0.1,12345,1000,1e-12,
                              f,g,scratch,p);
    if(rc!=CHEAP_OK)return 3;
    if(fabs(p[0]+p[1]-0.5)>1e-10)return 4;
    if(cheap_transport_cost(2,scratch,p,0.5,&v)!=CHEAP_OK)return 5;
    if(scratch[0]==c[0] && scratch[1]==c[1])return 6;
    puts("transport extensions: PASS");return 0;
}
