#include <math.h>
#include <stdio.h>
#include "cheap_transport.h"
int main(void) {
    const double a[2]={0.5,0.5},b[2]={0.5,0.5};
    const double c[4]={0,1,1,0};
    double f[2],g[2],p[4],v;
    int rc=cheap_transport_sinkhorn(2,a,b,c,0.2,1000,1e-12,f,g,p);
    if(rc!=CHEAP_OK)return 1;
    if(fabs(p[0]+p[1]-0.5)>1e-10 || fabs(p[2]+p[3]-0.5)>1e-10)return 2;
    if(fabs(p[0]+p[2]-0.5)>1e-10 || fabs(p[1]+p[3]-0.5)>1e-10)return 3;
    if(!(p[0]>p[1] && p[3]>p[2]))return 4;
    if(cheap_transport_cost(2,c,p,0.2,&v)!=CHEAP_OK||!isfinite(v))return 5;
    if(cheap_transport_sinkhorn(2,a,b,c,-1,100,1e-8,f,g,p)!=CHEAP_EINVAL)return 6;
    puts("dense Sinkhorn: PASS");return 0;
}
