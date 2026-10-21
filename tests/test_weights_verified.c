#include <math.h>
#include <stdio.h>
#include "cheap_weights_verified.h"
static int near(double a, double b) {
    return isfinite(a) && isfinite(b) &&
           fabs(a-b) <= 2e-12 * fmax(1.0, fmax(fabs(a),fabs(b)));
}
int main(void) {
    double mu[4] = {0, 0.25, 1, 4};
    double cov[4], sample[4], wave[4], res[4], wiener[4];
    double noise[4] = {1,1,1,1};
    if (cheap_weights_matern_covariance_ev(4,mu,2,1.3,1.5,2.0,cov)) return 1;
    if (cheap_weights_matern_sampling_ev(4,mu,2,1.3,1.5,2.0,sample)) return 2;
    for (int i=0;i<4;++i) if (!near(sample[i]*sample[i],cov[i])) return 3;
    if (cheap_weights_wave_cos_ev(4,mu,0.0,wave)) return 4;
    for (int i=0;i<4;++i) if (!near(wave[i],1.0)) return 5;
    if (cheap_weights_resolvent_ev(4,mu,2.0,res)) return 6;
    for (int i=0;i<4;++i) if (!near(res[i]*(mu[i]+2.0),1.0)) return 7;
    if (cheap_weights_wiener_signal_ev(4,cov,noise,wiener)) return 8;
    for (int i=0;i<4;++i)
        if (!near(wiener[i],cov[i]/(cov[i]+1.0))) return 9;
    double a[6],b[6];
    if (cheap_weights_anisotropic_matern_2d_verified(2,3,1.0,2.0,3.0,1.5,1.0,a)) return 10;
    if (cheap_weights_anisotropic_matern_2d_verified(3,2,1.0,3.0,2.0,1.5,1.0,b)) return 11;
    for (int i=0;i<2;++i)
        for (int j=0;j<3;++j)
            if (!near(a[i*3+j],b[j*2+i])) return 12;
    if (!cheap_weights_matern_covariance_ev(4,mu,0,1,1,1,cov)) return 13;
    puts("verified weights: PASS");
    return 0;
}
