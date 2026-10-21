# Diligence addendum: increment-domain fGn competitor (2026-10)

## Exact reduction
Let A be the finite-grid fBm covariance at t_i=(i+1)/N, and D the nonsingular lower-bidiagonal first-difference matrix with first row e_0^T. Define G=D A D^T, which is the stationary fractional Gaussian noise covariance. For regularized original system (A+rho I)x=b, put y=D^{-T}x. Then
(G+rho D D^T)y=D b,  x=D^T y.
This equivalence is exact. The Toeplitz lags of G are
g_d = (|d-1|^(2H)-2|d|^(2H)+|d+1|^(2H))/(2 N^(2H)).
We compare CG on A+rho I, DCT-PCG on A+rho I, and fGn increment-domain CG with a Chan-optimal circulant approximation of G+rho D D^T. The last circulant is SPD in the tested configurations. The circulant coefficients are the Frobenius projection of the transformed operator, not a clipped naive fBm circulant.

## Important accuracy caveat
The transformed residual ||(G+rho DD^T)y-Db||/||Db|| is not the same as the original-system residual ||(A+rho I)D^Ty-b||/||b||. A 1e-9 transformed tolerance failed the latter test (1.59e-8 for one N=256 case). The increment-domain solver was therefore run to relative transformed tolerance 1e-12; **every solver was accepted only if the true original residual was <= 1.1e-9**. This asymmetric internal tolerance is needed for original-system accuracy but must be disclosed in all timing comparisons.

## Repeated independent RHS comparison
36 configurations: N=256,1024,4096; H=.3,.7; rho/lambda_max=1e-4,1e-3; seeds=0,1,2. 108 solver runs. All passed original residual verification. FFT matrix-free Python/SciPy, single-threaded BLAS requested, times exclude common and method-specific setup. These are exploratory, not native FFTW end-to-end timing results.

At N=4096, rho/lambda_max=1e-4, over three random RHS:
| H | CG iterations | DCT-PCG iterations | fGn-Chan-PCG iterations | CG median ms | DCT median ms | fGn-Chan median ms |
|---|---|---|---|---|---|---|
| .3 | 110-112 | 10 | 13 | 19.126 | 2.200 | 3.246 |
| .7 | 47-48 | 10 | 16 | 7.885 | 2.108 | 3.937 |

The three RHS are not a statistically powerful sample; no performance superiority claim is justified. The fGn solver is an appropriately structured comparator, but other stronger algorithms may exist.

## Literature context / priority correction
Cosine-transform preconditioners, optimal transform-algebra projections, and fast Toeplitz/circulant preconditioning are established work. The projection theorem is not novel; the efficient fBm specialization may still be a useful engineering contribution but requires thorough novelty review. Relevant works:
- T. F. Chan, *An Optimal Circulant Preconditioner for Toeplitz Systems*, SIAM J. Sci. Stat. Comput. 9 (1988), 766–771, doi:10.1137/0909051.
- R. H. Chan and G. Strang, *Toeplitz Equations by Conjugate Gradients with Circulant Preconditioner*, SIAM J. Sci. Stat. Comput. 10 (1989), 104–119, doi:10.1137/0910009.
- R. H. Chan, M. K. Ng, C. K. Wong, *Sine transform based preconditioners for symmetric Toeplitz systems*, Linear Algebra Appl. 232 (1996), 237–259, doi:10.1016/0024-3795(94)00049-2.
- M. K. Ng, R. H. Chan, T. F. Chan, A. M. Yip, *Cosine transform preconditioners for high resolution image reconstruction*, Linear Algebra Appl. 316 (2000), 89–104, doi:10.1016/S0024-3795(99)00274-8.

## Valid mathematical bound, not a mesh-independence proof
Let B=Q^T diag(diag(QAQ^T)) Q be the Frobenius projection, M=B+rho I and E=A-B. Since A is SPD and rho>0, M is SPD. Define delta=||M^{-1/2} E M^{-1/2}||_2. If delta<1, then all eigenvalues of M^{-1/2}(A+rho I)M^{-1/2} lie in [1-delta,1+delta], hence kappa <= (1+delta)/(1-delta). This follows directly from the Rayleigh quotient and is standard perturbation theory. No N-independent upper bound on delta has yet been proved for fBm. The empirical 10-iteration result does not establish mesh-independent convergence.

## Remaining publication blockers
- The complete native FFTW benchmark source and official FFTW header reproducibility have not yet been pushed to GitHub.
- Native C implementation of the fGn-Chan baseline, including separated setup timings and multiple RHS, is still required.
- Add strong alternatives: hierarchical solvers, increment-domain alternatives, boundary-corrected and optimal-circulant methods; run independent hardware and compiler checks.
- Parameter sweep H near 0 and 1, larger N, random RHS and severe regularization; track numerical breakdown.
- Review novelty and write a manuscript with limitations and raw data, not a premature SOTA claim.
