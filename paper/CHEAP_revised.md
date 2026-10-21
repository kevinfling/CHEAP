# The CHEAP Trick: Error-Controlled Spectral Operators
## Revised research manuscript — working draft (20 October 2026)
**Kevin Fling — Independent Researcher**

> This is a replacement research draft, not a claim that the original 33-page manuscript has been fully revalidated. The original contains a detailed catalog of proposed weights; their implementation status and measured fidelity remain to be audited.

### Abstract
A transform-diagonal multiplier of the form T(w)=Q^T diag(w) Q permits O(N log N) application and O(N) auxiliary storage when Q is a fast orthogonal transform. This observation unifies exact discrete PDE propagators, spectral regularizers, and certain covariance models. It does not establish diagonalization of arbitrary Toeplitz matrices, stationary covariances, or finite-grid fractional Brownian motion. We separate exact operators from deliberately constructed surrogates, derive the optimal Frobenius projection onto a fixed transform-diagonal algebra, and describe error-aware iterative correction. We propose reproducible comparisons with exact FFT-based Toeplitz products and established preconditioners. No empirical advantage or new theorem is claimed before those experiments.

### 1. Definitions
For k,j=0,...,N-1 let Q[k,j]=sqrt((2-delta[k,0])/N) cos(pi*k*(j+1/2)/N). Then Q Q^T=I. For a real weight vector w define T(w)=Q^T diag(w) Q. Symmetry is automatic; positive semidefiniteness holds exactly when w>=0 componentwise.

For grid spacing h>0, the symmetric reflected-boundary second-difference L has diagonal [1,2,...,2,1]/h^2 and adjacent entries -1/h^2. It is diagonalized by Q with eigenvalues mu[k]=4 sin^2(pi*k/(2N))/h^2. Thus f(L)=T(f(mu)) for functions defined on its spectrum. This is exact finite-dimensional spectral calculus, not an asymptotic assertion.

### 2. Approximating another operator
For real symmetric A, write S=Q A Q^T. Minimizing ||A-T(w)||_F^2 over w gives w*=diag(S) and
||A-T(w*)||_F^2 = sum_{i != j} S[i,j]^2.
This is a standard orthogonal projection identity. If the target A is not explicitly available, the diagonal of S may be estimated with matrix-vector probes; estimator bias, variance, and work must be measured.

Let B=T(w*) and M=B+rho I. If A+rho I and M are positive definite, let delta=||M^(-1/2)(A-B)M^(-1/2)||_2. For delta<1 the eigenvalues of M^(-1/2)(A+rho I)M^(-1/2) lie in [1-delta,1+delta], so its condition number is at most (1+delta)/(1-delta). This is a standard perturbation bound. Its practical use depends on estimating delta inexpensively.

### 3. Exact and conditional weight families
For the Neumann discrete Laplacian: heat exp(-t mu); implicit Euler 1/(1+t mu); Crank-Nicolson (1-t mu/2)/(1+t mu/2); screened Poisson 1/(kappa^2+mu); spectral fractional diffusion exp(-t mu^alpha); spectral fractional power mu^alpha with an explicit zero-mode convention; and wave propagation cos(t sqrt(mu)) for zero initial velocity. A Neumann Poisson inverse requires a zero-mean right-hand side and uses the pseudoinverse weight 1/mu for k>0 and zero at k=0.

A discrete Matérn-type SPDE covariance in dimension d uses spectral density proportional to (kappa^2+mu)^(-(nu+d/2)); to sample from it, apply its square root to independent standard-normal transform coefficients. Domain boundaries, grid spacing, variance normalization and the distinction between a discrete SPDE model and the restriction of a continuum stationary covariance must be stated.

Wiener filtering needs the signal and noise power spectra S and N: w=S/(S+N). With a known transform-diagonal blur response psi, regularized deconvolution uses conjugate(psi)/(abs(psi)^2+alpha R), not a real-valued formula unless psi is known real. A frequency cutoff is not automatically kernel PCA. A deterministic linear transform of Gaussian noise is still Gaussian, so a Gamma-ratio multiplier alone does not establish multifractal scaling.

### 4. Corrections to the earlier manuscript
- The earlier DCT-II definition omitted orthonormal scaling.
- The claimed universal O(1/N) fBm eigenbasis convergence is not proved by analytic perturbation in H; withdraw it pending an appropriate theorem or controlled evidence.
- A Toeplitz matrix is not generally DCT-diagonal. An exact Toeplitz matvec can be performed through a sufficiently padded circulant embedding.
- The Sinkhorn Gibbs kernel is K[i,j]=exp(-C[i,j]/epsilon), an entrywise exponential, not exp(-C/epsilon) as a matrix function. Heat-semigroup weights cannot substitute without an operator-specific derivation.
- For x>=0, exp(-x)<=1/(1+x): backward Euler damps less than exact heat evolution. Crank-Nicolson is A-stable for diffusion but may oscillate in sign for high modes.
- The nullspace of L^2 is the nullspace of L, namely constants, not generally the affine functions associated with other fourth-order boundary conditions.
- A spectral power of the Laplacian does not automatically represent an ordinary directional derivative.
- Marchenko-Pastur/BBP eigenvalue shrinkage requires an appropriate sample-covariance model; transform frequencies are not sample covariance eigenvalues.

### 5. Experiments required before publication
Build reference dense operators for small N and exact FFT Toeplitz products for large N. Compare DCT projection, FFT/circulant preconditioners, and DCT-preconditioned conjugate gradients. Report operator Frobenius and spectral errors, true-system residual, solution error, condition number, iteration count, wall-clock time, planning/setup cost, and peak memory. For stochastic models measure empirical covariance error with confidence intervals; for Sinkhorn measure marginal feasibility and objective gap. Sweep N, Hurst H, grid spacing, regularization, boundary conditions, and conditioning. Publish code, seeds, compiler flags, hardware, transform-planning settings, and negative results.

### 6. Novelty and limits
The transform multiplier, Frobenius projection identity, and preconditioned spectral enclosure are established numerical mathematics. The candidate contribution is an adaptive implementation with efficient, validated error diagnostics and demonstrably competitive performance. Claims of broad superiority, exact finite-grid fBm diagonalization, and universal kernel equivalence are withdrawn.

### Selected prior art to verify and cite in final submission
- G. Strang, *A proposal for Toeplitz matrix calculations*, Studies in Applied Mathematics (1986).
- R. H. Chan, *Circulant preconditioners for Hermitian Toeplitz systems*, SIAM Journal on Matrix Analysis and Applications (1989).
- U. Grenander and G. Szegő, *Toeplitz Forms and Their Applications*.
- H. Rue and L. Held, *Gaussian Markov Random Fields: Theory and Applications*.
- F. Lindgren, H. Rue, and J. Lindström, *An explicit link between Gaussian fields and Gaussian Markov random fields: the stochastic partial differential equation approach*, JRSS B (2011).
- M. Cuturi, *Sinkhorn distances: Lightspeed computation of optimal transport*, NeurIPS (2013).

**Status:** mathematical rewrite and experimental specification. Results, benchmarks and code validation are pending.
