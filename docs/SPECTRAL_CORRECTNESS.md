# CHEAP: mathematical correctness and reproducibility audit

Status: research branch; claims below are corrections and proposed experiments, not benchmark results.

## Operator contract
Let Q be the **orthonormal** DCT-II, Q[k,j] = sqrt((2 - delta[k,0])/N) cos(pi*k*(j+1/2)/N). Define T(w)=Q^T diag(w) Q. Every T(w) is symmetric; it is positive semidefinite iff every w[k]>=0. This does NOT imply that an arbitrary Toeplitz matrix is DCT diagonal.

For the symmetric Neumann second-difference operator L with endpoint diagonal 1, interior diagonal 2, and nearest-neighbor entries -1, L=Q^T diag(mu) Q, mu[k]=4 sin^2(pi*k/(2N))/h^2. The spectral square L^2 has only the constant null vector; it is not interchangeable with an independently specified fourth-derivative boundary-value operator.

For any symmetric A, its closest DCT-diagonal approximation in Frobenius norm is P_Q(A)=Q^T diag(diag(Q A Q^T)) Q, and ||A-P_Q(A)||_F^2 equals the squared Frobenius norm of the off-diagonal entries of Q A Q^T. This is standard orthogonal projection, not a novelty claim.

## Mandatory corrections
1. Remove the unsupported theorem that fractional Brownian motion covariance is DCT diagonalized with universal O(1/N) error. Analytic perturbation in H does not imply convergence in N. Test finite-grid covariance for H in (0,1), with clearly stated normalization.
2. Toeplitz: use FFT circulant embedding for an exact matvec (with sufficient padding), or identify DCT use as a surrogate/preconditioner. Do not call a DCT of the first Toeplitz column its exact eigenvalues.
3. Gaussian sampling: if s[k] denotes covariance eigenvalues, apply sqrt(s[k]) to independent standard-normal DCT coefficients, not s[k].
4. Matérn SPDE covariance: s[k] proportional to (kappa^2+mu[k])^(-(nu+d/2)); sampling uses exponent -(nu+d/2)/2. If the API accepts the raw exponent p, call it p rather than Matérn smoothness nu. Discrete Neumann-domain covariance is not generally the restriction of the stationary Euclidean Matérn kernel.
5. Sinkhorn: K[i,j]=exp(-C[i,j]/epsilon) entrywise. exp(-t*mu[k]) is the heat semigroup of the discrete Laplacian, not automatically the spectrum of that Gibbs kernel. For translation-invariant grid costs, construct the actual convolution kernel and use exact padded FFT matvecs.
6. Wiener filter: s[k]/(s[k]+n[k]) requires a prior signal PSD s and noise PSD n. Laplacian eigenvalues are not a signal PSD by default.
7. RMT shrinkage needs a sample-covariance/random-matrix model; DCT frequency indices are not sample-covariance eigenvalues.
8. Rename hard low-frequency projection to spectral cutoff, not kernel PCA without proof of covariance eigenbasis ordering.
9. A deterministic linear filter of Gaussian noise is Gaussian; a complex Gamma-ratio multiplier alone does not construct multifractality. Relabel experimental or remove the claim.
10. Implicit Euler: for x>=0, exp(-x) <= 1/(1+x). Implicit Euler damps LESS than exact heat evolution at fixed step. Crank-Nicolson is A-stable for x>=0, although it is not positivity preserving and high modes can change sign when x>2.
11. Biharmonic: distinguish L^2 (one-dimensional constant nullspace) from continuous d^4/dx^4 under separately imposed boundary conditions.
12. Discrete derivative covariance is not automatically diagonal in the same DCT basis: boundary terms and the derivative's codomain matter. Avoid claiming that multiplying by mu^m implements an ordinary mth derivative covariance without qualification.

## Weight taxonomy
- **Exact discrete:** heat exp(-t mu); implicit Euler 1/(1+t mu); Crank-Nicolson (1-t mu/2)/(1+t mu/2); screened Poisson 1/(kappa^2+mu); fractional heat exp(-t mu^alpha); spectral L^s with explicit DC policy; wave displacement cos(t sqrt(mu)).
- **Well-defined model, interpretation qualified:** discrete SPDE Matérn covariance/sampling; Bessel potential; Gaussian spectral covariance exp(-ell^2 mu/2); Sobolev Tikhonov 1/(1+alpha mu^p).
- **Conditional:** deconvolution with known transform-diagonal PSF and conjugation for complex transfer functions; Wiener with supplied PSDs; OU as separate deterministic propagation plus stochastic innovation.
- **Experimental/relabel:** Flandrin/fBm surrogate, rough-volatility synthesis, RMT shrinkage, multifractal Gamma ratio, kernel PCA, generic Toeplitz DCT solve.

## Research protocol
For each target A and surrogate B, report relative Frobenius error, relative spectral error, ||b-A*xhat||/||b||, solution error against reference, and where relevant PSD violation, sampling covariance error and Sinkhorn marginal/objective error. Compare exact FFT Toeplitz matvec, dense LAPACK, DCT surrogate, DCT-preconditioned CG, and established circulant preconditioners. Vary N, boundary conditions, H, conditioning, regularization, and dimensions. Separate setup/planning from repeated execution. Run sanitizers and numerical tests.

Potential research direction: adaptive transform choice with a cheap certified or probabilistic bound on ||M^(-1/2)(A-B)M^(-1/2)||_2. If this quantity delta<1, the preconditioned condition number is at most (1+delta)/(1-delta). The inequality is standard; a useful estimator and competitive empirical performance would be the contribution.

No new speedups, approximation rates, or publication novelty are asserted here.
