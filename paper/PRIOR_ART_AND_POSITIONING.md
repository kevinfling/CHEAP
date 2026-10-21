# Prior-art map and revised research positioning

Status: targeted literature reconnaissance, not a systematic exhaustive search. Sources below are publication landing pages or abstracts; full proofs and priority claims must be checked against full texts before submission.

## 1. Fast Toeplitz and circulant preconditioning
- T. F. Chan (1988), *An Optimal Circulant Preconditioner for Toeplitz Systems*, SIAM J. Sci. Stat. Comput., DOI 10.1137/0909051. Frobenius-optimal circulant approximation; therefore minimizing ||A-P||_F over a fast transform algebra is established.
- Chan and Strang (1989), *Toeplitz Equations by Conjugate Gradients with Circulant Preconditioner*, DOI 10.1137/0910009. Classical spectral clustering and PCG analysis.
- Ku and Kuo, *Design and Analysis of Toeplitz Preconditioners*, IEEE Transactions on Signal Processing (1992; conference precursor 1990 DOI 10.1109/ICASSP.1990.115843). Fast transform-based preconditioners and clustering.
- Chen, Hurvich and Lu (2004 working paper, posted 2008), *On the Correlation Matrix of the Discrete Fourier Transform and the Fast Solution of Large Toeplitz Systems for Long-Memory Time Series*. Already addresses fast PCG solves for long-memory covariance Toeplitz systems; claimed O(n log^(5/2) n) under assumptions. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1293595

## 2. Trigonometric transform preconditioners (high overlap)
- Chan, Ng and Wong (1996), *Sine transform based preconditioners for symmetric Toeplitz systems*, Linear Algebra Appl. 232, 237-259, DOI 10.1016/0024-3795(94)00049-2. Frobenius-optimal sine-transform projection; O(n) construction for Toeplitz A in their setting. Important: our earlier O(n log n) DCT projection is not by itself a novelty claim.
- *Fast Transform Based Preconditioners for Toeplitz Equations*, SIAM J. Matrix Anal. Appl. 16 (1995), 628-645, DOI 10.1137/S0895479893254269. O(n log n) sine-transform preconditioner construction and spectral clustering.
- Kailath and Olshevsky, *Displacement Structure Approach to Discrete-Trigonometric-Transform Based Preconditioners of G. Strang Type and of T. Chan Type*, DOI 10.1137/S0895479896312560. Explicit constructions for eight DCT/DST transform algebras, with spectral clustering under Wiener-class assumptions. **Highest-priority full-text comparison**: our DCT-II formula may already be covered or reducible to their general construction.
- *Computing preconditioners via subspace projection*, Linear Algebra Appl. 302-303 (1999), 347-353, DOI 10.1016/S0024-3795(99)00143-3. Directly studies Frobenius projections onto matrices diagonalized by orthogonal trigonometric transforms.
- Ng, R. H. Chan, T. F. Chan and Yip (2000), *Cosine transform preconditioners for high resolution image reconstruction*, Linear Algebra Appl. 316, 89-104, DOI 10.1016/S0024-3795(99)00274-8. DCT preconditioning and eigenvalue clustering in a different inverse problem.

## 3. DCT and fractional Brownian motion covariance (closest direct overlap)
- Gupta and Joshi (2013), *DCT and Eigenvectors of Covariance of 1st and 2nd order Discrete fractional Brownian motion*, arXiv:1302.5556, https://arxiv.org/abs/1302.5556. Asymptotic DCT approximation of dfBm covariance eigenvectors via perturbation theory. **Precludes claiming novelty of the DCT/fBm covariance connection.**
- Gupta, Joshi and Singh (2018), *On the approximate discrete KLT of fractional Brownian motion and applications*, Journal of the Franklin Institute, DOI 10.1016/j.jfranklin.2018.09.023. Further approximate Karhunen–Loève/DCT connection. Important distinction: approximate covariance eigenvectors are not the same as preconditioned convergence guarantees.
- *Some Studies on the Structure of Covariance Matrix of Discrete-Time fBm*, prior work on the dominant nonstationary eigenvalue and eigenvector structure. Check full bibliographic details and mathematical hypotheses.

## 4. Stationary increments and alternative fBm algorithms
- Davies–Harte / Wood–Chan circulant embedding: exact FFT-based sampling of fGn and hence fBm on uniform grids; not a solver for (A+rho I)x=b, but a powerful algorithmic comparator for *sampling tasks*. See https://academic.oup.com/biomet/article/102/4/809/237610.
- SuperGauss (generalized Schur / Ammar–Gragg) implements fast Toeplitz likelihood computations with O(n log^2 n) scaling and FFTW. https://mirror.vtti.vt.edu/cran/web/packages/SuperGauss/vignettes/SuperGauss-quicktut.html. Compare if claiming advantage for likelihood, inverse-covariance application, or Gaussian inference.
- Increment-domain fGn–Chan PCG is the directly relevant solver baseline currently implemented in experiments/fbm_increment_domain_comparison.py. The exact change of variables is D(A+rho I)D^T = G+rho DD^T, but transformed residuals require explicit original-system checking.

## 5. What is and is not new
**Established:** DCT as approximate fBm KLT; trigonometric algebra projection; Frobenius-optimal preconditioning; O(n log n) fast transform construction; circulant Toeplitz solvers; fGn stationary increments.

**Potential, NOT YET ESTABLISHED:** an especially practical and well-engineered specialization for regularized finite-grid fBm covariance, adaptive selection between DCT and increment-domain fGn preconditioners, strong end-to-end empirical comparison in the relevant parameter regimes, and a sharper fBm-specific spectral bound.

**Necessary correction to manuscript:** do not describe O(n log n) projection alone, DCT/fBm alignment alone, or Frobenius optimality as mathematical novelties. A publishable contribution needs an explicitly demonstrated gap versus Kailath–Olshevsky and Gupta–Joshi, and better controlled baselines.

## 6. Publication plan
1. Read the full Kailath–Olshevsky and Gupta–Joshi papers; derive a theorem-to-theorem overlap matrix.
2. Separate regularized covariance solves from exact Gaussian sampling and Toeplitz likelihood, which solve different tasks.
3. Add native C99 FFTW fGn–Chan baseline and multi-RHS end-to-end timings with common setup charged fairly.
4. Measure spectrum and condition estimates versus N, H and rho; seek fBm-specific clustering, not generic perturbation bounds.
5. Only then finalize title, abstract, and novelty statement. Conservative provisional title: *Fast Trigonometric Preconditioning of Regularized Fractional Brownian Covariance Systems: A Reproducible Comparison*.
