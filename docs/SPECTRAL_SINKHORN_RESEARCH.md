# Spectral Sinkhorn research: correctness, novelty, and release gates

## Established result, not a novelty claim

The classical entropic OT problem has K_ij = exp(-C_ij/epsilon), **entrywise**.
For positive marginals, Sinkhorn scaling uses P = diag(u) K diag(v).
A spectral operator exp(-L/epsilon) is generally **not** the same kernel.
Even if exp(-L/epsilon) is positive semidefinite, it need not be entrywise
nonnegative. A DCT-diagonal surrogate for a Toeplitz cost likewise does
not preserve entrywise exponentiation.

A genuine fast operator Sinkhorn method is possible when a fast-to-apply
**entrywise positive** K is provided and is demonstrably related to a
cost. Heat-kernel OT and convolutional Wasserstein distances are established
prior art (Solomon et al., SIGGRAPH 2015); graph heat-kernel Sinkhorn and
low-rank Sinkhorn also exist. FFT-based acceleration alone is not novel.

## Potential contributions requiring proof and measurement

1. **Positive spectral kernel certification.** Given DCT eigenvalues
   w_k, certify K = Q diag(w) Q^T has strictly positive entries, or
   construct a positive, fast-to-apply approximation with quantified
   kernel error. Nonnegative w_k is NOT sufficient.
2. **Cost-aware spectral compression.** Approximate exp(-C/epsilon)
   rather than exp(-Q diag(lambda) Q^T/epsilon), with explicit bounds
   on marginal feasibility and regularized objective error.
3. **Adaptive hybrid operator.** Use separable DCT/FFT convolution for
   stationary blocks and sparse/dense corrections for boundaries or
   heterogeneity; compare wall-clock including planning and setup.
4. **Certified fallback.** Detect negative kernel entries, underflow,
   marginal stagnation, or failure of error bounds and fall back to
   log-domain dense Sinkhorn. Record the fallback rate.
5. **Reusable multiscale plans.** Amortize transforms and warm starts
   across many marginals and epsilon schedules; benchmark repeated
   calls, not just one isolated solve.

None of these are claimed new until checked against prior art and
demonstrated experimentally.

## Release matrix and known blocker (2026-10-20)

The six-way workflow runs FFTW3 and NOTORIOUS in Debug, ASan and UBSan.
At commit ae86d393, all **three FFTW3 jobs passed**. All three
NOTORIOUS jobs failed tests: the pinned Notorious backend rejects some
dimensions, leading to failed context initialization in SIMD/2D/3D tests.
Do not disable those tests or describe the matrix as green.
Non-power-of-two r2r support and correct ND execution must be validated
in the pinned backend before claiming parity.

The original cheap_sinkhorn(ctx,...) uses a spectral operator exponential,
not an arbitrary-cost entropic-OT Gibbs matrix. Treat it as an operator
scaling heuristic, and migrate actual EOT callers to cheap_transport.h.
Do not call its result an EOT transport plan without verifying K_ij >= 0.

## Benchmarks needed

- Dense reference: log-domain Sinkhorn on arbitrary nonnegative costs.
- Structured reference: separable Gaussian convolution and FFT/DCT kernel
  scaling where the cost admits an exact fast convolution.
- Alternatives: low-rank Sinkhorn, graph heat-kernel Sinkhorn.
- Cases: regular 1D/2D/3D grids, irregular boundaries, nonstationary
  costs, sparse and highly imbalanced marginals, small epsilon.
- Report: marginal residual, objective bias, Sinkhorn divergence,
  runtime (setup and amortized), peak memory, precision and failures.
- Sanitizers: ASan+UBSan and leak detection for both backends.

## References

- Cuturi (2013), Sinkhorn Distances: Lightspeed Computation of Optimal Transport.
- Solomon et al. (2015), Convolutional Wasserstein Distances.
- Scetbon et al. (2021), Low-Rank Sinkhorn Factorization.
- Geodesic Sinkhorn for Fast and Accurate Optimal Transport on Manifolds (2023).
- Accelerating Sinkhorn for Sparse Multi-Marginal OT via FFTs (2022).
