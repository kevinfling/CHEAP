# 1D PSD-validated circulant embedding and spectral diagnostics

This phase is **sampling and diagnostics only**. It does not alter the
DCT-diagonal surrogate sampler, PCG, or any 2D/3D operator.

## Covariance contract

For stationary covariance lags `c[0..n-1]`, construct the symmetric
circulant first row `[c0,...,c(n-1),0,...,0,c(n-1),...,c1]` at
power-of-two length `L >= 2n-1`. Its real DFT eigenvalues are cached.
If all eigenvalues are nonnegative, the first `n` components of the
circulant Gaussian vector have the requested Toeplitz covariance, up to
floating-point error.

In strict mode, `tol=0` requires no negative computed eigenvalues.
With `tol>0`, negative values whose magnitude is within
`tol * max(abs(min_eig),abs(max_eig))` are set to zero: this is a
numerical-tolerance interpretation of exactness, with a spectral
perturbation bounded by the clipping threshold. It is **not** a claim
of literal unperturbed covariance equality. A materially negative
eigenvalue rejects strict init with `CHEAP_EDOM` and
`info.status=CHEAP_EMBED_NOT_PSD`; resources are released. The
explicit `cheap_circulant_sampler_init_approx_clip` variant allows
material clipping and marks `CHEAP_EMBED_APPROX_CLIP`. Do not call
those samples exact.

A larger embedding length can sometimes eliminate negative eigenvalues,
but no automatic escalation is implemented in this cut. Supply an
explicit power-of-two L to test another embedding. Not all stationary
covariances have a PSD embedding at useful finite lengths.

## Transform convention and normal input

The backend is selected at build time via `CHEAP_USE_NOTORIOUS_FFT`.
The `backend` argument is currently required to be zero (configured
FFTW-compatible backend); runtime switching is not supported.

Input `z` has exactly `L` independent standard normals:
`z[0]` drives DC, `z[2k-1], z[2k]` drive real and imaginary parts
of interior frequencies, and `z[L-1]` drives Nyquist when L is even.
Each interior component uses `sqrt(lambda_k/2)`, and DC/Nyquist
use `sqrt(lambda_k)`. The unnormalized C2R inverse is divided by
`sqrt(L)`; thus `Cov(X)=C` for an unmodified PSD spectrum.

No RNG is created or owned by the sampler. `draw` uses cached FFT
plans, spectral square roots, and scratch: no allocations or planning.
Separate sampler contexts can run concurrently; one context cannot
be used concurrently. Zero-initialize before init, destroy before
reinit, and do not copy live contexts.

## Diagnostics

`cheap_diag_from_weights` reports finite minimum/maximum, counts of
negative and zero weights, and a condition estimate only when all
weights are strictly positive. Undefined quantities are NaN.

`cheap_diag_projection_error` compares a Toeplitz matrix with the
orthonormal DCT-II `Q diag(w) Q^T` surrogate using an independent
dense O(n³) reference for n<=256. It reports relative Frobenius error,
not an SPD certificate for the original Toeplitz matrix.

`cheap_diag_stochastic_probe` uses disclosed seeded Rademacher
vectors and reports `sqrt(sum ||(A-Atilde)v||² / sum ||Av||²)`.
It is an empirical RMS discrepancy, **not an upper bound**.
The diagnostics routine allocates temporary vectors; only the sampler
draw hot path is allocation-free.

## Stationarity

Fractional Gaussian noise increments are stationary and may be
sampled when their embedding passes the PSD test. Fractional Brownian
motion *levels* are nonstationary and are not represented by this
stationary-lag API. Constructing levels by integrating increments
requires an explicit covariance derivation and independent validation.

## Verification

The new `embedding_diagnostics` CTest inherits FFTW3/Notorious ×
Debug/ASan/UBSan from the existing six-job workflow. It checks
deterministic Monte Carlo lag recovery, explicit non-PSD rejection,
opt-in approximate clipping, lifecycle, dense projection diagnostics,
and seeded operator probes. Monte Carlo thresholds are statistical
test tolerances, not universal covariance-error guarantees.
