# Spectral weight audit (research branch)

CHEAP's DCT-II basis exactly diagonalizes its specified cell-centered discrete Neumann Laplacian, but does **not** generally diagonalize Toeplitz covariance matrices. A weight is a function of the eigenvalues of a *specified* operator, not a universal model.

## Corrections to the existing periodic table

| Existing family | Decision | Mathematical contract |
|---|---|---|
| Laplacian / heat / Poisson | Keep | Discrete Neumann Laplacian, with explicit zero-mode policy; grid spacing is currently unit spacing |
| Fractional | Clarify | `(2 sin(pi k/2n))^d = mu_k^(d/2)` for k>0; existing epsilon floor at k=0 is a **regularized** fractional multiplier, not the true pseudoinverse |
| Wiener denoising | Rename default | `mu/(mu+noise)` is Wiener only if `mu` is a **signal power spectrum**. With Laplacian eigenvalues it is a high-pass spectral response |
| Specnorm | Rename in future major API | `(s+eps)^(-1/2)` is regularized covariance whitening, not neural-network spectral norm normalization |
| KPCA hard/soft | Clarify | Frequency truncation is not PCA unless indices are sorted by covariance eigenvalue; soft shrinkage is not a projector |
| Gamma ratio | Quarantine as generic multiplier | Magnitude of complex Gamma ratio alone does not establish multifractal scaling |
| RMT shrinkage | Quarantine pending derivation | Needs correct aspect-ratio convention, covariance versus singular-value loss, and independent numerical reference |
| Wiener deconvolution | Clarify | `h/(h*h+eta)` assumes real symmetric response and white signal/noise with constant variance ratio. General Wiener uses `S_x conj(h)/(|h|² S_x + S_n)` |
| Matérn | Correct in new API | In d dimensions, conventional Matérn covariance spectrum is proportional to `(kappa²+mu)^(-(nu+d/2))`; **sampling** uses the square root. Existing `cheap_weights_matern_*` use exponent `-nu` and `cheap_sample_matern_*` multiply by covariance weights instead of square roots |
| Anisotropic Matérn | Correct in new API | Separate positive mass `kappa²` from positive directional metric coefficients `a_j`; never substitute a tiny DC floor for mass |
| Biharmonic | Fix documentation | The square of the Neumann Laplacian has only the constant null mode. Do not zero k=1 as an alleged affine null mode |
| Higher-order Tikhonov | Fix documentation | Penalty `mu^p` is squared `H^p` seminorm (since `L^(p/2)` is applied before squaring), not `H^(p/2)` |
| Sinkhorn | Separate | `exp(-operator_eigenvalue/eps)` is a matrix heat-type semigroup, not in general the elementwise Gibbs kernel `exp(-C_ij/eps)` of entropic transport |

## Additive corrected constructors

`include/cheap_weights_verified.h` introduces explicitly named, additive APIs so existing users do not silently change numerical results:

- `cheap_weights_matern_covariance_ev` — conventional d-dimensional Matérn covariance multiplier with caller-supplied amplitude.
- `cheap_weights_matern_sampling_ev` — its square-root covariance multiplier for unit-variance white spectral coefficients.
- `cheap_weights_anisotropic_matern_2d_verified` — positive mass and directional metric.
- `cheap_weights_resolvent_ev`, `cheap_weights_heat_ev`, `cheap_weights_wave_cos_ev`, `cheap_weights_wiener_signal_ev` — generic, explicitly typed spectral operators.

These are **discrete spectral models**, not automatically exact restrictions of a continuum Matérn covariance. Amplitude/variance calibration, boundary conditions, and grid spacing remain caller responsibilities.

## Review of companion uploads

### gumbel.h
- Gaussian-copula transform produces Gumbel marginals **only when each Gaussian component is standardized to N(0,1)**. `ctx->sqrt_lambda` does not guarantee unit marginal variances. The DCT inverse normalization and covariance must be verified.
- `cheap_gumbel_sinkhorn` perturbs f/g then calls `cheap_sinkhorn`, which resets both to zero; perturbations are discarded. The code is not a Gumbel-Sinkhorn implementation. Correlated Gumbel perturbations of dual potentials are not equivalent to independent perturbations of assignment costs.
- Needs tests for marginal CDF, spatial dependence, and actual perturbation effect.

### cheap_sinkhorn.h and chizat.h
- `cheap_sinkhorn_T` is **not** an entropic OT solver: it computes squared Euclidean distance between two four-vectors divided by lambda, minus a constant. The constant cancels in the stated self-debiased divergence, so Richardson extrapolation does not deliver a meaningful second-order OT bias correction.
- Treat four-vectors as **features** rather than probability measures unless a measure/cost construction is supplied.
- Implement a genuine log-domain Sinkhorn transport solver with a specified cost matrix and marginals; test against exact small linear programs and known symmetries before attaching the Chizat/Sinkhorn divergence name.
- `_Alignas` in a typedef declaration needs C11 portability review; advertised strict C99 is incompatible with unconditional `_Alignas`.

### timberwolf.h
- **Critical out-of-bounds**: `tw_compute_ashi(scratch1,n,scratch2,scratch2+n/2,...)` passes an n/2-length energy buffer where n elements are required, and overlaps the n-element DCT buffer with that energy buffer. Allocate two distinct n-element buffers.
- `tw_quickselect` needs randomized/adversarial tests for partition progress and bounds. `tw_robust_mad` uses upper middle order statistic rather than interpolated median for even n.
- `tw_dct_ii` replans every call; reuse plans and aligned scratch for performance.
- `tw_process` recomputes whole-signal kurtosis and skewness inside the per-sample loop (quadratic sorting work), although those statistics are invariant over that loop.
- Validate `temp > 0`, `n_imfs > 0`, allocation overflow, NaN inputs and the CHEAPDAN API. Avoid `-ffast-math` until NaN/robust-statistics behavior is specified.
- This is an application-layer diagnostic pipeline, not a spectral weight family; keep separate from CHEAP's core.

## Verification gates

1. Independent dense DCT matrix reference for n=2..64; compare every weight's resulting operator.
2. For Matérn sampling, Monte Carlo empirical covariance against `Q^T diag(s) Q`; verify sample weights square to covariance weights.
3. For anisotropy, exchange axes and coefficients and verify transpose equivariance.
4. For heat, check identity, semigroup, and DC preservation; for wave, `t=0` identity and boundedness.
5. Run FFTW3/Notorious, C99/C11/C++17, ASan/UBSan; require no regressions before changing legacy API semantics.
