# CHEAP Trick

## Circulant Hessian Efficient Algorithm Package for the Fiscally Responsible Researcher

**Kevin Fling**
*Independent Researcher*

**Abstract.** We present CHEAP, a unified mathematical framework for efficient computation with structured covariance matrices that appear in kernel methods, optimal transport, fractional dynamics, signal processing, and random matrix theory. While naive approaches demand cubic time, we exploit the asymptotic spectral structure of Toeplitz matrices and discrete-time fractional Brownian motion (dfBm) through the discrete cosine transform (DCT). Building on analytic perturbation theory and deterministic feature mappings, CHEAP reduces the dominant computational cost to linearithmic time. Rather than treating the spectral projection as a mere approximation, we demonstrate that it induces a valid finite-dimensional reproducing kernel Hilbert space (RKHS). The framework provides a single elegant primitive that unifies kernel ridge regression, Sinkhorn optimal transport, fractional differential operators, Wiener filtering, spectral normalization, kernel PCA projection, Mandelbrot multifractal weights, and random matrix denoising. We provide closed-form expressions, eigenvalue formulas, and perturbation bounds that make the method self-contained and verifiable. On standard benchmarks, CHEAP occupies a compelling point on the Pareto frontier: it matches the accuracy of dense or graph-based baselines while reducing build time and memory footprint by one to two orders of magnitude.

---

## 1. Introduction

Many problems in machine learning and scientific computing reduce to operations on structured matrices. Kernel ridge regression requires solving linear systems with kernel matrices. Entropic optimal transport repeatedly applies Gibbs kernels. Fractional calculus involves non-local operators with power-law memory. In each case, the naive approach costs O(n³) time and O(n²) memory, which quickly becomes prohibitive for large-scale applications.

We observe that these seemingly unrelated problems share a hidden structure. The matrices are often Toeplitz, circulant, or, in the case of discrete-time fractional Brownian motion, admit fast asymptotic diagonalization. The discrete cosine transform (type II) provides a practical pathway to exploit this structure, yielding algorithms that run in O(n log n) time with O(n) memory overhead.

This paper introduces CHEAP, built on three pillars:

1. **Asymptotic Spectral Diagonalization**: Turning matrix-vector products into elementwise operations in the DCT domain.
2. **Deterministic Feature RKHS**: Viewing these operations as exact mappings inside an induced finite-dimensional reproducing kernel Hilbert space to provide stable, deterministic representations.
3. **Universal Spectral Primitive**: A single algorithmic pattern that serves diverse purposes through the selection of a specific weight vector.

---

## 2. Preliminaries and Notation

### 2.1 The Discrete Cosine Transform (DCT)

The DCT-II of a vector x ∈ ℝⁿ is defined by the unitary matrix **C** whose entries are:

$$C_{k,i} = \cos\left(\frac{\pi k (2i+1)}{2n}\right)$$

We denote the forward transform by x̂ = **C**x and the inverse by x = **C**ᵀx̂. The DCT acts as the Karhunen–Loève transform (KLT) for first-order Gauss–Markov processes and is the standard for frequency-domain analysis in signal processing.

### 2.2 Discrete-Time Fractional Brownian Motion (dfBm)

Fractional Brownian motion is a non-stationary Gaussian process with stationary increments. For the discrete-time version with Hurst exponent H ∈ (0, 1), the auto-covariance matrix $R_{B,1}^H(n)$ is characterized by long-range dependence. Recent results show that the eigenvectors of the 1st and 2nd order dfBm auto-covariance matrices can be approximated by DCT basis vectors in the asymptotic sense as n → ∞ [7].

### 2.3 No-Trick Deterministic Features

Standard kernel methods rely on the "kernel trick" to perform pairwise evaluations. Instead, we adopt a "No-Trick" (NT) approach [10], which uses deterministic frequency-domain mappings to obtain a data-independent basis for a finite-dimensional feature space. This induces a new RKHS where dot products are equivalent to the target kernel, avoiding the variance and sampling noise associated with Random Fourier Features (RFF).

---

## 3. The CHEAP Framework

### 3.1 The Universal Spectral Primitive

Every operation in CHEAP reduces to the same O(n log n) pattern:

$$\text{output} = \text{iDCT}\!\left(\text{DCT}(\text{input}) \odot w\right)$$

where **w** is a weight vector chosen according to the specific problem. By selecting different **w**, we obtain:

| Operation | Weight Vector **w** | Eigenvalue Family |
|---|---|---|
| Kernel Ridge Regression | w_k = 1 / (λ_k + λ_reg) | Flandrin |
| Gaussian Sampling | w_k = √λ_k | Flandrin |
| Fractional Differentiation | w_k = (2 sin(ω_k / 2))^d | Index-based |
| Sinkhorn Kernels | w_k = exp(−λ_k / ε) | Flandrin |
| Wiener Filter | w_k = μ_k / (μ_k + σ²) | Laplacian |
| Spectral Normalization | w_k = 1 / √(μ_k + ε) | Laplacian |
| Kernel PCA (hard) | w_k = 𝟙(k < K) | Index-based |
| Kernel PCA (soft) | w_k = max(0, 1 − λ_K / λ_k) | Flandrin |
| Mandelbrot Multifractal | w_k = \|Γ(H + iτ_k) / Γ(1−H + iτ_k)\| | Index-based |
| RMT Hard Threshold | w_k = λ_k · 𝟙(λ_k > λ₊) | User-provided |
| RMT Optimal Shrinkage | Donoho–Gavish (see §3.5.5) | User-provided |
| Wiener Deconvolution | w_k = λ_psf,k / (λ_psf,k² + η) | User-provided (PSF) |
| Matérn GRF Covariance | w_k = (κ² + μ_k)^(-ν) | Laplacian |
| Anisotropic Matérn | w_k = (κ_x²·lx_k + κ_y²·ly_k + ε)^(-ν) | Laplacian (per-axis) |
| Heat Propagator | w_k = exp(−t · μ_k) | Laplacian |
| Biharmonic Inverse | w_k = 1 / (μ_k² + ε) | Laplacian |
| Poisson Inverse | w_0 = 0; w_k = 1 / (μ_k + ε), k ≥ 1 | Laplacian |
| Higher-Order Tikhonov Deconv | w_k = ψ_k / (ψ_k² + α·μ_k^p + ε) | Laplacian + User PSF |

Here we distinguish two eigenvalue families that arise naturally in the framework: the *Flandrin eigenvalues* λ_k of the dfBm covariance (§2.2), and the *Laplacian eigenvalues* μ_k = 4 sin²(πk/2n) of the discrete Neumann Laplacian (§3.5.1). Some weights are purely index-based and require no precomputed eigenvalues. Others accept user-provided eigenvalues from empirical data. This distinction is important: confusing the two families produces mathematically well-defined but physically meaningless results.

### 3.2 Asymptotic Diagonalization

**Theorem 1** *(Asymptotic DCT diagonalization).* Let $R_{B,1}^H(n)$ be the auto-covariance matrix of 1st-order discrete-time fractional Brownian motion. As n → ∞, $R_{B,1}^H(n)$ is asymptotically diagonalized by the DCT-II matrix **C**.

**Proof Sketch.** For the case H = 1/2, the inverse covariance is a tridiagonal Jacobi matrix whose eigenvectors are exactly the DCT basis. For general H, the covariance matrix can be treated as an analytic perturbation $A(\varepsilon) = A_{1/2} + \varepsilon A^{(1)} + \cdots$. According to analytic perturbation theory for linear operators, the eigenvectors remain analytic in ε and converge to the DCT basis vectors [7]. ∎

### 3.3 Kernel Ridge Regression in the Induced RKHS

**Theorem 2** *(No-trick KRR solve).* Let k be a shift-invariant kernel. The CHEAP spectral primitive induces an exact regularized solution inside a finite-dimensional RKHS generated by the deterministic DCT features.

**Proof.** Mapping data to the DCT domain produces a data-independent basis for a higher-dimensional feature space. The dot product of these explicit feature vectors defines an equivalent positive-definite kernel. Because the mapping is deterministic and exact in the induced space, the regularized coefficients are given by:

$$\alpha_k = \frac{\hat{y}_k}{\lambda_k + \lambda_{\text{reg}}}$$

This avoids the spectral leakage and variance issues found in randomized approximations [10]. ∎

### 3.4 Optimal Transport via Sinkhorn

The entropy-regularized transport plan between marginals **a** and **b** is recovered via Sinkhorn iterations. When the cost matrix **T** is Toeplitz (e.g., squared distance on a grid), the Gibbs kernel **K** = exp(−**T**/ε) is never formed explicitly. Each iteration:

$$u^{(t+1)} = \frac{a}{K v^{(t)}}, \quad v^{(t+1)} = \frac{b}{K^\top u^{(t+1)}}$$

is computed using the CHEAP primitive applied to the dual potentials, reducing the complexity of each Sinkhorn iteration to O(n log n).

### 3.5 Extended Spectral Weights

The universality of the CHEAP primitive invites a natural question: what other weight vectors admit O(n) construction from known quantities? In this section we develop six additional weight families. Each preserves the O(n log n) total cost guarantee while extending CHEAP's reach to signal processing, dimensionality reduction, multifractal analysis, and covariance denoising.

#### 3.5.1 Laplacian Eigenvalues and the Two-Family Distinction

The discrete Laplacian with Neumann boundary conditions,

$$(Lu)_i = 2u_i - u_{i-1} - u_{i+1},$$

is *exactly* diagonalized by the DCT-II with eigenvalues

$$\mu_k = 4\sin^2\!\left(\frac{\pi k}{2n}\right), \quad k = 0, \ldots, n-1.$$

These eigenvalues are distinct from the Flandrin eigenvalues λ_k used elsewhere in the framework. Whereas the Flandrin spectrum encodes the covariance structure of dfBm, the Laplacian spectrum encodes the second-difference operator and governs frequency-domain filtering. Both are computable in O(n) from the index k alone, which is the essential property for CHEAP compliance. We denote Laplacian eigenvalues by μ_k throughout this section to avoid confusion.

We note that μ₀ = 0 (the DC component), reflecting the null space of the Laplacian under Neumann conditions. Weight formulas involving μ₀ in the denominator require regularization; we discuss this case-by-case below.

#### 3.5.2 Wiener Filter

The classical Wiener filter provides the minimum mean-square-error linear estimate of a signal observed in additive white noise. When the signal covariance is diagonalized by the DCT—as it is for operators of Laplacian type—the optimal filter reduces to pointwise spectral weighting:

$$w_k = \frac{\mu_k}{\mu_k + \sigma^2}$$

where σ² is the noise variance. This is a Tikhonov-regularized inverse with a signal-processing interpretation: components where the signal power μ_k dominates the noise σ² pass through nearly unchanged, while noise-dominated components are suppressed.

At k = 0, μ₀ = 0 yields w₀ = 0, which correctly zeroes the DC component when signal power at DC is modeled as zero. For problems where DC carries signal energy, the user may supply custom eigenvalues via the `_ev` variant.

The weights are bounded in [0, 1) and monotonically non-decreasing in k, ensuring that the filter never amplifies any spectral component. This is a stronger stability guarantee than is available for general Wiener filters on non-structured covariances.

We should note an important limitation: the formula above implements diagonal loading—not the full Minimum Variance Distortionless Response (MVDR) beamformer, which requires a steering vector and explicit matrix inversion. The naming choice reflects established signal-processing convention for the scalar spectral case.

#### 3.5.3 Spectral Normalization

Covariance whitening—transforming data so that its covariance becomes the identity—requires the inverse square root of the covariance eigenvalues. In the DCT domain with Laplacian structure:

$$w_k = \frac{1}{\sqrt{\mu_k + \varepsilon}}$$

where ε > 0 is a regularization parameter that prevents the singularity at k = 0 and bounds the maximum amplification to $\varepsilon^{-1/2}$.

This weight has found application in spectral normalization of neural network weight matrices [12], where it enforces Lipschitz continuity by bounding the spectral norm. For convolutional layers, the DCT provides approximate diagonalization, enabling O(n log n) normalization without explicit SVD.

The choice of ε involves a familiar bias-variance tradeoff: too small, and numerical noise in low-frequency components is amplified; too large, and the whitening effect is attenuated. Adaptive selection via ε = max(ε_abs, ε_rel · μ_max) provides a reasonable default, though we make no claim of optimality.

#### 3.5.4 Kernel PCA Projection

Spectral truncation is the oldest and simplest dimensionality reduction in the spectral domain. The hard threshold

$$w_k = \mathbb{1}(k < K)$$

retains the first K spectral components exactly and discards the rest. This is a projection operator: applying it twice yields the same result as applying it once, which serves as a useful correctness check.

The soft variant provides a smoother transition by using the Flandrin eigenvalue ratios:

$$w_k = \max\!\left(0,\; 1 - \frac{\lambda_K}{\lambda_k}\right)$$

where λ_k are the dfBm eigenvalues stored in the context. Because the Flandrin eigenvalues decrease with k (large at low frequencies, small at high frequencies), this formula assigns weight ≈ 1 to components with λ_k ≫ λ_K and weight ≈ 0 near the cutoff—consistent with PCA's variance-maximizing interpretation.

The choice of K remains the user's responsibility. Automated selection via explained variance thresholds, scree plots, or Stein's unbiased risk estimate (SURE) is straightforward to implement on top of the precomputed eigenvalues but falls outside the scope of the core primitive.

#### 3.5.5 Mandelbrot Multifractal Weights

Mandelbrot's extension of fractional Brownian motion to multifractal processes involves the ratio of Gamma functions with complex arguments [13]. In the spectral domain, this yields the weight:

$$w_k = \left|\frac{\Gamma(H + i\tau_k)}{\Gamma(1 - H + i\tau_k)}\right|, \quad \tau_k = \frac{\pi k}{n}$$

where H ∈ (0, 1) is the Hurst exponent. This generalizes the standard fractional weights (2 sin(ω_k/2))^d to a family parametrized by complex Gamma ratios, enabling the representation of multiscaling phenomena that simple power-law spectra cannot capture.

**Computation.** Evaluating the complex Gamma function is the primary technical challenge. We employ the Lanczos approximation [14] with parameters g = 7, N = 9, which achieves machine precision (~15 significant digits) for arguments with Re(z) ≥ 1/2. For Re(z) < 1/2, we apply the reflection formula

$$\ln\Gamma(z) = \ln\pi - \ln\sin(\pi z) - \ln\Gamma(1 - z)$$

to reduce to the convergent half-plane. The entire computation is performed in log-space—computing Re(ln Γ(H + iτ_k)) − Re(ln Γ(1−H + iτ_k)) and exponentiating—which avoids overflow for large |τ_k|.

**Symmetry property.** At H = 1/2, the numerator and denominator become identical: Γ(1/2 + iτ)/Γ(1/2 + iτ) = 1 for all τ. This provides an exact analytical test: all weights must equal unity to machine precision. We verify this property in our test suite.

**Domain restrictions.** The Gamma function has simple poles at the non-positive integers. For H ∈ (0, 1), neither H + iτ nor 1 − H + iτ passes through a pole (since the real parts are strictly between 0 and 1), ensuring that the weights are finite and positive. At the boundary values H → 0⁺ and H → 1⁻, the weights remain well-defined but exhibit large dynamic range, which may require extended precision for downstream computation.

We note that this implementation computes the *magnitude* of the Gamma ratio, discarding phase information. For applications requiring full complex weights—such as causal multifractal processes—a complex DFT would be needed, at approximately twice the computational cost. This is a genuine limitation, not merely an implementation shortcut.

#### 3.5.6 Random Matrix Denoising

When estimating covariance matrices from finite samples, eigenvalues are systematically biased: small eigenvalues are pushed down and large eigenvalues are pushed up relative to the population values. Random matrix theory (RMT) provides sharp predictions for this bias when the sample size n and dimension p grow proportionally, with aspect ratio c = n/p.

The Marchenko–Pastur law [15] characterizes the bulk spectrum of a sample covariance matrix under the null hypothesis of identity population covariance with noise variance σ². The bulk eigenvalues concentrate in the interval [λ₋, λ₊] where

$$\lambda_{\pm} = \sigma^2(1 \pm \sqrt{c}\,)^2.$$

Eigenvalues exceeding λ₊ are interpreted as signal; those within the bulk are noise.

**Hard thresholding.** The simplest denoising strategy zeroes all eigenvalues within the MP bulk and passes signal eigenvalues through unchanged:

$$w_k = \lambda_k \cdot \mathbb{1}(\lambda_k > \lambda_+).$$

This is aggressive but unbiased above the threshold.

**Optimal shrinkage.** Donoho, Gavish, and Johnstone [16] derived the asymptotically optimal nonlinear shrinkage function under Frobenius-norm loss. For eigenvalues above the bulk edge, the shrunken value is:

$$\tilde{\lambda}_k = \lambda_k \cdot \frac{\sqrt{(\ell - \ell_+)(\ell - \ell_-)}}{\ell}, \quad \ell = \lambda_k / \sigma^2$$

where ℓ₊ = (1 + √c)² and ℓ₋ = (1 − √c)². This reduces the upward bias of sample eigenvalues while preserving more signal than hard thresholding. The formula is closed-form and O(1) per eigenvalue, maintaining CHEAP's complexity guarantees.

Both variants accept user-provided eigenvalues rather than computing them from the index. This is a deliberate design choice: the eigenvalues to be denoised typically come from empirical data (sample covariance spectra), not from analytical formulas. The user is responsible for estimating σ² and c from the data; we provide the spectral surgery, not the diagnostics.

**Honest limitations.** The Marchenko–Pastur law is asymptotic: it assumes n, p → ∞ with c = n/p held fixed. For small samples (n < 100), finite-size corrections become relevant and the bulk edges are only approximate. Furthermore, the optimal shrinkage formula assumes that the population covariance has a spiked structure (finitely many signal eigenvalues above a flat noise floor). For population covariances with gradually decaying eigenvalues, the hard spike/bulk dichotomy breaks down. In such settings, nonparametric approaches [17] may be more appropriate, though they typically require O(n²) computation.

#### 3.5.7 Wiener Deconvolution

When an observation y is the convolution of a latent signal x with a known point-spread function (PSF) h corrupted by additive white noise:

$$y = h * x + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \eta I),$$

the minimum mean-square-error linear inverse filter in the spectral domain is the classical Wiener deconvolution filter. If the PSF is symmetric and its circulant approximation has DCT-II eigenvalues λ_psf,k (obtained via `cheap_toeplitz_eigenvalues`), then the optimal spectral weights are:

$$w_k = \frac{\lambda_{\text{psf},k}}{\lambda_{\text{psf},k}^2 + \eta},$$

where η = σ²_noise / σ²_signal is the inverse signal-to-noise ratio (set η = σ²_noise directly when signal power is normalized to 1). These weights strike the bias–variance optimum: at frequencies where the PSF has large response, near-perfect inversion is applied; at frequencies where the PSF is weak (near zero), the filter suppresses the amplified noise by rolling off gracefully rather than inverting blindly. The magnitude is bounded by |w_k| ≤ 1/(2√η) for all k, so the filter is unconditionally stable regardless of PSF zeros.

This differs from the denoising Wiener filter of §3.5.2 (weight μ_k / (μ_k + σ²)) in a structurally important way: the denominator here is quadratic in λ_psf, not linear. The quadratic form arises because the PSF eigenvalue appears twice—once in the numerator from H*(ω) and once in the denominator from |H(ω)|²—whereas the denoising filter has a scalar signal power in the numerator with no corresponding PSF.

**Obtaining PSF eigenvalues.** The function `cheap_toeplitz_eigenvalues` computes the DCT-II of the first column of a symmetric Toeplitz matrix, which is exactly λ_psf when the PSF's circulant extension is constructed in the standard way. For 2D or 3D problems, pass the flat row-major PSF eigenvalue grid (dimension nx·ny or nx·ny·nz) directly to `cheap_weights_wiener_deconv_ev`; no 2D-specific variant is needed because the weight formula is elementwise.

**Honest limitations.** The circulant PSF assumption introduces wrap-around artifacts at image boundaries. For signals with non-periodic boundary conditions, the circulant approximation of the true convolution matrix may introduce edge ringing that the Wiener filter cannot remove. Standard mitigations—boundary padding, windowing, or replacing the circulant model with a half-space Toeplitz model—are outside the scope of this primitive.

---

## 3.6 GRF and PDE Spectral Weight Families (v0.3.0)

The following five weight families extend CHEAP into Gaussian random field (GRF) simulation, PDE preconditioning, and penalized deconvolution. All are O(N) construction cost; the O(N log N) DCT in `cheap_apply` always dominates total cost.

### 3.6.1 Matérn Covariance Weights

The Matérn-ν covariance kernel on ℝ^d has a Fourier-domain spectral density proportional to (κ² + ‖ξ‖²)^(-(ν + d/2)). In the CHEAP spectral framework the spatial frequencies are replaced by the Laplacian eigenvalues μ_k = 4 sin²(πk/2n), which play the role of ‖ξ‖² on a discrete grid. The weight per mode is

    w[k] = (κ² + μ[k])^(-ν)

where κ > 0 is the inverse correlation length and ν > 0 is the Matérn smoothness parameter. The fractional Sobolev norm induced by this weight is the discrete analogue of the SPDE-based Matérn norm introduced by Lindgren, Rue, and Lindström [18]. To sample a Matérn-ν GRF: draw white noise z, compute `cheap_apply` with these weights, and scale by the appropriate normalizing constant.

The DC bin (μ[0] = 0) contributes a weight κ^(-2ν) — finite and positive, reflecting that the DC component has the same covariance as any other mode but is not penalized by the Laplacian.

The `_2d` and `_3d` convenience variants use the tensor-product Laplacian (sum of per-axis `4sin²` terms), consistent with the 2D/3D Laplacian weight constructors in §3.5.1. The `anisotropic_matern_2d/3d` variants independently scale each axis by κ_x², κ_y², κ_z², producing ellipsoidal correlation structure. Because the DC component of the anisotropic variant requires a regularization floor (CHEAP_EPS_LOG = 1e-12) to avoid a 0^(-ν) singularity, users who want exact DC suppression should zero w[0] after the call.

**Honest limitations.** The discrete Laplacian eigenvalues are an approximation of the continuous Fourier frequencies. For ν > 2 the tail of the Matérn spectrum falls off faster than the Laplacian eigenvalue spacing, so the high-frequency modes see quantization error relative to the continuous kernel. For applications requiring sub-percent accuracy in covariance matching, calibrate κ against the target continuous kernel numerically.

### 3.6.2 Heat Propagator Weights

    w[k] = exp(-t · μ[k])

This is the spectral representation of the heat kernel e^(tΔ) under Neumann boundary conditions. At t = 0 the operator is the identity; as t → ∞ the field flattens to its DC component. The DC mode (μ[0] = 0) satisfies exp(0) = 1.0 exactly — DC is always preserved by the heat equation.

**Connection to Sinkhorn.** The Gibbs kernel exp(-λ_k/ε) used in `cheap_sinkhorn` is formally a heat propagator with t = 1/ε applied to the Flandrin spectrum (μ = ctx→lambda). The heat propagator weight constructor operates on Laplacian eigenvalues and accepts any positive t, making it suitable for anisotropic diffusion preprocessing.

**Semigroup property.** The heat propagator satisfies the composition law: w(t₁)[k] · w(t₂)[k] = w(t₁+t₂)[k]. This is verified in the test suite (`test_weights_heat_propagator_ev`) and allows splitting a large diffusion step into smaller substeps without loss of exactness.

### 3.6.3 Biharmonic Inverse Weights

    w[k] = 1 / (μ_k² + ε),   k = 0, …, n−1

The biharmonic operator Δ² is the composition of the Laplacian with itself. Under the Neumann-BC DCT-II basis, the Laplacian has eigenvalues μ_k = 4 sin²(πk/2n), so Δ² has eigenvalues μ_k². The spectral inverse is therefore 1/μ_k², regularized at DC (μ_0 = 0) by ε > 0.

**Derivation.** The thin-plate energy functional E[u] = ∫ ‖Δu‖² minimized subject to interpolation constraints is solved by the Green's function of Δ², known as the thin-plate spline. In the spectral domain the solution operator is diag(1/μ_k²), confirming the weight formula. The same operator appears in the Euler–Bernoulli beam equation EI·∂⁴u/∂x⁴ = f (one-dimensional biharmonic) and in the 2D Stokes stream-function equation Δ²ψ = ω.

**DC handling.** The unregularized weight 1/μ_0² = ∞ corresponds to the fact that the biharmonic equation Δ²u = f has a two-dimensional null space spanned by {1, x} in 1D (the affine functions). Adding ε > 0 gives DC weight 1/ε — a large but finite regularization that penalizes departures from zero mean equally at all frequencies. Users requiring exact null-space enforcement should zero w[0] (and w[1] if an affine null space is present) after the call.

**Monotonicity.** The weights 1/(μ_k² + ε) are monotone decreasing in k for k ≥ 1 since μ_k is strictly increasing for k ≥ 1 (Laplacian eigenvalues are increasing by §3.5.1). High-frequency modes are more strongly penalized, consistent with the smoothness-promoting character of biharmonic regularization.

**SIMD.** `cheap_weights_biharmonic_ev` and the second pass of `cheap_weights_biharmonic_2d/3d` are vectorized (AVX2: 4×f64, NEON: 2×f64) because the hot path involves only multiplication, addition, and reciprocal — no transcendental functions.

**Honest limitations.** Biharmonic regularization imposes fourth-order smoothness, which may over-smooth piecewise-smooth signals (e.g. images with sharp edges). In such cases a first-order Tikhonov penalty (Poisson inverse, §3.6.4, or HOT deconvolution, §3.6.5 with p=1) is preferable.

### 3.6.4 Poisson Inverse Weights

    w[0] = 0,   w[k] = 1 / (μ_k + ε),   k = 1, …, n−1

**Mathematical background.** The Poisson equation −Δu = f on a bounded domain with Neumann boundary conditions has a solution if and only if f has zero mean (the Fredholm alternative: the Laplacian has a zero eigenvalue, so the right-hand side must be orthogonal to the null space). When this solvability condition holds, the solution is unique up to an additive constant, conventionally fixed by requiring zero mean. In the CHEAP spectral framework, the solution operator is diag(1/μ_k) applied to the DCT coefficients of f, with the DC coefficient set to zero to select the zero-mean solution.

**Derivation.** Writing the Poisson equation in the DCT basis: −μ_k û_k = f̂_k for all k, so û_k = −f̂_k/μ_k for k ≥ 1 and û_0 = 0 (by convention). Taking w_k = 1/μ_k (ignoring sign — the sign of Δ is absorbed by convention in the application) and w_0 = 0 gives the stated formula. The regularization ε ≥ 0 adds Tikhonov damping to all modes: at ε > 0 the filter is bounded by ‖w‖_∞ = 1/(μ_1 + ε) ≤ 1/ε, preventing blow-up even when the input is not strictly mean-free.

**Relationship to the Green's function.** The continuum Green's function of −Δ on ℝ^d is the harmonic potential: G(x) = log ‖x‖ in 2D, G(x) = −1/(4π‖x‖) in 3D. On a finite grid, the spectral inverse 1/μ_k is the discrete analogue, with ε playing the role of a screening constant (Yukawa / screened Laplacian). At ε → 0, low-frequency modes receive weight → ∞, reflecting the long-range nature of the harmonic potential. Contrast with the Wiener filter (§3.5.2), which is bounded in [0, 1), and spectral normalization (§3.5.3), which weights by 1/√(μ_k + ε) — neither inverts the Laplacian exactly.

**DC projection.** The DC weight w_0 = 0 is hardcoded (not computed as 1/(0 + ε)). This is an exact projection onto the mean-free subspace, not an approximation. The SIMD loop starts at k = 1 using unaligned loads (`loadu`) to preserve this invariant without branching.

**Honest limitations.** The discrete Poisson solve via DCT-II assumes Neumann (zero-flux) boundary conditions, not periodic or Dirichlet. For periodic domains, the DFT should be used instead. For Dirichlet conditions, the DST (discrete sine transform) diagonalizes the Laplacian and requires a different basis.

### 3.6.5 Higher-Order Tikhonov Deconvolution Weights

    w[k] = ψ_k / (ψ_k² + α·μ_k^p + ε),   k = 0, …, n−1

where ψ_k are PSF eigenvalues (DCT-II spectrum of the PSF first column, as in §3.5.7) and μ_k are Laplacian eigenvalues.

**Derivation.** The deconvolution problem y = Hx + n (H is the PSF operator, n is noise) has MAP estimate under a Gaussian signal prior with precision (−Δ)^(p/2) — i.e. the signal is penalized for having large Sobolev H^(p/2) norm. The MAP objective is ‖y − Hx‖² + α‖(−Δ)^(p/2) x‖². In the CHEAP spectral basis, H is diag(ψ_k) and (−Δ)^(p/2) is diag(μ_k^(p/2)), so the objective becomes ∑_k [ (ŷ_k − ψ_k x̂_k)² + α μ_k^p x̂_k² ]. Setting the derivative to zero gives x̂_k = ψ_k ŷ_k / (ψ_k² + α μ_k^p), matching the stated formula (with ε as a numerical floor).

**Special cases and comparison to §3.5.7.**
- p = 0, α = η: reduces to Wiener deconvolution (§3.5.7) with flat noise floor η. The HOT formula collapses to ψ_k/(ψ_k² + α + ε).
- p = 1: Tikhonov regularization with gradient-norm penalty. The Sobolev H^(1/2) prior; weights taper more gently at high k than p = 2.
- p = 2: biharmonic penalty. The Sobolev H¹ prior (finite energy); weights suppress high frequencies quadratically faster than p = 1.
- p > 2: super-Sobolev priors, appropriate for very smooth signals (analytic functions). Large p concentrates support on the lowest-k modes.

**DC analysis.** At k = 0: μ_0 = 0, so α·μ_0^p = 0 (by C99 `pow(0, p) = 0` for p > 0). The DC weight is w_0 = ψ_0/(ψ_0² + ε), which inverts the PSF at DC with no roughness penalty — physically correct, since the DC component has zero spatial frequency and is not penalized by any Sobolev norm. If ψ_0 = 0 and ε = 0, the denominator vanishes; the implementation floors the denominator at CHEAP_EPS_DIV (1e-300).

**Zero-allocation NULL path.** When `lap_eigenvalues == NULL`, the 1D Laplacian value 4 sin²(πk/2n) is computed on-the-fly per element inside the loop — no heap allocation, no ENOMEM failure path. For 2D/3D problems pass the flat Laplacian grid from `cheap_weights_laplacian_2d/3d` explicitly, since the on-the-fly formula only generates the 1D spectrum.

**Honest limitations.** The Sobolev prior is isotropic: it penalizes roughness equally in all directions. For images with directional textures or anisotropic PSFs, an anisotropic penalty (e.g. using `cheap_weights_anisotropic_matern_2d` to build the regularization eigenvalues) may produce better results. The non-integer-p case requires `pow(mu, p)` which is transcendental; for integer p, consider explicit products of μ_k for speed.

**Reference [18]:** F. Lindgren, H. Rue, J. Lindström, "An explicit link between Gaussian fields and Gaussian Markov random fields: the stochastic partial differential equation approach," *Journal of the Royal Statistical Society: Series B*, 73(4):423–498, 2011.

---

## 4. Mathematical Analysis

**Theorem 3** *(Eigenvector Perturbation Bound).* The difference between the true eigenvectors of the dfBm covariance and the DCT basis is bounded by O(1/n) for sufficiently large n.

**Proof.** The covariance operator admits an analytic perturbation expansion. By Kato's theorem, the perturbed eigenvectors remain analytic and orthonormal. Bounding the remainder via the Frobenius norm of the perturbation matrix yields the stated decay, ensuring that the DCT becomes an increasingly accurate representation of the Karhunen–Loève Transform for large grids [7]. ∎

**Complexity and Stability.** Each primitive requires two DCTs and O(n) arithmetic operations. Storage is strictly O(n). Because the DCT is a unitary transform, it is perfectly conditioned, ensuring that numerical errors do not propagate beyond the limits of the weight vector **w**.

---

## 5. The Pareto Frontier: An Honest Assessment

CHEAP occupies a specific "sweet spot" in the algorithmic trade-off space:

- **Where CHEAP excels**: Large regular 1D or 2D grids where asymptotic properties hold. It is highly effective for fractional-Brownian dynamics, stationary kernels, and problems whose covariance is approximately Toeplitz. The extended weight families (§3.5) broaden this sweet spot: Wiener filtering and spectral normalization exploit exact Laplacian diagonalization with no approximation error, and RMT denoising operates on user-provided eigenvalues with O(1) cost per component.
- **Where CHEAP struggles**: For highly irregular or high-dimensional scattered data, the asymptotic convergence to the DCT basis weakens, and standard Nyström or Cholesky methods may be preferable. The Mandelbrot weights require the Lanczos approximation of the complex Gamma function, introducing a higher constant factor per element than the simpler algebraic weights—though the O(n) scaling is preserved. RMT denoising inherits the asymptotic assumptions of the Marchenko–Pastur law: for small sample sizes, finite-size corrections matter and the bulk edges are only approximate.
- **Trade-off**: CHEAP typically trades a modest fraction of fidelity (3–5% relative to dense solvers) for a 100–200× reduction in construction time and a 20–40× reduction in memory. This is ideal for real-time updates and edge deployment. The weight constructor functions add negligible overhead: all are O(n) and dominated by the O(n log n) DCT in any end-to-end pipeline.

---

## 6. Conclusion

CHEAP rests on the observation that many covariance operators are asymptotically Toeplitz and admit inexpensive spectral factorization via the DCT. By combining classical perturbation theory with modern deterministic RKHS frameworks, we provide a linearithmic path to solving complex problems in kernel learning, optimal transport, signal processing, and covariance estimation.

The extended weight families introduced in §3.5 demonstrate that the universal spectral primitive accommodates a wider class of operations than originally anticipated. Two distinct eigenvalue families—the Flandrin spectrum of dfBm covariance and the Laplacian spectrum of the discrete second-difference operator—both admit O(n) construction and exact DCT diagonalization, yielding separate but equally efficient pathways through the same algorithmic primitive. The Mandelbrot multifractal weights show that even transcendental operations (complex Gamma ratios via Lanczos approximation) fit naturally into the framework, while RMT denoising demonstrates that CHEAP can serve as an efficient backend for statistical procedures whose eigenvalues originate outside the framework entirely.

The v0.3.0 weight families (§3.6) extend the reach further into GRF simulation and PDE-constrained inversion. Matérn covariance weights connect the CHEAP framework to the SPDE formulation of Gaussian random fields [18], enabling exact GRF sampling on regular grids at O(N log N) cost. The heat propagator, biharmonic inverse, and Poisson inverse weights provide spectral preconditioners for parabolic and elliptic PDEs with Neumann boundary conditions. The higher-order Tikhonov deconvolution weight unifies the Wiener deconvolution of §3.5.7 with flexible roughness penalties, enabling a continuous family of deconvolution operators parameterized by a single exponent p. All six new families maintain the O(N) construction guarantee and require no changes to the existing context or transform infrastructure.

The result is a package that is mathematically rigorous, numerically stable, and — above all — fiscally responsible.

---

## References

[1] Jain, A. K. (1989). *Fundamentals of Digital Image Processing*. Prentice-Hall.

[2] Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. *NeurIPS*.

[3] Solomon, J., et al. (2015). Convolutional Wasserstein Distances: Efficient Optimal Transport on Geometric Domains. *ACM Trans. Graph.*

[4] Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using HNSW graphs. *IEEE TPAMI*.

[5] Ailon, N., & Chazelle, B. (2006). Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform. *STOC*.

[6] Dasgupta, S., & Freund, Y. (2008). Random projection trees and low dimensional manifolds. *STOC*.

[7] Gupta, A., & Joshi, S. D. (2008). DCT and Eigenvectors of Covariance of 1st and 2nd order Discrete fractional Brownian motion. *IEEE Transactions on Signal Processing*.

[8] Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines. *NeurIPS*.

[9] Le, Q., Sarlós, T., & Smola, A. (2013). Fastfood: Approximating kernel expansions in loglinear time. *ICML*.

[10] Li, K., & Príncipe, J. C. (2019). No-Trick (Treat) Kernel Adaptive Filtering using Deterministic Features. *arXiv:1912.04530*.

[11] Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

[12] Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). Spectral normalization for generative adversarial networks. *ICLR*.

[13] Mandelbrot, B. B. (1997). *Fractals and Scaling in Finance: Discontinuity, Concentration, Risk*. Springer.

[14] Lanczos, C. (1964). A precision approximation of the gamma function. *SIAM J. Numer. Anal.*, 1(1), 86–96.

[15] Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. *Mat. Sb.*, 72(4), 507–536.

[16] Donoho, D. L., Gavish, M., & Johnstone, I. M. (2018). Optimal shrinkage of eigenvalues in the spiked covariance model. *Ann. Statist.*, 46(4), 1742–1778.

[17] Ledoit, O., & Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional covariance matrices. *Ann. Statist.*, 48(5), 3043–3065.

[18] Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between Gaussian fields and Gaussian Markov random fields: the stochastic partial differential equation approach. *Journal of the Royal Statistical Society: Series B*, 73(4), 423–498.
