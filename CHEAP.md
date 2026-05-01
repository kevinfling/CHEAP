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

where **w** ∈ ℝⁿ is a weight vector chosen according to the specific problem. The DCT and iDCT each cost O(n log n); the elementwise multiplication is O(n); and the construction of **w** itself is O(n) for every primitive in this paper. The total cost is therefore O(n log n) regardless of which primitive one selects. The expressive power of the framework lies entirely in the catalog of valid weight choices.

We organize this catalog as a *periodic table* in §4. Each "period" groups primitives by the underlying continuum operator they spectrally represent: covariance kernels (Period 1), parabolic propagators (Period 2), elliptic inverses (Period 3), hyperbolic propagators (Period 4), regularization and statistical denoising (Period 5), and fractional/multifractal weights (Period 6). The four classical applications — kernel ridge regression, Sinkhorn optimal transport, Gaussian sampling, and fractional differentiation — are recovered as specific weight choices spread across these periods.

Two distinct eigenvalue families arise naturally:

- The **Flandrin eigenvalues** λ_k of the dfBm covariance (§2.2), decreasing in k, which encode long-range temporal dependence.
- The **Laplacian eigenvalues** μ_k = 4 sin²(πk/2n) of the discrete Neumann Laplacian (§4.1.0), increasing in k, which encode local smoothness.

Many primitives also accept **user-provided eigenvalues** (e.g., from a sample covariance spectrum or a measured PSF) via `_ev` constructor variants. A small number are purely **index-based**, requiring no precomputed spectrum at all. Confusing these families produces mathematically well-defined but physically meaningless results, and we are explicit throughout about which family each primitive expects.

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

---

## 4. The Spectral Periodic Table

The remainder of this paper develops a comprehensive catalog of spectral weight families, organized by the underlying mathematical structure rather than by application domain. We borrow the metaphor of the periodic table from chemistry: the *period* of a primitive denotes the class of continuum operator it represents, while the *base* (Flandrin, Laplacian, index, user-provided) plays the role of group. Periods and bases together provide a two-dimensional taxonomy that exposes structural relationships invisible in a flat list.

The intent is to be exhaustive. We document not only the families currently shipped in the library but also those that admit O(n log n) construction in principle, including primitives whose stability or representational constraints prevent shipping (the **Hazard Log**, §4.7). For each primitive we provide: closed-form weight formula, derivation from the underlying continuum object, DC/edge handling, monotonicity and stability bounds, and an honest discussion of limitations.

### 4.0 Master Table

The following table summarizes every primitive in the periodic table. Status legend: **S** = shipped in the current release; **P** = planned for a future release; **C** = composable in one line from existing `_ev` constructors; **R** = rejected (see §4.7). Each primitive name links to its detail subsection. C99 function signatures and SIMD hot-path snippets for every shipped and planned primitive are collected in [APPENDIX.md](APPENDIX.md).

| Period | Primitive | Formula | Base | Status |
|---|---|---|---|---|
| 1 | [Matérn (SPDE)](#411-matérn-covariance) | (κ² + μ_k)^{−ν} | Laplacian | S |
| 1 | [Anisotropic Matérn](#411-matérn-covariance) | (κ_x²·μ_x + κ_y²·μ_y + ε)^{−ν} | Laplacian | S |
| 1 | [Powered Exponential](#412-powered-exponential) | exp(−(ℓ²μ_k)^β), β∈(0,2] | Laplacian | P |
| 1 | [Cauchy / Rational-Quadratic](#413-cauchy--rational-quadratic) | (1 + κ²μ_k)^{−ν} | Laplacian | P |
| 1 | [Squared Exponential (RBF)](#414-squared-exponential-rbf) | exp(−ℓ²μ_k/2) | Laplacian | C |
| 1 | [Whittle](#415-whittle-kernel) | (κ² + μ_k)^{−1} | Laplacian | C |
| 1 | [Bessel Potential](#416-bessel-potential) | (1 + μ_k)^{−s/2} | Laplacian | P |
| 1 | [GP-Derivative Covariance](#417-gp-derivative-covariance) | μ_k^m · (κ² + μ_k)^{−ν} | Laplacian | P |
| 1 | [Rough Volatility](#418-rough-volatility) | μ_k^{−H−1/2} | Flandrin or Laplacian | P |
| 2 | [Heat Propagator](#421-heat-propagator) | exp(−tμ_k) | Laplacian | S |
| 2 | [Implicit Euler (heat)](#422-implicit-euler-heat-step) | (1 + tμ_k)^{−1} | Laplacian | C |
| 2 | [Fractional Heat](#423-fractional-heat) | exp(−tμ_k^s) | Laplacian | C |
| 2 | [Ornstein–Uhlenbeck](#424-ornstein–uhlenbeck-propagator) | exp(−θt) + (1−exp(−2θt))/(2θ) · w_k | Laplacian | P |
| 2 | [Cahn–Hilliard (linearized)](#425-cahn–hilliard-linearized) | 1/(1 + t(εμ_k + μ_k²/ε)) | Laplacian | P |
| 3 | [Poisson Inverse](#431-poisson-inverse) | 1/μ_k (DC=0) | Laplacian | S |
| 3 | [Biharmonic Inverse](#432-biharmonic-inverse) | 1/μ_k² (DC=0) | Laplacian | S |
| 3 | [Higher-Order Tikhonov Deconv](#433-higher-order-tikhonov-deconvolution) | ψ_k/(ψ_k² + αμ_k^p) | PSF + Laplacian | S |
| 3 | [Helmholtz / Yukawa](#434-helmholtz--yukawa-inverse) | 1/(κ² + μ_k) | Laplacian | P |
| 3 | [Fractional Integral / Riesz Potential](#435-fractional-integral--riesz-potential) | μ_k^{−s} (DC=0) | Laplacian | P |
| 3 | [Tikhonov H^p Prior (no PSF)](#436-tikhonov-hp-prior-no-psf) | 1/(1 + αμ_k^p) | Laplacian | P |
| 4 | [Wave Propagator](#441-wave-propagator) | cos(ct√μ_k) | Laplacian | P |
| 4 | [Damped Wave / Klein–Gordon](#442-damped-wave--klein–gordon) | exp(−γt) cos(t√(μ_k + m²)) | Laplacian | P |
| 5 | [Wiener (S/(S+N))](#451-wiener-filter) | μ_k/(μ_k + σ²) | Laplacian or User-ev | S |
| 5 | [Wiener Deconvolution](#452-wiener-deconvolution) | ψ_k/(ψ_k² + η) | PSF | S |
| 5 | [Spectral Normalization](#453-spectral-normalization) | 1/√(μ_k + ε) | Laplacian or User-ev | S |
| 5 | [Kernel PCA (hard)](#454-kernel-pca-projection-hard-and-soft) | 𝟙(k < K) | Index | S |
| 5 | [Kernel PCA (soft)](#454-kernel-pca-projection-hard-and-soft) | max(0, 1 − λ_K/λ_k) | Flandrin | S |
| 5 | [RMT Hard Threshold](#455-random-matrix-denoising-hard-and-optimal) | λ_k · 𝟙(λ_k > λ_+) | User-ev | S |
| 5 | [RMT Optimal Shrinkage (DGJ)](#455-random-matrix-denoising-hard-and-optimal) | λ_k · √((ℓ−ℓ_+)(ℓ−ℓ_−))/ℓ | User-ev | S |
| 5 | [RMT BBP Spiked](#456-rmt-bbp-spiked-estimator) | (λ_k > λ_+) ? (λ_k² − cσ⁴)/λ_k : 0 | User-ev | P |
| 5 | [Geman–McClure Robust Shrinkage](#457-geman–mcclure-robust-shrinkage) | λ_k²/(λ_k² + σ²) | User-ev | P |
| 5 | [Cesàro / Fejér Window](#458-cesàro--fejér-window) | max(0, 1 − k/K) | Index | C |
| 5 | [Lanczos σ-Factor](#459-lanczos-σ-factor) | sinc(πk/N) | Index | C |
| 6 | [Fractional Differentiation](#461-fractional-differentiation) | (2 sin(ω_k/2))^d | Index | S |
| 6 | [Fractional Laplacian](#462-fractional-laplacian) | μ_k^s | Laplacian | S |
| 6 | [Sobolev Synthesizer](#463-sobolev-synthesizer) | (1 + μ_k)^{s/2} | Laplacian | P |
| 6 | [Kolmogorov 5/3 Turbulence](#464-kolmogorov-53-turbulence) | μ_k^{−5/6} | Laplacian | P |
| 6 | [Mandelbrot Multifractal](#465-mandelbrot-multifractal-weights) | \|Γ(H+iτ_k)/Γ(1−H+iτ_k)\| | Flandrin | S |
| HZ | [Backward Heat](#471-backward-heat-equation) | exp(+tμ_k) | Laplacian | R |
| HZ | [Real Schrödinger](#472-real-schrödinger--complex-wave-propagators) | exp(−itμ_k) | Laplacian | R |
| HZ | [Turing Growth](#473-turing-growth-reaction–diffusion-linearization) | exp(t(a − bμ_k − c/μ_k)) | Laplacian | R |
| HZ | [Riesz Transform](#474-riesz-transform-vector-valued) | k_d/√μ_k | Per-axis index | R |
| HZ | [Spectral Entropy](#475-spectral-entropy) | −λ_k ln λ_k | Spectrum | R |
| HZ | [Ideal Brick-Wall](#476-ideal-brick-wall-filter) | 𝟙(k < K) | Index | R (Gibbs) |
| HZ | [Van Cittert](#477-van-cittert-iteration) | 1 − (1 − ψ_k)^n | PSF | R |

*Composability note:* Status **C** primitives ship as one-line recipes inside their detail subsection (e.g. `w[k] = 1.0 / (kappa*kappa + mu[k])` for Whittle), composable on top of the shipped `cheap_weights_*_ev` constructors. They do not ship as standalone functions because their bodies fit on the call site without library overhead.

*Base column:* "Laplacian" denotes the discrete Neumann Laplacian spectrum μ_k = 4 sin²(πk/2n); "Flandrin" denotes the dfBm covariance spectrum (§2.2); "Index" denotes a closed-form function of k requiring no precomputed eigenvalues; "User-ev" denotes a primitive whose `_ev` constructor accepts arbitrary user-supplied eigenvalues (typically from a sample covariance, an empirical PSF, or a foreign spectrum); "PSF" denotes the DCT-II spectrum of a measured point-spread function. "Laplacian or User-ev" entries default to Laplacian via the base constructor and accept a custom spectrum via `_ev`.

---

### 4.1 Period 1 — GRF / Covariance Kernels

Period 1 collects positive-definite spectral multipliers that arise as the *Fourier symbol* of an isotropic stationary covariance kernel. The defining property of every primitive in this period is that the inverse DCT of the weight vector — equivalently, the convolution kernel diag(w) acts as in the spatial domain — is a valid covariance function. Operationally, given a Period 1 weight w, sampling the corresponding Gaussian random field reduces to: draw white noise z ∼ 𝒩(0, I), apply `cheap_apply` with weight √w (the *spectral square root*), and the result has covariance diag(w) in the DCT basis.

We begin by establishing the Laplacian eigenvalue family that underpins most of Periods 1–4 and 6.

#### 4.1.0 Laplacian Eigenvalues and the Two-Family Distinction

The discrete Laplacian with Neumann boundary conditions,

$$(Lu)_i = 2u_i - u_{i-1} - u_{i+1},$$

is *exactly* diagonalized by the DCT-II with eigenvalues

$$\mu_k = 4\sin^2\!\left(\frac{\pi k}{2n}\right), \quad k = 0, \ldots, n-1.$$

These eigenvalues are distinct from the Flandrin eigenvalues λ_k introduced in §2.2. Whereas the Flandrin spectrum encodes the covariance structure of dfBm and is *decreasing* in k (most variance at low frequencies), the Laplacian spectrum encodes the second-difference operator and is *increasing* in k (zero at DC, maximum at Nyquist). Both are computable in O(n) from the index k alone, which is the essential property for CHEAP compliance, but they are not interchangeable: weights designed for one family produce mathematically well-defined but physically meaningless results when applied to the other.

We note that μ₀ = 0, reflecting the null space of the Laplacian under Neumann conditions — namely, the constant functions. Weight formulas involving μ₀ in the denominator therefore require explicit DC handling, which we discuss case-by-case throughout the periodic table.

In 2D and 3D, the Laplacian spectrum is the *sum* (not product) of per-axis eigenvalues:

$$\mu_{j,k} = 4\sin^2\!\left(\frac{\pi j}{2 n_x}\right) + 4\sin^2\!\left(\frac{\pi k}{2 n_y}\right),$$

and analogously in 3D. By contrast, tensor-product covariance kernels (Period 1) typically use the *product* of per-axis spectra. The constructor names reflect this distinction: `cheap_weights_laplacian_2d` returns the sum, while `cheap_init_2d` populates `ctx->lambda` with the Flandrin tensor product.

#### 4.1.1 Matérn Covariance

The Matérn-ν covariance kernel on ℝ^d has continuum Fourier-domain spectral density

$$\hat{k}_{\text{Mat}}(\xi) \propto (\kappa^2 + \|\xi\|^2)^{-(\nu + d/2)},$$

where κ > 0 is the inverse correlation length and ν > 0 controls smoothness (ν = ∞ recovers the squared-exponential limit; ν = 1/2 gives the exponential kernel). On a discrete grid, ‖ξ‖² is replaced by the Laplacian eigenvalue μ_k, giving

$$w_k = (\kappa^2 + \mu_k)^{-\nu}.$$

The exponent absorbs the dimension-dependent shift d/2 into the user-supplied ν, which we treat as a free smoothness parameter. The fractional Sobolev norm induced by this weight is the discrete analogue of the SPDE-based Matérn norm of Lindgren, Rue, and Lindström [18], who showed that Matérn fields are *exactly* the stationary solutions of the fractional SPDE (κ² − Δ)^{(ν+d/2)/2} u = 𝒲, where 𝒲 is white noise. The CHEAP spectral primitive realizes this SPDE in a single elementwise multiplication.

**DC handling.** μ₀ = 0 yields w₀ = κ^{−2ν}, finite and positive. The DC component of a Matérn field has the same per-mode variance as any other mode but is not penalized by the Laplacian; this is the correct behavior for a translation-invariant covariance kernel.

**2D/3D extensions.** The `_2d` and `_3d` variants use the *summed* Laplacian (§4.1.0). The `anisotropic_matern_2d/3d` variants weight each axis independently:

$$w_{j,k} = (\kappa_x^2 \mu_x[j] + \kappa_y^2 \mu_y[k] + \varepsilon)^{-\nu},$$

producing ellipsoidal correlation structure with correlation lengths 1/κ_x and 1/κ_y. The ε floor prevents 0^{−ν} at DC; users requiring exact DC suppression should zero w[0] after the call.

**Honest limitations.** The discrete Laplacian eigenvalues approximate the continuous Fourier frequencies only to leading order in the grid spacing. For ν > 2 the Matérn spectrum decays faster than the spacing of {μ_k}, so high-frequency modes see quantization error relative to the continuum kernel. Sub-percent covariance fidelity requires numerical calibration of κ.

**Covariance error (proved).** Combining the eigenvector perturbation bound of Gupta–Joshi [7] (eigenvectors converge to the DCT basis at rate O(1/n) in operator norm, uniformly for H ∈ [0.1, 0.9]) with the Kato remainder for the rescaled Laplacian symbol gives ‖R_exact − R_cheap‖_F / ‖R_exact‖_F = O(1/n). The constant degrades as κ → 0 (long correlation lengths sample the boundary more strongly) and as ν → 0 (rougher fields concentrate energy at high k where the Laplacian/Fourier eigenvalue mismatch is largest).

**Library status.** Shipped (`cheap_weights_matern_ev`, `cheap_weights_anisotropic_matern_2d/3d`).

#### 4.1.2 Powered Exponential

The powered-exponential family generalizes the Gaussian and Laplace kernels via a stretched-exponential decay:

$$w_k = \exp\!\left(-(\ell^2 \mu_k)^{\beta}\right), \quad \beta \in (0, 2].$$

The parameter β controls tail thickness: β = 1 recovers the squared-exponential limit (heavy at low frequencies, rapid decay at high), β = 1/2 gives an exponential decay in √μ (moderate-tailed), and β → 0⁺ approaches the constant kernel. β = 2 yields a "super-Gaussian" decay rarely used in practice but valid as a positive-definite symbol on the half-line.

**Derivation.** The continuum spectral density exp(−(ℓ²‖ξ‖²)^β) is the Fourier symbol of the *stable* covariance kernel of order 2β; for β ≤ 1 it is positive-definite by Bochner's theorem, with the proof relying on the subordinator construction of stable processes. For β ∈ (1, 2] positivity holds on the half-line ξ ≥ 0 but breaks down on ℝ; on the discrete DCT lattice, where μ_k ≥ 0 always, β ∈ (0, 2] is admissible.

**DC handling.** w₀ = exp(0) = 1 exactly — the kernel has unit variance at DC.

**Stability.** The weights are bounded in (0, 1], so the implied filter is contractive at every frequency. No regularization is required.

**Honest limitations.** β > 1 produces a kernel that is *not* a smoothness-weighted Sobolev norm and admits no SPDE realization; the weight is mathematically valid but interpretively unusual. Users selecting β > 1 should verify that downstream procedures (e.g., GP regression, sampling) do not implicitly assume a Matérn-style fractional Sobolev structure.

**Covariance error (heuristic).** For β = 1 the powered-exponential coincides with the squared-exponential and inherits the same O(1/n) Frobenius bound as Matérn (proved). For β ≠ 1, no operator-norm proof is known to us; numerical experiments suggest the same O(1/n) rate for β ∈ [1/2, 1] but with constants that grow as β → 0 (slow tail decay) or β → 2 (super-Gaussian tails). We mark this as conjectural pending a formal Kato-style remainder.

**Library status.** Planned. Composable today as `w[k] = exp(−pow(ell*ell*mu[k], beta))`.

#### 4.1.3 Cauchy / Rational-Quadratic

The rational-quadratic kernel arises as a continuous mixture of squared-exponential kernels with inverse-Gamma-distributed length scales:

$$w_k = (1 + \kappa^2 \mu_k)^{-\nu}.$$

Equivalent spelling: with α = ν and ℓ² = 1/(2κ²), the formula matches Rasmussen and Williams's standard parametrization. As ν → ∞ with κ²ν fixed, the kernel converges to squared exponential; for finite ν, the tails decay as μ^{−ν}, making this a heavier-tailed alternative for problems where length scales are themselves uncertain.

**DC handling.** w₀ = 1 — the DC mode receives unit weight, like Matérn.

**Stability.** Weights monotone-decreasing in μ_k and bounded in (0, 1]. No regularization needed.

**Relationship to Matérn.** The Cauchy weight differs from Matérn (κ² + μ_k)^{−ν} only in the placement of the κ² factor. Matérn admits an SPDE interpretation; Cauchy/RQ does not, but it admits the scale-mixture interpretation (mixing squared-exponential kernels over inverse-Gamma length scales), which Matérn lacks.

**Honest limitations.** The Cauchy kernel is *not* a Bessel potential and does not correspond to any local differential operator; its scale-mixture origin makes it appropriate for nonparametric Bayesian settings where the length scale is treated as a nuisance parameter, but inappropriate for settings requiring Markov field structure.

**Covariance error (heuristic).** No SPDE realization, so Gupta–Joshi does not apply directly; however, the weight is a smooth bounded function of μ_k with no DC singularity, and a routine perturbation argument yields the same O(1/n) Frobenius rate as Matérn. We mark as conjectural.

**Library status.** Planned.

#### 4.1.4 Squared Exponential (RBF)

The Gaussian / squared-exponential kernel is the canonical choice in Gaussian process regression:

$$w_k = \exp\!\left(-\ell^2 \mu_k / 2\right).$$

**Derivation.** The continuum spectral density of the Gaussian kernel exp(−‖x‖²/(2ℓ²)) on ℝ^d is itself Gaussian: exp(−ℓ²‖ξ‖²/2). Substituting μ_k for ‖ξ‖² gives the discrete weight directly. Equivalently, this is the powered-exponential primitive (§4.1.2) at β = 1.

**DC handling.** w₀ = 1.

**Stability.** Weights bounded in (0, 1], monotone decreasing.

**Honest limitations.** The Gaussian kernel produces *infinitely smooth* sample paths almost surely. For most physical and statistical applications this is unrealistic; Matérn at moderate ν is preferred. We document the Gaussian primitive for completeness and because it is composable as a powered-exponential at β = 1.

**Covariance error (proved).** As a special case of Matérn (κ → ∞ limit with appropriate rescaling), the squared-exponential inherits the O(1/n) Frobenius bound. The constant is small in practice because the spectrum decays rapidly, leaving most of the operator norm in low-k modes where the Laplacian/Fourier mismatch is smallest.

**Library status.** Compose: `w[k] = exp(-0.5 * ell*ell * mu[k])`.

#### 4.1.5 Whittle Kernel

The Whittle kernel is the spectral inverse of the screened Laplacian:

$$w_k = (\kappa^2 + \mu_k)^{-1}.$$

This is the special case ν = 1 of Matérn (§4.1.1). The corresponding spatial covariance is the Bessel function K₀(κr) in 2D, giving Markovian (nearest-neighbor) conditional independence properties — the foundation of the SPDE approach to GMRFs [18]. Numerically, the Whittle kernel is the most efficient Matérn variant: no `pow` call, just one division per element.

**DC handling.** w₀ = 1/κ², finite.

**Covariance error (proved).** The Whittle kernel is Matérn at ν = 1, so the same O(1/n) Frobenius bound applies with the smallest known constant in the family — division-only construction means no `pow` cancellation error.

**Library status.** Compose: `w[k] = 1.0 / (kappa*kappa + mu[k])`. Equivalent to Helmholtz inverse (§4.3.4); we list it twice because the Period 1 (covariance) and Period 3 (PDE inverse) interpretations are distinct, even though the formula is identical.

#### 4.1.6 Bessel Potential

The Bessel potential of order s is the Fourier multiplier (1 + |ξ|²)^{−s/2}. On the DCT lattice:

$$w_k = (1 + \mu_k)^{-s/2}.$$

This is the Matérn primitive at κ = 1 with ν = s/2, and its action defines the Bessel potential space H^s(ℝ^d), the standard fractional Sobolev space used in PDE analysis. The Bessel potential is the *low-frequency-stable* alternative to the pure fractional Laplacian μ_k^{s/2} (§4.6.2), which has a singularity at DC for negative s.

**DC handling.** w₀ = 1 exactly. The "+1" inside the parentheses regularizes the DC mode, making this primitive numerically robust without an explicit ε.

**Mapping to Matérn.** Bessel potential is Matérn with κ = 1, ν = s/2. We document it as a separate primitive because its standard parametrization in PDE/functional-analysis literature uses s ∈ ℝ (potentially negative) directly, rather than ν > 0 with κ separate.

**Honest limitations.** The unit length scale κ = 1 is a hard convention; problems with intrinsic length scale ℓ ≠ 1 should use the full Matérn primitive instead. Negative s gives a *gain* (analyzer); positive s gives a smoother (synthesizer).

**Covariance error (proved).** As a κ = 1 specialization of Matérn, the same O(1/n) Frobenius bound applies; the "+1" inside the parentheses keeps the constant uniformly bounded for all s ∈ ℝ (no DC singularity, no κ → 0 degeneration).

**Library status.** Planned. Composable as Matérn with κ = 1.

#### 4.1.7 GP-Derivative Covariance

Given a Matérn-ν Gaussian process X(t), its m-th derivative ∂^m X (when it exists in L²) is itself a stationary GP with spectral density |ξ|^{2m} times the Matérn density:

$$w_k = \mu_k^{m} \cdot (\kappa^2 + \mu_k)^{-\nu}.$$

This primitive enables direct sampling of GP derivatives and joint sampling of (X, ∂X, ∂²X, …) by elementwise multiplication of the appropriate spectral square roots. It also appears in *physics-informed* GP regression, where derivative observations are conditioned on alongside function values.

**DC handling.** μ₀^m = 0 for m ≥ 1, so w₀ = 0 exactly — derivatives have no DC content, which is correct (the derivative of a constant is zero).

**Edge of validity.** The weight is mathematically a valid spectral multiplier for any real m ≥ 0 and any ν > 0, but its interpretation as the *covariance of the m-th derivative of a Matérn field* requires (a) m to be a non-negative integer, and (b) m < ν − d/2 + 1/2 so that the m-th derivative exists in L². Outside this range, w_k still defines a positive-definite covariance, but it is no longer the derivative covariance — it is just a *high-pass-tilted Matérn*. We document this freely.

**Covariance error (proved within the validity range).** When (a) and (b) hold, the GP-derivative covariance is the spectral product of a polynomial multiplier μ_k^m with the parent Matérn weight; by submultiplicativity of operator norm, the Frobenius bound is O(m/n) — proportionally larger than Matérn but still O(1/n) for fixed m. Outside the validity range, the bound holds for the resulting positive-definite kernel but no longer connects to a derivative process.

**Library status.** Planned.

#### 4.1.8 Rough Volatility

In quantitative finance, *rough* stochastic volatility models (Bayer–Friz–Gatheral; Gatheral–Jaisson–Rosenbaum) require simulation of fractional Brownian motion with Hurst exponent H ≪ 1/2. The corresponding spectral weight on the Flandrin or Laplacian basis is

$$w_k = \mu_k^{-H - 1/2}, \quad H \in (0, 1/2).$$

The exponent −H − 1/2 lies in (−1, −1/2), giving a singularity at DC that is integrable in the spatial domain (so the implied covariance is finite) but requires regularization in the discrete setting.

**DC handling.** μ₀^{−H−1/2} diverges; we add an ε floor: `w[k] = pow(mu[k] + eps, -H - 0.5)`. The choice of ε determines the effective low-frequency cutoff, equivalent to a cap on the longest correlation time.

**Connection to fBm.** For H ∈ (0, 1/2), this weight produces the spectral density of dfBm increments under the *long-memory* parameterization. Sample paths are α-Hölder continuous for any α < H, hence "rougher" than standard Brownian (H = 1/2).

**Equivalence to Davies–Harte / circulant embedding.** The standard Davies–Harte algorithm for *exact* simulation of fBm on a uniform grid embeds the n × n covariance into a 2n × 2n circulant whose spectrum is computed via FFT. On the DCT-II basis, the same construction reduces to the spectral square root of μ_k^{−H−1/2} applied to white noise — that is, `cheap_apply` with weight √w followed by appropriate normalization. The two algorithms produce statistically identical samples (up to boundary handling: Davies–Harte uses periodic extension, CHEAP uses Neumann-symmetric extension), and the spectral construction inherits Davies–Harte's *exactness* property: the covariance of the simulated path matches the target covariance to machine precision, not asymptotically.

**Malliavin / pathwise sensitivities.** The spectral square root of the rough-volatility weight is also the operator that maps standard Gaussian increments to the simulated path. As a result, pathwise derivatives of payoffs with respect to model parameters (Malliavin Greeks: ∂_H, ∂_κ where κ enters via Bessel-stabilized variants) can be computed by reusing the same DCT pipeline applied to the integration-by-parts weight. This *shares infrastructure* with the simulation — one DCT plan, one workspace — rather than coming "for free": each Greek still requires its own forward-and-backward pass through the path and an integration-by-parts factor that depends on the specific payoff. The cost per Greek is one additional `cheap_apply`.

**Honest limitations.** The Flandrin eigenvalues λ_k of dfBm provide an alternative basis for this primitive (§2.2). For pure spectral simulation of rough volatility on a regular grid, the Laplacian basis suffices and is faster to construct. For statistical inference (parameter estimation from observed paths), the Flandrin basis is more directly interpretable. The ε floor at DC is a model choice equivalent to capping the longest correlation horizon; for finance applications calibrated to short trading windows this cap is innocuous, but for multi-decade volatility studies the cap should be set explicitly rather than left at default.

**Covariance error (proved on the Flandrin basis, heuristic on Laplacian).** On the Flandrin basis the rough-volatility weight is exact (the eigenvalues are λ_k themselves up to scaling), so the only error source is the Gupta–Joshi eigenvector perturbation: O(1/n) Frobenius. On the Laplacian basis the additional symbol replacement introduces a second O(1/n) term; the combined error is still O(1/n) but with a constant proportional to the spectral mismatch at low k (worst near μ → 0, where the −H − 1/2 exponent is most singular). The ε floor explicitly bounds the constant.

**Library status.** Planned.

---

### 4.2 Period 2 — Diffusion / Parabolic Operators

Period 2 collects spectral multipliers that arise as time-evolution operators of *parabolic* PDEs. Every primitive in this period has the form e^{−tA(μ_k)} for some non-negative function A: thus each is bounded in (0, 1], contractive, and parametrized by a non-negative time t. The defining structural feature is the *semigroup property*: composing two time steps multiplies the spectral weights, w(t₁) · w(t₂) = w(t₁ + t₂).

#### 4.2.1 Heat Propagator

The fundamental parabolic primitive is the heat kernel:

$$w_k = \exp(-t \mu_k), \quad t \geq 0.$$

This is the spectral representation of e^{tΔ} under Neumann boundary conditions. At t = 0 the operator is the identity; as t → ∞ it projects onto the DC component. The DC mode μ₀ = 0 satisfies exp(0) = 1 exactly: DC is preserved by the heat equation, reflecting conservation of mass under Neumann conditions (the spatial integral of the solution is constant in time).

**Connection to Sinkhorn.** The Gibbs kernel exp(−λ_k/ε) used in `cheap_sinkhorn` is formally a heat propagator with t = 1/ε applied to the *Flandrin* spectrum. Period 2 primitives operate on the Laplacian spectrum and accept any positive t, making them suitable for image filtering, anisotropic diffusion preprocessing, and JKO gradient flows on Wasserstein space.

**Semigroup.** w(t₁)[k] · w(t₂)[k] = w(t₁+t₂)[k] holds elementwise to machine precision. This enables splitting a large diffusion step into smaller substeps without loss of exactness, which is useful for embedding the heat propagator inside implicit-explicit (IMEX) splittings of nonlinear reaction-diffusion equations.

**Stability.** Weights bounded in (0, 1]. No regularization needed; the propagator is unconditionally stable for any t ≥ 0 and any grid spacing.

**Honest limitations.** The Neumann BC is hard-coded by the choice of DCT-II. For periodic domains, use the DFT; for Dirichlet conditions, use the DST.

**Library status.** Shipped (`cheap_weights_heat_propagator_ev`).

#### 4.2.2 Implicit Euler Heat Step

The implicit Euler discretization of ∂_t u = Δu with step size t solves (I − tΔ) u^{n+1} = u^n, giving the spectral weight

$$w_k = (1 + t \mu_k)^{-1}.$$

This is the *stable but only first-order accurate* alternative to the exact heat propagator. The advantages are (a) cheaper construction (no transcendental function — one division per element), and (b) compatibility with Jacobi/Gauss-Seidel-style relaxation when the spatial discretization is not exactly DCT-diagonal.

**DC handling.** w₀ = 1, identical to the exact heat propagator.

**Comparison to exact propagator.** Implicit Euler weights satisfy (1 + tμ)^{−1} ≤ exp(−tμ) for all t, μ ≥ 0, with equality only at μ = 0. Implicit Euler thus *over-smooths* relative to the exact heat equation: a single step damps high frequencies more than the true PDE would. For equal smoothing fidelity, multiple smaller steps of implicit Euler are required, which compose by w(t)^N applied N times — *not* the same as w(Nt) (the semigroup fails).

**Higher-order extension.** Crank–Nicolson gives w_k = (1 − tμ_k/2) / (1 + tμ_k/2), second-order accurate in t. This is also composable, though stability requires t·μ_max < 2.

**Library status.** Compose: `w[k] = 1.0 / (1.0 + t * mu[k])`.

#### 4.2.3 Fractional Heat

The fractional heat equation ∂_t u = −(−Δ)^s u models *anomalous diffusion* with sub-diffusive (s < 1) or super-diffusive (s > 1) behavior:

$$w_k = \exp(-t \mu_k^s).$$

For s = 1, this reduces to the standard heat propagator. For s ∈ (0, 1), the propagator describes Lévy-flight-like long jumps interspersed with local diffusion, relevant in continuous-time random walk models, plasma physics, and certain biological transport processes.

**DC handling.** μ₀^s = 0 for any s > 0, so w₀ = 1 exactly.

**Stability.** Weights in (0, 1]. The semigroup property holds exactly: w(t₁) · w(t₂) = w(t₁+t₂).

**Honest limitations.** For non-integer s, the underlying operator (−Δ)^s is non-local, and the *spatial* discretization (e.g., as a graph Laplacian on a finite domain) requires care. The CHEAP primitive computes the spectral propagator under the assumption that the spatial operator is diagonal in the DCT basis; for finite-volume or finite-element discretizations of fractional Laplacians on irregular domains, the spectral diagonalization is approximate.

**Library status.** Compose: `w[k] = exp(-t * pow(mu[k], s))`.

#### 4.2.4 Ornstein–Uhlenbeck Propagator

The Ornstein–Uhlenbeck (OU) process is the unique stationary, mean-reverting Gaussian Markov process. As an SPDE, it is dX_t = −θ X_t dt + σ dW_t, with stationary variance σ²/(2θ). On a spatial grid with spatially-correlated noise (replacing dW_t with colored noise of spectrum w_noise(μ_k)), the time-stepping operator combines exponential mean-reversion with the heat-style update of the noise covariance:

$$X_{t + \Delta t}[k] = e^{-\theta \Delta t} X_t[k] + \sigma \sqrt{\frac{1 - e^{-2\theta \Delta t}}{2\theta}} \cdot w_{\text{noise}}^{1/2}(\mu_k) \cdot z[k],$$

where z is white noise. The mean-reversion factor e^{−θΔt} is a Period 2 primitive (constant in μ_k); the noise scaling factor is the *exact* stationary variance of the OU process truncated to the time interval Δt. Both pieces are O(n) per step.

**Edge of validity.** The OU spectral form above assumes the noise spectrum w_noise is constant in time. For time-varying noise spectra, the integral becomes a Volterra integral, and the closed form breaks. For spatially homogeneous OU (no spatial correlation in noise), the per-frequency OU collapses to a sequence of independent scalar OU processes, one per DCT mode.

**Connection to Matérn-1/2.** The stationary covariance of a 1D OU process is exponential: Cov(X_s, X_t) = (σ²/2θ) exp(−θ|t−s|). Sampling the OU process at fixed time and varying space (with spatially correlated noise) recovers a Matérn-1/2 field; this is the *exponential* limit of the Matérn family.

**Library status.** Planned. The mean-reversion and noise-scaling factors are easy to compose; the integrated formula is provided as a convenience.

#### 4.2.5 Cahn–Hilliard (Linearized)

The Cahn–Hilliard equation ∂_t u = Δ(εΔu − u/ε + W'(u)) describes phase separation with conserved order parameter. Linearizing around u = 0 (the spinodal) and dropping the nonlinear potential W'(u) gives ∂_t u = εΔ²u − Δu/ε, with spectral symbol

$$\sigma(\mu_k) = \varepsilon \mu_k^2 - \mu_k / \varepsilon.$$

The implicit Euler step solves (I − tσ)u^{n+1} = u^n in the spectral domain, giving

$$w_k = \frac{1}{1 + t (\varepsilon \mu_k^2 + \mu_k / \varepsilon)}.$$

**DC handling.** w₀ = 1, identical to other parabolic implicit-Euler primitives.

**Stability.** Both terms in the denominator are non-negative (note: we have absorbed the destabilizing −μ_k/ε of the unstable mode into the stable form by considering ∂_t u = −(εΔ² − Δ/ε)u — the *gradient flow of the Ginzburg–Landau energy*, which is the form actually used as a preconditioner). For the unstable linearization with a sign flip, weights would diverge: see §4.7 for the analogous backward-heat hazard.

**Edge of validity (regime of validity).** The linearization is exact only for u² ≪ 1 (small perturbations from the spinodal). The full Cahn–Hilliard equation is nonlinear, and the linearized spectral propagator should be used as a *preconditioner* inside a Newton or fixed-point iteration on the nonlinear residual, not as a standalone solver. Treating the linearized propagator as the full physics will mis-predict coarsening exponents and stationary morphologies.

**Library status.** Planned. Useful as a preconditioner inside iterative solvers for phase-field PDEs.

---

### 4.3 Period 3 — Elliptic / Poisson-type Inverses

Period 3 collects spectral multipliers that *invert* a positive-semidefinite elliptic operator, transforming a right-hand side into a solution of a boundary value problem. Where Period 2 weights are bounded in (0, 1], Period 3 weights are *unbounded* near the null space of the operator (typically DC), requiring explicit regularization or projection. Every primitive in this period satisfies w_k → ∞ as μ_k → 0⁺, so DC handling is the central design question.

#### 4.3.1 Poisson Inverse

The Poisson equation −Δu = f on a bounded domain with Neumann boundary conditions has spectral inverse

$$w_0 = 0, \quad w_k = \frac{1}{\mu_k + \varepsilon}, \quad k \geq 1.$$

**Mathematical background.** The Laplacian under Neumann BC has a one-dimensional null space (the constants), so −Δu = f admits a solution if and only if f has zero mean (the Fredholm alternative). When this solvability condition holds, the solution is unique up to an additive constant, conventionally fixed by requiring zero mean of u. The CHEAP primitive enforces this convention by hardcoding w₀ = 0.

**Derivation.** In the DCT-II basis, the spatial Laplacian acts as multiplication by −μ_k: −μ_k û_k = f̂_k. For k ≥ 1 we solve û_k = −f̂_k/μ_k, with the sign of Δ absorbed into the convention. For k = 0, the equation reads 0·û_0 = f̂_0, which is solvable only if f̂_0 = 0 (the zero-mean condition); we project f̂_0 ↦ 0 and set û_0 = 0.

**Regularization.** The ε > 0 floor in the denominator turns the Poisson inverse into a *screened Poisson* (Yukawa, §4.3.4): instead of inverting −Δ, we invert (−Δ + εI). The DC weight becomes 1/(0 + ε) = 1/ε if we drop the hardcoded w₀ = 0, but we keep the projection to preserve the zero-mean convention exactly.

**Green's function.** The continuum Green's function of −Δ on ℝ^d is G(x) = log‖x‖/(2π) in 2D and −1/(4π‖x‖) in 3D. The discrete spectral inverse 1/μ_k is the discrete analogue, with ε playing the role of a screening constant.

**Library status.** Shipped (`cheap_weights_poisson_inverse_ev`).

#### 4.3.2 Biharmonic Inverse

The biharmonic operator Δ² has spectrum μ_k² in the DCT-II basis. Its spectral inverse is

$$w_0 = 0, \quad w_k = \frac{1}{\mu_k^2 + \varepsilon}, \quad k \geq 1.$$

**Derivation.** The thin-plate energy E[u] = ∫ ‖Δu‖² minimized subject to interpolation constraints is solved by the Green's function of Δ², known as the thin-plate spline. In the spectral domain the solution operator is diag(1/μ_k²). The same operator appears in the Euler–Bernoulli beam equation EI · ∂⁴u/∂x⁴ = f and in the 2D Stokes stream-function equation Δ²ψ = ω.

**DC handling.** The unregularized weight 1/μ₀² = ∞ corresponds to the fact that Δ²u = f has a *two-dimensional* null space spanned by {1, x} in 1D — both constants and affine functions are biharmonic. Adding ε > 0 gives DC weight 1/ε; users requiring exact null-space enforcement should zero w[0] (and w[1] if affine functions are in the null space) after the call.

**Monotonicity.** Weights are strictly decreasing in k for k ≥ 1, consistent with the fourth-order smoothness penalty.

**SIMD.** The hot path involves only multiplication, addition, and reciprocal — no transcendentals — and is vectorized (AVX2: 4×f64, NEON: 2×f64).

**Honest limitations.** Biharmonic regularization imposes fourth-order smoothness, which over-smooths piecewise-smooth signals (e.g., images with sharp edges). For such signals, a first-order Tikhonov or Poisson-style penalty is preferable.

**Library status.** Shipped (`cheap_weights_biharmonic_ev`, `_2d`, `_3d`).

#### 4.3.3 Higher-Order Tikhonov Deconvolution

When deconvolving an observation y = Hx + n with a known PSF H (spectrum ψ_k) under a Gaussian signal prior with precision (−Δ)^{p/2}, the MAP estimate is

$$w_k = \frac{\psi_k}{\psi_k^2 + \alpha \mu_k^p + \varepsilon}.$$

**Derivation.** The MAP objective ‖y − Hx‖² + α ‖(−Δ)^{p/2} x‖² becomes ∑_k [(ŷ_k − ψ_k x̂_k)² + α μ_k^p x̂_k²] in the spectral basis. Setting the derivative to zero gives x̂_k = ψ_k ŷ_k / (ψ_k² + α μ_k^p), matching the stated formula.

**Special cases.**
- p = 0, α = η: reduces to Wiener deconvolution (§4.5.2) with flat noise floor η.
- p = 1: gradient-norm penalty (Sobolev H^{1/2} prior); weights taper gently at high k.
- p = 2: biharmonic penalty (Sobolev H¹ prior); weights suppress high frequencies quadratically faster than p = 1.
- p > 2: super-Sobolev priors, appropriate for very smooth signals.

**DC analysis.** At k = 0, μ₀^p = 0 (for p > 0), so w₀ = ψ₀/(ψ₀² + ε), inverting the PSF at DC with no roughness penalty — physically correct, since DC has zero spatial frequency. If ψ₀ = 0 and ε = 0, the denominator is floored at CHEAP_EPS_DIV (1e-300).

**Zero-allocation NULL path.** When `lap_eigenvalues == NULL`, the 1D Laplacian value 4 sin²(πk/2n) is computed on-the-fly, eliminating the auxiliary allocation. For 2D/3D problems pass the flat Laplacian grid explicitly.

**Honest limitations.** The Sobolev prior is isotropic; for images with directional textures or anisotropic PSFs, an anisotropic regularization (e.g., from `cheap_weights_anisotropic_matern_2d`) may produce better results.

**Library status.** Shipped (`cheap_weights_hot_deconv_ev`).

#### 4.3.4 Helmholtz / Yukawa Inverse

The Helmholtz equation (κ² − Δ) u = f (with κ² > 0) is the *screened Poisson* equation, equivalent to the Yukawa potential in physics. Its spectral inverse is

$$w_k = \frac{1}{\kappa^2 + \mu_k}.$$

**DC handling.** w₀ = 1/κ², bounded for any κ > 0. The screening parameter κ regularizes the DC singularity present in the unscreened Poisson inverse (§4.3.1) without requiring a hardcoded projection.

**Equivalence to Whittle.** Identical formula to the Whittle covariance kernel (§4.1.5). The Period 1 (covariance) and Period 3 (PDE inverse) interpretations are mathematically dual: the inverse of a positive-definite elliptic operator is, by definition, the covariance of the Gaussian process with that precision operator.

**Stability.** Weights bounded in (0, 1/κ²]. No regularization beyond κ > 0.

**Honest limitations.** True Helmholtz scattering problems involve *negative* κ² (the wave equation in the frequency domain), giving an oscillatory Green's function with phase information. The real positive-κ² primitive here solves the screened (decaying) Helmholtz; the true scattering Helmholtz requires complex weights and is in the Hazard Log (§4.7.2).

**Library status.** Planned. Composable: `w[k] = 1.0 / (kappa*kappa + mu[k])`.

#### 4.3.5 Fractional Integral / Riesz Potential

The Riesz potential I_s = (−Δ)^{−s/2} is the inverse of the fractional Laplacian (§4.6.2). Its spectral weight is

$$w_0 = 0, \quad w_k = \mu_k^{-s/2}, \quad k \geq 1.$$

**Derivation.** The continuum Riesz potential is convolution with the kernel c_{d,s} ‖x‖^{s−d}, defined for 0 < s < d. In the Fourier domain it acts as ‖ξ‖^{−s}, which on the DCT lattice becomes μ_k^{−s/2} (with the factor of 2 absorbed into the exponent). For s ∈ (0, d), the kernel is locally integrable and the potential is well-defined; outside this range, regularization (the +ε floor) is needed.

**DC handling.** The unregularized weight diverges at DC; we hardcode w₀ = 0 to enforce the zero-mean projection (analogous to Poisson inverse).

**Connection to long-memory processes.** The Riesz potential is the spatial analogue of the Riemann–Liouville fractional integral; applied to white noise, it produces fractional Brownian motion in the spatial domain. This makes it a primitive both for *deterministic* PDE inversion and for *stochastic* GP construction.

**Honest limitations.** For s ≥ d, the Riesz potential is no longer locally integrable, and the discrete spectral weight diverges faster than μ_k^{−1}. The ε floor caps the divergence at 1/ε^{s/2}, but the resulting filter loses the conformal-invariance properties of the true Riesz potential.

**Library status.** Planned.

#### 4.3.6 Tikhonov H^p Prior (no PSF)

When regularizing an inverse problem *without* a forward operator (e.g., denoising or smoothing), the Tikhonov estimator under an H^p Sobolev prior reduces to

$$w_k = \frac{1}{1 + \alpha \mu_k^p}.$$

**Derivation.** The objective ‖y − x‖² + α ‖(−Δ)^{p/2} x‖² has spectral form ∑_k [(ŷ_k − x̂_k)² + α μ_k^p x̂_k²]. Setting the derivative to zero yields x̂_k = ŷ_k / (1 + α μ_k^p).

**Special cases.**
- p = 0: w_k = 1/(1 + α), a uniform shrinkage by 1/(1+α). Trivial.
- p = 1: gradient penalty, equivalent to the Bessel potential (1 + αμ_k)^{−1} — a single-step implicit Euler heat operator with t = α.
- p = 2: biharmonic penalty; weights w_k = 1/(1 + αμ_k²) suppress high frequencies twice as fast as p = 1.

**Relationship to HOT deconvolution.** Setting ψ_k = 1 in the HOT formula (§4.3.3) recovers Tikhonov H^p exactly. We list it separately because the no-PSF case is structurally simpler, has different numerical properties (no PSF zero handling), and admits closed-form parameter selection via Morozov's discrepancy principle or Stein's unbiased risk estimator.

**DC handling.** w₀ = 1 exactly — the prior penalizes departures from smoothness, not departures from zero mean.

**Library status.** Planned. Composable from HOT with ψ_k ≡ 1.

---

### 4.4 Period 4 — Hyperbolic / Wave Propagators

Period 4 collects spectral multipliers that arise as time-evolution operators of *hyperbolic* PDEs. Unlike Period 2 (parabolic), where weights are bounded and dissipative, Period 4 weights are *oscillatory* in time, with magnitudes bounded by 1 but signs that change. The defining structural feature is the *unitarity* of the propagator: the L² norm of the solution is conserved exactly under exact time evolution.

Hyperbolic propagators are a CHEAP-feasible class because the cosine and sine of √μ remain real-valued; we do not need to leave the real-double regime, in contrast to the Schrödinger propagator (§4.7.2), whose phase factor exp(−itμ) is fundamentally complex.

#### 4.4.1 Wave Propagator

The wave equation ∂²_t u = c² Δu with initial conditions (u₀, v₀) admits the d'Alembert formula in spectral form:

$$\hat{u}(t)[k] = \cos(c t \sqrt{\mu_k}) \cdot \hat{u}_0[k] + \frac{\sin(c t \sqrt{\mu_k})}{c \sqrt{\mu_k}} \cdot \hat{v}_0[k].$$

The CHEAP wave propagator splits this into two primitives:

$$w_k^{\text{cos}} = \cos(c t \sqrt{\mu_k}), \qquad w_k^{\text{sinc}} = \frac{\sin(c t \sqrt{\mu_k})}{c \sqrt{\mu_k}}.$$

The cos-weight propagates the position; the sinc-weight propagates the velocity. Both are real, smooth functions of μ_k, and both are bounded by 1 (sinc → t at μ_k → 0). One full time step requires two `cheap_apply` calls — one for each weight — plus an addition.

**DC handling.** At μ₀ = 0: w_cos(0) = 1, w_sinc(0) = t. Both are finite; the velocity weight grows linearly with t at DC, reflecting the unbounded translation of the mean displacement under the wave equation.

**Stability.** The exact spectral propagator is unconditionally stable: |w_cos| ≤ 1 and |w_sinc| ≤ t for all t. Unlike finite-difference time-domain (FDTD) wave solvers, there is no Courant–Friedrichs–Lewy (CFL) condition restricting t in terms of the grid spacing — *because* the spatial discretization is exact in the DCT basis. This is one of the clearest expressions of the CHEAP advantage in this period.

**Composition.** The wave propagator does *not* admit a semigroup property (the wave equation is second-order in time), but it admits a *symplectic* composition: applying w(t₁) followed by w(t₂) is equivalent to applying w(t₁+t₂) to a four-component (position, velocity, position-from-t₁, velocity-from-t₁) state vector. In practice, users typically just apply the full t = T propagator in one call.

**Honest limitations.** The cos/sin construction discards the *phase* relationship between modes that travel in opposite directions; for problems requiring directional decomposition (e.g., one-way wave propagation in seismic imaging), a more elaborate splitting is needed. The CHEAP primitive computes the full undirected wave field.

**Library status.** Planned. The two weights are easy to compute; the only design question is whether to expose them separately or as a paired interface.

#### 4.4.2 Damped Wave / Klein–Gordon

The damped wave / Klein–Gordon equation ∂²_t u + 2γ ∂_t u + m² u = c² Δu generalizes the wave equation with both *dissipation* (γ) and *mass* (m). Its spectral propagator splits into a real part (damping) and an oscillatory part (modified frequency):

$$w_k = e^{-\gamma t} \cos(t \sqrt{c^2 \mu_k + m^2 - \gamma^2}).$$

This is the *underdamped* form, valid when c²μ_k + m² > γ². For overdamped modes (c²μ_k + m² < γ²), the cosine becomes a hyperbolic cosine of an imaginary argument — equivalently, a sum of two real exponentials with rates γ ± √(γ² − c²μ_k − m²). The CHEAP primitive selects the appropriate branch based on the sign of the discriminant.

**DC handling.** At μ₀ = 0: w₀ = exp(−γt) cos(t √(m² − γ²)), which is bounded for all real γ, m. For m > γ (underdamped DC), the DC mode oscillates and decays; for m < γ (overdamped DC), it decays without oscillation; for m = 0 = γ (pure wave), it equals 1 (no DC change, consistent with §4.4.1 in the limit).

**Stability.** Weights bounded by exp(−γt) for all k, so γ > 0 ensures unconditional dissipative stability. γ = 0 recovers the unitary Klein–Gordon propagator (m > 0) or the wave propagator (m = 0).

**Honest limitations.** Like the undamped wave, the propagator is undirected. For seismic visco-acoustics with explicit absorbing boundary conditions, more elaborate constructions (perfectly matched layers, complex frequency shifts) are needed.

**Library status.** Planned.

---

### 4.5 Period 5 — Regularization, Denoising, and RMT Shrinkage

Period 5 collects spectral multipliers that arise from *statistical* — rather than physical — considerations. The defining feature is that the weight is constructed from estimated quantities (signal-to-noise ratios, sample eigenvalues, target dimensions) rather than from an analytical PDE or covariance model. Period 5 primitives are typically applied to *user-provided eigenvalues* via `_ev` constructors, since the eigenvalues to be processed come from the data, not from a closed-form formula.

#### 4.5.1 Wiener Filter

The classical Wiener filter provides the minimum mean-square-error linear estimate of a signal observed in additive white noise. When the signal covariance is diagonal in the DCT basis with eigenvalues {μ_k}:

$$w_k = \frac{\mu_k}{\mu_k + \sigma^2}.$$

**Derivation.** Under signal/noise model y_k = x_k + n_k with x_k ∼ 𝒩(0, μ_k) and n_k ∼ 𝒩(0, σ²) independent, the Bayes posterior mean is x̂_k = (μ_k/(μ_k + σ²)) y_k. This is the linear MMSE estimator, exact when the signal is Gaussian.

**DC handling.** At μ₀ = 0 (Laplacian basis), w₀ = 0, which zeroes the DC component when DC signal power is modeled as zero. For problems where DC carries signal energy, supply a custom μ₀ > 0 via `_ev`.

**Stability.** Weights bounded in [0, 1), monotonically non-decreasing in μ_k. The filter never amplifies any spectral component — a stronger guarantee than is available for general (non-spectral) Wiener filters.

**Honest limitations.** The formula implements scalar diagonal loading, *not* the full Minimum Variance Distortionless Response (MVDR) beamformer, which requires a steering vector and explicit matrix inversion. The naming follows established signal-processing convention for the scalar spectral case.

**Library status.** Shipped (`cheap_weights_wiener_ev`).

#### 4.5.2 Wiener Deconvolution

When the observation is the convolution of a latent signal with a known PSF h corrupted by additive white noise, y = h ∗ x + ε with ε ∼ 𝒩(0, ηI), the MMSE deconvolution filter in the spectral domain is

$$w_k = \frac{\psi_k}{\psi_k^2 + \eta},$$

where ψ_k are the DCT-II eigenvalues of the symmetric circulant extension of the PSF.

**Derivation.** The optimal linear estimate minimizes 𝔼‖x − Wy‖². In the spectral basis with H = diag(ψ_k), the optimal weight matrix is W = (H^*H + ηI)^{−1} H^*, which is diagonal with entries ψ_k*/(ψ_k² + η) = ψ_k/(ψ_k² + η) for real ψ_k (PSF is symmetric real).

**Distinction from §4.5.1.** The denominator here is *quadratic* in ψ_k, not linear. The quadratic form arises because the PSF eigenvalue appears both in the forward model (numerator from H^*) and in its conjugate-transpose product (denominator from H^*H), whereas the denoising Wiener filter has a scalar signal power in the numerator with no corresponding PSF.

**Stability bound.** The maximum filter gain is |w_k| ≤ 1/(2√η), achieved at ψ_k = √η. This is a finite, η-controlled amplification, in contrast to the unregularized inverse 1/ψ_k which diverges at PSF zeros.

**Obtaining PSF eigenvalues.** Use `cheap_toeplitz_eigenvalues` to compute the DCT-II of the first column of the circulant PSF extension. For 2D/3D, pass the flat row-major PSF eigenvalue grid directly.

**Honest limitations.** The circulant PSF assumption introduces wrap-around artifacts at non-periodic boundaries. Standard mitigations (boundary padding, windowing, half-space Toeplitz models) are outside the primitive's scope.

**Library status.** Shipped (`cheap_weights_wiener_deconv_ev`).

#### 4.5.3 Spectral Normalization

Covariance whitening — transforming data so its covariance becomes the identity — requires the inverse square root of the covariance eigenvalues:

$$w_k = \frac{1}{\sqrt{\mu_k + \varepsilon}}.$$

The ε > 0 floor prevents the singularity at μ₀ = 0 and bounds the maximum amplification to ε^{−1/2}.

**Application: Lipschitz neural networks.** Spectral normalization of weight matrices [12] enforces Lipschitz continuity by bounding the spectral norm. For convolutional layers, the DCT provides approximate diagonalization, enabling O(n log n) normalization without explicit SVD.

**Bias-variance tradeoff in ε.** Too small: numerical noise in low-frequency components is amplified. Too large: whitening effect is attenuated. Adaptive selection ε = max(ε_abs, ε_rel · μ_max) provides a reasonable default; we make no claim of optimality.

**Library status.** Shipped (`cheap_weights_specnorm_ev`).

#### 4.5.4 Kernel PCA Projection (Hard and Soft)

Spectral truncation is the simplest dimensionality reduction in the spectral domain. The hard threshold

$$w_k = \mathbb{1}(k < K)$$

retains the first K spectral components and discards the rest. This is a *projection*: applying it twice yields the same result as applying it once, a useful correctness check.

The soft variant uses Flandrin eigenvalue ratios:

$$w_k = \max\!\left(0, 1 - \frac{\lambda_K}{\lambda_k}\right).$$

Since Flandrin eigenvalues decrease with k, this assigns weight ≈ 1 to components with λ_k ≫ λ_K and weight ≈ 0 near the cutoff — consistent with PCA's variance-maximizing interpretation.

**Choice of K.** Automated selection via explained-variance thresholds, scree plots, or Stein's unbiased risk estimator (SURE) is straightforward to implement on top of the precomputed eigenvalues but lies outside the core primitive.

**Library status.** Shipped (`cheap_weights_kpca_hard`, `cheap_weights_kpca_soft`).

#### 4.5.5 Random Matrix Denoising (Hard and Optimal)

Sample covariance eigenvalues are systematically biased: small population eigenvalues are pushed down and large ones are pushed up. The Marchenko–Pastur law [15] characterizes the bulk spectrum under identity-covariance noise with variance σ² and aspect ratio c = n/p, giving bulk edges

$$\lambda_{\pm} = \sigma^2 (1 \pm \sqrt{c})^2.$$

**Hard thresholding.** w_k = λ_k · 𝟙(λ_k > λ_+) zeroes bulk eigenvalues and preserves signal eigenvalues unchanged. Aggressive but unbiased above threshold.

**Optimal shrinkage (Donoho–Gavish–Johnstone [16]).** The Frobenius-optimal nonlinear shrinkage above the bulk edge is

$$\tilde{\lambda}_k = \lambda_k \cdot \frac{\sqrt{(\ell - \ell_+)(\ell - \ell_-)}}{\ell}, \quad \ell = \lambda_k / \sigma^2,$$

with ℓ_± = (1 ± √c)². This reduces upward bias while preserving more signal than hard thresholding. Closed-form, O(1) per eigenvalue.

**Honest limitations.** Marchenko–Pastur is asymptotic (n, p → ∞ with c fixed). For n < 100, finite-size corrections matter. The optimal shrinkage assumes spiked covariance structure; for gradually decaying population spectra, nonparametric methods [17] may be more appropriate (typically O(n²)).

**Library status.** Shipped (`cheap_weights_rmt_hard`, `cheap_weights_rmt_shrink`).

#### 4.5.6 RMT BBP Spiked Estimator

The Baik–Ben Arous–Péché (BBP) phase transition gives a sharper spike-detection criterion than the Marchenko–Pastur edge for *spiked* covariance models. Under the BBP criterion, an eigenvalue λ_k is identified as a signal if and only if it exceeds a critical threshold related to the population spike:

$$w_k = \begin{cases} (\lambda_k^2 - c \sigma^4)/\lambda_k & \text{if } \lambda_k > \lambda_+, \\ 0 & \text{otherwise.} \end{cases}$$

**Derivation.** Above the BBP threshold, the population spike θ relates to the sample spike λ via θ = λ + cσ²λ/(λ − σ²). Solving for the corrected sample-to-population estimate and substituting back gives the bias-corrected estimator above.

**Comparison.** BBP shrinkage is structurally similar to DGJ optimal shrinkage but uses a different bias correction valid in the *spike-detection* regime (where one cares about identifying signal eigenvalues at all, not just shrinking them optimally).

**Library status.** Planned. Useful for principal component selection in high-dimensional PCA.

#### 4.5.7 Geman–McClure Robust Shrinkage

The Geman–McClure ρ-function ρ(t) = t²/(1 + t²) is a robust loss with bounded influence. As a spectral shrinkage operator, its derivative gives the multiplier

$$w_k = \frac{\lambda_k^2}{\lambda_k^2 + \sigma^2}.$$

**Derivation.** Maximum-likelihood estimation under a Cauchy noise model (heavy tails) yields the Geman–McClure influence function ψ(t) = 2t/(1 + t²)². Treating the spectral coefficients as observations contaminated with heavy-tailed noise, the corresponding shrinkage is the squared-magnitude variant of the linear Wiener filter.

**Comparison to Wiener.** The standard Wiener filter w = λ/(λ + σ²) is *linear in λ*; Geman–McClure w = λ²/(λ² + σ²) is *quadratic*. The quadratic form has steeper transition between "noise" and "signal" regimes: small λ are suppressed more aggressively, while large λ are shrunk less. This is desirable when the noise is impulsive or heavy-tailed rather than Gaussian.

**DC handling.** w₀ = 0 (when λ₀ = 0).

**Stability.** Bounded in [0, 1), monotone in λ_k.

**Library status.** Planned. Useful as a robust alternative to standard Wiener for non-Gaussian noise.

#### 4.5.8 Cesàro / Fejér Window

The Cesàro mean of a Fourier series is the partial sum convolved with the Fejér kernel. As a spectral multiplier, this is the triangular window

$$w_k = \max\!\left(0, 1 - k/K\right).$$

Where the brick-wall projection (§4.5.4) introduces Gibbs ringing, the Fejér window damps it: the partial sum's Fourier coefficients are tapered linearly to zero at k = K, which produces a non-negative spatial-domain kernel and eliminates Gibbs oscillations.

**Stability.** Weights in [0, 1], piecewise linear in k. The implied spatial kernel is the Fejér kernel, which is non-negative and integrates to 1 — a valid probability density.

**Comparison to brick-wall.** Both are index-based, both truncate at k = K. Fejér is the *anti-Gibbs* counterpart, trading sharper frequency cutoff for cleaner spatial response.

**Library status.** Compose: `w[k] = (k < K) ? (1.0 - (double)k / K) : 0.0`.

#### 4.5.9 Lanczos σ-Factor

The Lanczos σ-factor is the windowed-sinc multiplier used to suppress Gibbs ringing in truncated Fourier series:

$$w_k = \frac{\sin(\pi k / N)}{\pi k / N} = \text{sinc}(\pi k / N).$$

In contrast to Cesàro, which damps linearly to zero, Lanczos damps via a sinc envelope, which has zeros (oscillations) at k = N, 2N, … For typical use (N >> K with K the truncation index), Lanczos preserves more of the low-frequency content than Cesàro at the cost of a slightly worse spatial-domain envelope.

**DC handling.** w₀ = lim_{k→0} sinc(πk/N) = 1.

**Stability.** Bounded by 1 in magnitude, with sign changes at k = N, 2N, …

**Honest limitations.** The sign changes mean Lanczos is not a positive-definite multiplier; the implied spatial kernel takes negative values. For applications requiring a probability-density-like kernel, prefer Fejér.

**Library status.** Compose: `w[k] = (k == 0) ? 1.0 : sin(M_PI*k/N) / (M_PI*k/N)`.

---

### 4.6 Period 6 — Fractional, Multifractal, and Advanced Spectral Operators

Period 6 collects spectral multipliers whose constructions involve *transcendental* functions of the spectrum (powers, gamma functions, transcendental decay laws) rather than simple algebraic combinations. The defining feature is that the weight family captures *long-range correlations* or *self-similar* (scale-invariant) statistics that simpler algebraic weights cannot represent.

#### 4.6.1 Fractional Differentiation

The fractional derivative of order d (real, positive or negative) on a uniform grid acts in the DCT basis as

$$w_k = (2 \sin(\omega_k / 2))^d, \quad \omega_k = \pi k / n.$$

For d = 1, this recovers the central-difference derivative; for d = −1, the cumulative integral; for fractional d, the Grünwald–Letnikov fractional difference operator. The factor 2 sin(ω/2) is the DCT-II symbol of the discrete first-difference operator.

**DC handling.** At ω₀ = 0, w₀ = 0 for d > 0 (derivative of constant is zero) and w₀ = ∞ for d < 0 (integral of constant diverges). For d < 0, the implementation floors the divergence at CHEAP_EPS_LOG.

**Library status.** Shipped (`cheap_weights_fractional`).

#### 4.6.2 Fractional Laplacian

The fractional Laplacian (−Δ)^s is the Fourier multiplier ‖ξ‖^{2s}. On the DCT lattice:

$$w_k = \mu_k^s.$$

For s = 1, this recovers the discrete Laplacian; for s = 2, the biharmonic operator; for s = 1/2, the half-Laplacian, used in Lévy-flight models and edge-enhancement image processing.

**DC handling.** μ₀^s = 0 for s > 0, w₀ = 0 exactly. For s < 0, see Riesz potential (§4.3.5).

**Library status.** Shipped (composable from `cheap_weights_laplacian` + `pow`).

#### 4.6.3 Sobolev Synthesizer

The Sobolev synthesizer (1 + μ_k)^{s/2} is the *positive*-exponent counterpart of the Bessel potential (§4.1.6):

$$w_k = (1 + \mu_k)^{s/2}.$$

For s > 0, this *amplifies* high frequencies relative to low — the spectral form of an analyzer that promotes roughness, used in residual analysis and high-pass filtering.

**DC handling.** w₀ = 1 exactly. The "+1" prevents a zero at DC, distinguishing this from the pure fractional Laplacian.

**Synthesis-analysis duality.** Bessel potential (§4.1.6) and Sobolev synthesizer satisfy w_Bessel · w_synth = 1 when their s parameters are negatives. Composing the two recovers the identity, providing a useful sanity check.

**Library status.** Planned. Composable: `w[k] = pow(1.0 + mu[k], s / 2.0)`.

#### 4.6.4 Kolmogorov 5/3 Turbulence

Kolmogorov's 1941 turbulence theory predicts that the energy spectrum of inertial-range turbulence follows the −5/3 power law. As a spectral weight:

$$w_k = \mu_k^{-5/6}.$$

The exponent is −5/6 (not −5/3) because the weight acts on the *amplitude* (square root of power); applying it to white noise produces a velocity field whose energy spectrum E(k) ∝ k^{−5/3}.

**DC handling.** Diverges at μ₀; floor with ε.

**Special case of Riesz potential.** Kolmogorov 5/3 is the Riesz potential (§4.3.5) at s = 5/3. We list it separately because it is a *named* primitive in turbulence literature with specific physical content (energy cascade in 3D incompressible Navier–Stokes), not just an arbitrary fractional integral.

**Library status.** Planned. Composable.

#### 4.6.5 Mandelbrot Multifractal Weights

Mandelbrot's extension of fractional Brownian motion to multifractal processes involves the ratio of Gamma functions with complex arguments [13]:

$$w_k = \left|\frac{\Gamma(H + i \tau_k)}{\Gamma(1 - H + i \tau_k)}\right|, \quad \tau_k = \frac{\pi k}{n}.$$

This generalizes the standard fractional weights (2 sin(ω_k/2))^d to a family parametrized by complex Gamma ratios, enabling representation of multiscaling phenomena that simple power-law spectra cannot capture.

**Computation.** The complex Gamma function is evaluated via the Lanczos approximation [14] with parameters g = 7, N = 9, achieving machine precision for Re(z) ≥ 1/2. For Re(z) < 1/2, the reflection formula

$$\ln \Gamma(z) = \ln \pi - \ln \sin(\pi z) - \ln \Gamma(1 - z)$$

reduces to the convergent half-plane. Computation is performed in log-space — Re(ln Γ(H + iτ_k)) − Re(ln Γ(1−H + iτ_k)), then exponentiate — which avoids overflow for large |τ_k|.

**Symmetry.** At H = 1/2, numerator and denominator coincide: all weights equal 1 to machine precision. We verify this in the test suite.

**Domain.** For H ∈ (0, 1), neither H + iτ nor 1 − H + iτ passes through a Gamma pole, so weights are finite and positive throughout. Boundary values H → 0⁺ or H → 1⁻ remain well-defined but exhibit large dynamic range.

**Honest limitations.** This implementation computes the *magnitude* of the Gamma ratio, discarding phase. Causal multifractal processes require full complex weights and a complex DFT, at roughly 2× the cost.

**Library status.** Shipped (`cheap_weights_mandelbrot`).

---

### 4.7 The Hazard Log: Rejected Primitives and Their Failure Modes

Several spectral multipliers admit O(n log n) construction in principle but are *rejected* from the library — meaning either no `_ev` constructor is provided, or one is provided with a prominent warning. We document each rejection with the underlying mathematical reason, because a complete periodic table must explain not only what is admitted but what is excluded and why.

The CHEAP doctrine that defines admissibility consists of four rules, in order of precedence: (1) **Real-only**: weights must be real-valued doubles, no complex arithmetic. (2) **Stability**: weights must not amplify floating-point noise without bound. (3) **SIMD-friendly**: weights should be computable in branchless or near-branchless inner loops to support AVX2/NEON vectorization. (4) **One-array**: weights should be expressible as a single elementwise scalar multiplier per DCT bin, not as a vector-valued or matrix-valued operation.

#### 4.7.1 Backward Heat Equation

The backward heat equation ∂_t u = −Δu is the time-reversal of the heat equation. Its spectral propagator is

$$w_k = \exp(+t \mu_k).$$

**Why it fails.** The weights diverge exponentially in μ_k. At the Nyquist mode μ_max = 4, even modest t produces astronomical amplification: e^{4·1} ≈ 55, e^{4·5} ≈ 5×10⁸, e^{4·10} ≈ 2×10¹⁷. Floating-point noise present at the Nyquist mode is amplified to dominate the signal. This is not merely a numerical issue: the backward heat equation is *mathematically ill-posed* (Hadamard) — solutions exist only for analytic initial data, and small perturbations of the data produce arbitrarily large perturbations of the solution.

**What to use instead.** For "deblurring" applications, use Wiener deconvolution (§4.5.2) or Higher-Order Tikhonov (§4.3.3), which solve the *inverse problem* y = Hx + n with regularization, not the ill-posed time-reversal.

**Library status.** Rejected. Will not be shipped.

#### 4.7.2 Real Schrödinger / Complex Wave Propagators

The Schrödinger equation iℏ ∂_t ψ = (ℏ²/2m)(−Δ)ψ + Vψ has spectral propagator (in the V = 0 case)

$$w_k = \exp(-i t \mu_k / (2m)),$$

a unit-modulus complex phase factor. The CHEAP cosine propagator (§4.4.1) extracts the real part for the wave equation but cannot do so for Schrödinger because the wave equation is second-order in time — the cos and sin pieces correspond to position and momentum initial data — whereas Schrödinger is first-order in time and the cos and sin pieces are the *real and imaginary parts of the wave function*, both of which must propagate together.

**Why it fails.** Storing complex wave functions doubles buffer sizes, breaks the real-valued DCT contract (we would need a complex DFT or a paired DCT/DST construction), and prevents the SIMD primitives from operating on contiguous real-double arrays.

**What to use instead.** The wave propagator (§4.4.1) for hyperbolic problems; for genuinely quantum-mechanical problems, a separate complex-valued library is required.

**Library status.** Rejected. Outside scope.

#### 4.7.3 Turing Growth (Reaction–Diffusion Linearization)

The linearization of a Turing reaction–diffusion system around a homogeneous steady state has growth-rate symbol

$$\sigma(\mu_k) = a - b \mu_k - c / \mu_k$$

(for some signed parameters a, b, c). The spectral propagator over time t is exp(t σ(μ_k)).

**Why it fails (in part).** The c/μ_k term is *singular at DC*, and for c > 0, the propagator at μ_k → 0 grows without bound — a backward-heat-style instability concentrated at the lowest frequencies. For c < 0 the singularity is integrable but the propagator still grows exponentially in t at any fixed μ_k where σ > 0, requiring stability analysis at every parameter setting.

**What to use instead.** Compose manually using the heat propagator (§4.2.1) for the −bμ_k term and an explicit-Euler step for the reaction (a, c/μ_k) terms. Treating the spectrum as a black-box weight obscures the bifurcation structure that Turing analysis is meant to reveal.

**Library status.** Compose only. We document the formula but do not ship a constructor.

#### 4.7.4 Riesz Transform (Vector-Valued)

The Riesz transform is the d-dimensional generalization of the Hilbert transform, with d separate components R_1, …, R_d, each acting in the spectral domain as

$$\widehat{R_j u}(\xi) = -i \frac{\xi_j}{\|\xi\|} \hat{u}(\xi).$$

**Why it fails.** The Riesz transform is *not a scalar multiplier*: each component requires the per-axis spectral coordinate ξ_j divided by the magnitude ‖ξ‖, which on a 2D DCT lattice means computing k_x and k_y separately for each pixel and dividing by √(μ_x[k_x] + μ_y[k_y]). This requires per-axis index awareness and introduces a vector output per scalar input, breaking the one-array rule.

**What to use instead.** Implement directly with `cheap_weights_index_ev`-style constructors that take per-axis coordinates and produce per-axis output arrays. The Riesz transform is a useful operation; it just doesn't fit the universal scalar pattern.

**Library status.** Compose only.

#### 4.7.5 Spectral Entropy

The spectral entropy is the von Neumann entropy of the normalized spectrum:

$$S = -\sum_k \tilde{\lambda}_k \ln \tilde{\lambda}_k, \quad \tilde{\lambda}_k = \lambda_k / \sum_j \lambda_j.$$

**Why it fails.** This is a *reduction* (sum over k), not a pointwise multiplier. There is no weight w_k such that applying it to the input recovers the entropy as a spatial-domain output; the entropy is a single scalar summary statistic of the spectrum.

**What to use instead.** Compute the spectrum via `cheap_apply` with an identity weight, then sum −λ ln λ explicitly.

**Library status.** Rejected. Not a primitive in the periodic-table sense.

#### 4.7.6 Ideal Brick-Wall Filter

The hard frequency cutoff w_k = 𝟙(k < K) appears benign — and is in fact shipped as `cheap_weights_kpca_hard` (§4.5.4) — but it suffers from the *Gibbs phenomenon*: the spatial-domain impulse response is the Dirichlet kernel sin(πKx)/(πx), which has slowly-decaying side lobes. Applied to a step-discontinuous signal, the brick-wall filter introduces ~9% overshoot at the discontinuity, regardless of K.

**Why it fails (when used naively).** The Gibbs ringing means the brick-wall is unsuitable as a *smoothing* primitive; it only works as a *projection* primitive (idempotent dimensionality reduction), where the overshoot is a feature (preserves the discontinuity exactly in the truncated subspace) rather than a bug.

**What to use instead.** For low-pass filtering, use a smoother window (Cesàro §4.5.8, Lanczos §4.5.9, or Hanning). The brick-wall ships *only* under the kPCA-hard name, where its projection semantics are explicit.

**Library status.** Shipped under one specific interpretation (kPCA), rejected as a general low-pass filter.

#### 4.7.7 Van Cittert Iteration

The Van Cittert iterative deconvolution procedure x^{n+1} = x^n + (y − Hx^n) has spectral form

$$w_k^{(n)} = 1 - (1 - \psi_k)^n.$$

**Why it fails.** For PSF eigenvalues ψ_k > 1 or ψ_k < 0, the iteration *diverges*: |1 − ψ_k| > 1 means (1 − ψ_k)^n grows in magnitude. Even for well-behaved PSFs with ψ_k ∈ (0, 1), Van Cittert amplifies noise at frequencies where ψ_k ≪ 1 (since w_k → 1/ψ_k as n → ∞, recovering the unregularized inverse). It is a legacy method, superseded in every practical setting by Wiener deconvolution (§4.5.2) or HOT (§4.3.3).

**What to use instead.** Wiener deconvolution or HOT.

**Library status.** Compose only, documented for historical completeness.

---

## 5. Mathematical Analysis

**Theorem 3** *(Eigenvector Perturbation Bound).* The difference between the true eigenvectors of the dfBm covariance and the DCT basis is bounded by O(1/n) for sufficiently large n.

**Proof.** The covariance operator admits an analytic perturbation expansion. By Kato's theorem, the perturbed eigenvectors remain analytic and orthonormal. Bounding the remainder via the Frobenius norm of the perturbation matrix yields the stated decay, ensuring that the DCT becomes an increasingly accurate representation of the Karhunen–Loève Transform for large grids [7]. ∎

**Complexity and Stability.** Each primitive requires two DCTs and O(n) arithmetic operations. Storage is strictly O(n). Because the DCT is a unitary transform, it is perfectly conditioned, ensuring that numerical errors do not propagate beyond the limits of the weight vector **w**.

**Universality of the periodic table.** The six periods of §4 are exhaustive in the following operational sense: every spectral multiplier we have encountered in the literature on stationary covariances, parabolic/hyperbolic PDEs, elliptic inverses, regularized inverse problems, RMT-based denoising, or fractional/multifractal analysis admits expression as one of the listed primitives, a composition thereof, or a member of the Hazard Log (§4.7). The classification is not closed under all conceivable operations (e.g., directional Riesz transforms break the one-array rule), but within the scalar-multiplier doctrine it is — to the best of our knowledge — complete. Identification of new shipping-quality primitives, particularly outside Periods 1–6, would represent a meaningful extension of the framework.

---

## 6. The Pareto Frontier: An Honest Assessment

CHEAP occupies a specific "sweet spot" in the algorithmic trade-off space:

- **Where CHEAP excels**: Large regular 1D, 2D, or 3D grids where asymptotic properties hold. It is highly effective for fractional-Brownian dynamics, stationary kernels, and problems whose covariance is approximately Toeplitz. The Period 1 covariance kernels enable exact GRF sampling at O(N log N) cost; Period 2 propagators provide unconditionally stable parabolic time stepping with no CFL constraint; Period 3 inverses turn elliptic boundary value problems into a single elementwise multiplication; and Period 5 statistical denoisers operate at O(1) cost per eigenvalue, making them competitive with bespoke implementations even for one-shot use.
- **Where CHEAP struggles**: For highly irregular or high-dimensional scattered data, the asymptotic convergence to the DCT basis weakens, and standard Nyström or Cholesky methods may be preferable. Mandelbrot weights (§4.6.5) require Lanczos approximation of the complex Gamma function, introducing a higher constant factor per element than the simpler algebraic weights — though the O(n) scaling is preserved. RMT denoising (§4.5.5–4.5.6) inherits the asymptotic assumptions of the Marchenko–Pastur law: for small sample sizes, finite-size corrections matter and the bulk edges are only approximate. Period 4 wave propagators avoid the CFL constraint but at the cost of dense temporal coupling — a single propagator call jumps the entire interval [0, T] in one step, which is desirable when only the endpoint matters but not when intermediate snapshots are needed.
- **Trade-off**: CHEAP typically trades a modest fraction of fidelity (3–5% relative to dense solvers) for a 100–200× reduction in construction time and a 20–40× reduction in memory. The weight constructor functions add negligible overhead: all are O(n) and dominated by the O(n log n) DCT in any end-to-end pipeline. This profile is ideal for real-time updates, parameter sweeps over κ/ν/ε, and edge deployment where the alternative (a dense solver per call) would be prohibitive.

---

## 7. Conclusion

CHEAP rests on the observation that many of the structured operators that appear across applied mathematics — Toeplitz covariances, screened Poisson inverses, parabolic propagators, hyperbolic propagators, RMT shrinkage estimators, fractional/multifractal weights — share a common spectral structure under the DCT-II. By exploiting this structure we obtain a single algorithmic primitive (DCT, multiply, iDCT) that runs in O(N log N) time and serves an unusually wide range of computational tasks.

The periodic-table organization of §4 makes the structural unity explicit. Period 1 (covariance kernels), Period 2 (parabolic), Period 3 (elliptic), Period 4 (hyperbolic), Period 5 (statistical), and Period 6 (fractional/multifractal) together exhaust the scalar-multiplier doctrine: every primitive in the literature that fits the doctrine has, to our knowledge, been catalogued. Within each period we provide closed-form weights, derivations from the underlying continuum object, DC and stability analysis, and explicit honesty about the regime of validity.

We have been equally explicit about what is *excluded*. The Hazard Log (§4.7) catalogues seven primitives — backward heat, complex-valued Schrödinger propagators, singular Turing growth terms, vector-valued Riesz transforms, scalar reductions like spectral entropy, Gibbs-prone brick-wall filters, and divergent Van Cittert iteration — that violate one or more of the four CHEAP doctrine rules (real-only, stable, SIMD-friendly, one-array). Documenting *why* these do not ship is, we believe, as important as documenting what does: the doctrine is the reason the shipped primitives are predictable and composable.

Two distinct eigenvalue families — the Flandrin spectrum of dfBm covariance and the Laplacian spectrum of the discrete second-difference operator — both admit O(n) construction and exact DCT diagonalization, yielding separate but equally efficient pathways through the same primitive. Period 1 kernels (Matérn, Cauchy, Bessel) typically use Laplacian; Period 6 multifractals (Mandelbrot) use Flandrin; Period 5 statistical primitives accept user-provided eigenvalues from empirical data. The framework is designed so that switching between these bases is a single function-call substitution, not a structural change to the calling code.

Finally, the framework remains open. Several Period 1 (Powered Exponential, Cauchy/RQ, Bessel Potential, GP-Derivative, Rough Volatility), Period 2 (Ornstein–Uhlenbeck, Cahn–Hilliard), Period 3 (Helmholtz/Yukawa, Riesz Potential, Tikhonov H^p), Period 4 (Wave, Damped Wave), Period 5 (BBP Spiked, Geman–McClure), and Period 6 (Sobolev Synthesizer, Kolmogorov 5/3) primitives are catalogued as *Planned* — they are mathematically validated and within the doctrine but await implementation. Identification of additional doctrine-compliant primitives, particularly in physical regimes we have not yet considered (Maxwell, Dirac, Stokes, Reynolds), would represent valid extensions to the periodic table. The end goal is a comprehensive universal spectral periodic table; this paper is its current snapshot.

The result is a package that is mathematically rigorous, numerically stable, structurally organized, and — above all — fiscally responsible.

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

[19] Stein, M. L. (1999). *Interpolation of Spatial Data: Some Theory for Kriging*. Springer Series in Statistics. — Authoritative treatment of the Matérn family and its place in geostatistics.

[20] Whittle, P. (1954). On stationary processes in the plane. *Biometrika*, 41(3/4), 434–449. — Original paper on the Whittle (Matérn-1) covariance kernel.

[21] Adler, R. J., & Taylor, J. E. (2007). *Random Fields and Geometry*. Springer Monographs in Mathematics. — Reference for the geometric and topological structure of Gaussian random fields.

[22] Strang, G., & Fix, G. (2008). *An Analysis of the Finite Element Method* (2nd ed.). Wellesley-Cambridge Press. — Classical reference for Sobolev spaces, fractional Laplacians, and their discretization.

[23] Mallat, S. (2009). *A Wavelet Tour of Signal Processing: The Sparse Way* (3rd ed.). Academic Press. — Reference for Lanczos σ-factors, Cesàro/Fejér windows, and Gibbs-mitigation in spectral truncation.

[24] Cahn, J. W., & Hilliard, J. E. (1958). Free energy of a nonuniform system. I. Interfacial free energy. *J. Chem. Phys.*, 28(2), 258–267. — Original derivation of the Cahn–Hilliard phase-field equation.

[25] Bayer, C., Friz, P., & Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance*, 16(6), 887–904. — Foundational paper on rough volatility models requiring fractional Brownian simulation with H ≪ 1/2.

[26] Baik, J., Ben Arous, G., & Péché, S. (2005). Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. *Annals of Probability*, 33(5), 1643–1697. — Original derivation of the BBP phase transition for spiked covariance models.

[27] Geman, S., & McClure, D. E. (1987). Statistical methods for tomographic image reconstruction. *Bulletin of the International Statistical Institute*, 52(4), 5–21. — Origin of the Geman–McClure robust ρ-function.

[28] Kato, T. (1995). *Perturbation Theory for Linear Operators* (2nd ed., reprint). Springer. — Reference for the analytic perturbation theory underlying Theorems 1 and 3.

[29] Lions, J.-L., & Magenes, E. (1972). *Non-Homogeneous Boundary Value Problems and Applications*, Vol. I. Springer. — Reference for Bessel potentials and fractional Sobolev spaces on bounded domains.

[30] Sneddon, I. N. (1972). *The Use of Integral Transforms*. McGraw-Hill. — Reference for closed-form spectral propagators of the wave, Klein–Gordon, and damped wave equations.
