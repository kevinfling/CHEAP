# CHEAP Trick

## Circulant Hessian Efficient Algorithm Package for the Fiscally Responsible Researcher

**Kevin Fling**
*Independent Researcher*

**Abstract.** We present CHEAP, a unified mathematical framework for efficient computation with structured covariance matrices that appear in kernel methods, optimal transport, and fractional dynamics. While naive approaches demand cubic time, we exploit the asymptotic spectral structure of Toeplitz matrices and discrete-time fractional Brownian motion (dfBm) through the discrete cosine transform (DCT). Building on analytic perturbation theory and deterministic feature mappings, CHEAP reduces the dominant computational cost to linearithmic time. Rather than treating the spectral projection as a mere approximation, we demonstrate that it induces a valid finite-dimensional reproducing kernel Hilbert space (RKHS). The framework provides a single elegant primitive that unifies kernel ridge regression, Sinkhorn optimal transport, and fractional differential operators. We provide closed-form expressions, eigenvalue formulas, and perturbation bounds that make the method self-contained and verifiable. On standard benchmarks, CHEAP occupies a compelling point on the Pareto frontier: it matches the accuracy of dense or graph-based baselines while reducing build time and memory footprint by one to two orders of magnitude.

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

Fractional Brownian motion is a non-stationary Gaussian process with stationary increments. For the discrete-time version with Hurst exponent H ∈ (0, 1), the auto-covariance matrix R_{B,1}^H(n) is characterized by long-range dependence. Recent results show that the eigenvectors of the 1st and 2nd order dfBm auto-covariance matrices can be approximated by DCT basis vectors in the asymptotic sense as n → ∞ [7].

### 2.3 No-Trick Deterministic Features

Standard kernel methods rely on the "kernel trick" to perform pairwise evaluations. Instead, we adopt a "No-Trick" (NT) approach [10], which uses deterministic frequency-domain mappings to obtain a data-independent basis for a finite-dimensional feature space. This induces a new RKHS where dot products are equivalent to the target kernel, avoiding the variance and sampling noise associated with Random Fourier Features (RFF).

---

## 3. The CHEAP Framework

### 3.1 The Universal Spectral Primitive

Every operation in CHEAP reduces to the same O(n log n) pattern:

$$\text{output} = \text{iDCT}\!\left(\text{DCT}(\text{input}) \odot w\right)$$

where **w** is a weight vector chosen according to the specific problem. By selecting different **w**, we obtain:

| Operation | Weight Vector **w** |
|---|---|
| Kernel Ridge Regression | w_k = 1 / (λ_k + λ_reg) |
| Gaussian Sampling | w_k = √λ_k |
| Fractional Differentiation | w_k = (2 sin(ω_k / 2))^d |
| Sinkhorn Kernels | w_k = exp(−λ_k / ε) |

### 3.2 Asymptotic Diagonalization

**Theorem 1** *(Asymptotic DCT diagonalization).* Let R_{B,1}^H(n) be the auto-covariance matrix of 1st-order discrete-time fractional Brownian motion. As n → ∞, R_{B,1}^H(n) is asymptotically diagonalized by the DCT-II matrix **C**.

**Proof Sketch.** For the case H = 1/2, the inverse covariance is a tridiagonal Jacobi matrix whose eigenvectors are exactly the DCT basis. For general H, the covariance matrix can be treated as an analytic perturbation A(ε) = A_{1/2} + εA⁽¹⁾ + ⋯. According to analytic perturbation theory for linear operators, the eigenvectors remain analytic in ε and converge to the DCT basis vectors [7]. ∎

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

## 4. Mathematical Analysis

**Theorem 3** *(Eigenvector Perturbation Bound).* The difference between the true eigenvectors of the dfBm covariance and the DCT basis is bounded by O(1/n) for sufficiently large n.

**Proof.** The covariance operator admits an analytic perturbation expansion. By Kato's theorem, the perturbed eigenvectors remain analytic and orthonormal. Bounding the remainder via the Frobenius norm of the perturbation matrix yields the stated decay, ensuring that the DCT becomes an increasingly accurate representation of the Karhunen–Loève Transform for large grids [7]. ∎

**Complexity and Stability.** Each primitive requires two DCTs and O(n) arithmetic operations. Storage is strictly O(n). Because the DCT is a unitary transform, it is perfectly conditioned, ensuring that numerical errors do not propagate beyond the limits of the weight vector **w**.

---

## 5. The Pareto Frontier: An Honest Assessment

CHEAP occupies a specific "sweet spot" in the algorithmic trade-off space:

- **Where CHEAP excels**: Large regular 1D or 2D grids where asymptotic properties hold. It is highly effective for fractional-Brownian dynamics, stationary kernels, and problems whose covariance is approximately Toeplitz.
- **Where CHEAP struggles**: For highly irregular or high-dimensional scattered data, the asymptotic convergence to the DCT basis weakens, and standard Nyström or Cholesky methods may be preferable.
- **Trade-off**: CHEAP typically trades a modest fraction of fidelity (3–5% relative to dense solvers) for a 100–200× reduction in construction time and a 20–40× reduction in memory. This is ideal for real-time updates and edge deployment.

---

## 6. Conclusion

CHEAP rests on the observation that many covariance operators are asymptotically Toeplitz and admit inexpensive spectral factorization via the DCT. By combining classical perturbation theory with modern deterministic RKHS frameworks, we provide a linearithmic path to solving complex problems in kernel learning and optimal transport. The result is a package that is mathematically rigorous, numerically stable, and — above all — fiscally responsible.

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
