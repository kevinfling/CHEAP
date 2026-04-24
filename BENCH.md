# CHEAP Benchmarks

## 1D

### ARM64 (ARMv8 rev 1 @ ~2.0 GHz, GCC 11.4.0, `-O3 -march=native`)

| Algorithm | N=1,024 | N=8,192 | N=65,536 | Scaling |
|-----------|---------|---------|----------|---------|
| KRR solve | 0.018 ms | 0.180 ms | 2.54 ms | O(N log N) |
| Sinkhorn (50 iters) | 0.087 ms | 0.839 ms | 8.77 ms | O(N log N) |
| Reparameterization | 0.018 ms | 0.181 ms | 2.45 ms | O(N log N) |
| Toeplitz matvec | 0.018 ms | 0.180 ms | 2.54 ms | O(N log N) |
| RFF map (D=256) | 0.004 ms | — | — | O(D) |

### Statistical Benchmarks (30 trials, FFTW_PATIENT)

| Algorithm | N | Mean (ms) | CV% | Status |
|-----------|---|-----------|-----|--------|
| krr_solve | 65,536 | 5.279 | 0.08% | STABLE |
| reparam | 65,536 | 5.363 | 0.11% | STABLE |
| sinkhorn_50 | 65,536 | 16.613 | 0.46% | STABLE |
| toeplitz_matvec | 65,536 | 4.924 | 0.06% | STABLE |

All CV < 0.5%, 95% CI within 0.2% of mean.

## 2D

| N_total | Algorithm | cyc/el (vectorized) | cyc/el (scalar) | speedup | in-place savings % |
|---------|-----------|---------------------|-----------------|---------|-------------------|
| 4,096 (64×64) | apply_krr_2d | — | — | — | — |
| 16,384 (128×128) | apply_krr_2d | — | — | — | — |
| 65,536 (256×256) | apply_krr_2d | — | — | — | — |
| 262,144 (512×512) | apply_krr_2d | — | — | — | — |

Run `./build/bench_cheap` and `./build/bench_cheap_scalar` to fill the table.

## 3D

| N_total | Algorithm | cyc/el (vectorized) | cyc/el (scalar) | speedup | in-place savings % |
|---------|-----------|---------------------|-----------------|---------|-------------------|
| 4,096 (16³) | apply_krr_3d | — | — | — | — |
| 32,768 (32³) | apply_krr_3d | — | — | — | — |
| 262,144 (64³) | apply_krr_3d | — | — | — | — |

Run `./build/bench_cheap` and `./build/bench_cheap_scalar` to fill the table.

## Weight Constructors (v0.3.0)

Weight constructors are pure O(N) spectral math — no FFTW plans, no ctx required. The DCT in `cheap_apply` at O(N log N) dominates total cost; weight construction is never the bottleneck.

### ARM64 (ARMv8, `-O3 -march=native`, NEON enabled)

| Constructor | N = 1,024 | N = 65,536 | N = 1M | SIMD |
|---|---|---|---|---|
| `laplacian_ev` | — | — | — | scalar (sin) |
| `matern_ev` (κ=1, ν=1.5) | — | — | — | scalar (pow) |
| `matern_2d` (32×32) | — | — | — | scalar (pow) |
| `matern_3d` (16×16×4) | — | — | — | scalar (pow) |
| `anisotropic_matern_2d` (32×32) | — | — | — | scalar (pow) |
| `heat_propagator_ev` (t=0.1) | — | — | — | scalar (exp) |
| `heat_propagator_2d` (32×32) | — | — | — | scalar (exp) |
| `biharmonic_ev` | — | — | — | NEON 2×f64 |
| `biharmonic_2d` (32×32) | — | — | — | NEON (pass 2) |
| `poisson_ev` | — | — | — | NEON 2×f64 |
| `poisson_2d` (32×32) | — | — | — | NEON (pass 2) |
| `higher_order_tikhonov_deconv_ev` (p=2) | — | — | — | scalar (pow) |

Run `./build/bench_cheap` to fill. All times in ms or cycles/element.

**Expected speedup (biharmonic_ev, poisson_ev):** ≥ 1.8× over scalar on NEON (2×f64 per cycle vs 1×). AVX2 target: ≥ 3.5×.
