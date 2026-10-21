# Exact Toeplitz FFT matvec (Phase 1)

This module computes the **actual finite symmetric Toeplitz matrix-vector
product**, not the DCT-diagonal approximation. Given a first column
`c[0..n-1]`, the matrix is `T[i,j] = c[abs(i-j)]`. The operator
returns `T x` using a length-`L` circulant embedding:

```
[c0, c1, ..., c(n-1), zeros, c(n-1), ..., c1]
```

where `L` is the smallest power of two at least `2n-1`. The cached
embedding spectrum is multiplied by the FFT of a zero-padded input.
An unnormalized inverse FFT is divided by `L`, and the first `n`
samples are copied out.

## API and contracts

Include `cheap_toeplitz_operator.h`. Initialize a zeroed
`cheap_toeplitz_operator` with `cheap_toeplitz_operator_init`;
call `cheap_toeplitz_operator_apply` repeatedly; finish with
`cheap_toeplitz_operator_destroy`. The status-returning
`cheap_linear_operator` and `cheap_matvec_fn` permit future iterative
methods without adding PCG to this change.

- **Input:** finite first column, finite vector, `n >= 1`, representable
  embedding and allocation sizes.
- **Aliasing:** exact `x == y` is supported. Partial overlap is rejected.
  On a failed apply, `y` is unchanged.
- **Lifecycle:** initialize only a fresh zeroed/destroyed context;
  failed initialization leaves it empty. Destroy is idempotent.
- **Concurrency:** independent contexts can execute concurrently.
  A single context must not be applied concurrently or copied by value.
- **Hot path:** no heap allocation, plan creation, or plan destruction.
  Execution overwrites context-owned scratch buffers only.
- **Numerical:** the inverse FFT is normalized by `1/L`. The
  mathematical operator is Toeplitz matvec, up to floating-point error.
  The existing DCT-diagonal surrogate is a *different* operator.
- **Error codes:** `CHEAP_EINVAL` for invalid sizes/overlap,
  `CHEAP_EDOM` for nonfinite inputs/results, `CHEAP_EUNINIT` for
  unusable context, `CHEAP_ENOMEM` for allocation/planner failure.
  Planner failure is not yet distinguished from true allocation failure.

## Cost model

Let `L = 2^ceil(log2(max(1, 2n-1)))`.

- **Construction:** O(L log L) to FFT the circulant embedding, plus
  backend-specific plan construction.
- **Repeated application:** O(L log L) arithmetic, O(n) output copy,
  zero dynamic allocations or replanning.
- **Owned memory:** three FFT buffers: two real buffers of `L` doubles
  (one time-domain buffer and one implicit storage allocation equivalent
  to the two complex buffers) more precisely
  `8L + 2*16*(floor(L/2)+1)` bytes, excluding FFT plan overhead.
- **Numerical tolerance:** tests compare against long-double accumulated
  O(n²) reference, with an absolute term scaled by the row's sum of
  absolute products and an FFT-length logarithmic factor.

## CI acceptance

The test is registered with CTest and inherits the existing
FFTW3/Notorious × Debug/ASan/UBSan six-job GitHub Actions matrix.
Test lengths: 1, 2, 3, 5, 8, 17, 64, 257. Test vectors include impulses,
ones, alternating signs and near-cancellation; also repeated applies,
in-place use, nonfinite inputs, size overflow, independent contexts,
idempotent destroy, and transactional initialization.

No PCG, sampling, or claims of exactness for the DCT surrogate are
included in this phase.
