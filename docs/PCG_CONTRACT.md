# PCG contract hardening

The original `cheap_pcg_solve` signature is retained. New callers may
use `cheap_pcg_solve_opts` and `cheap_pcg_options` to select the residual
recomputation interval (including zero), zero initial guess, and
residual-replacement policy.

Both solve the caller's **original SPD A**, not the DCT surrogate.
The optional DCT-diagonal operator is only an SPD preconditioner;
strictly positive, finite spectral weights are required at construction.

## Residual validity

`result.residual_valid=1` means `true_residual` is the norm of a
successfully evaluated `b-Ax`. Otherwise the residual and relative
residual are NaN. An A callback failure on the final check invalidates
both values even if an earlier residual was known. A successful
max-iteration or non-curvature breakdown exit attempts a fresh A(x)
and reports a valid residual when that call succeeds.

A zero RHS uses relative residual 0 by convention, even when its
absolute residual is nonzero. The stopping rule is
`||b-Ax|| <= atol + rtol*||b||`, evaluated against a true residual
before reporting convergence.

## Replacement policy

`CHEAP_PCG_REFRESH_RESTART` (the legacy default) discards the prior
Krylov direction when a true residual replaces the recursive residual.
It is a sequence of restarted PCG cycles, not uninterrupted classical
CG, and may take more iterations.

`CHEAP_PCG_REFRESH_CONTINUE` substitutes the true residual while
retaining the prior search direction and rho. This is a finite-precision
residual-replacement heuristic; it is not guaranteed to preserve exact
CG conjugacy. Both policies require a fresh true residual before
declaring convergence. `residual_recompute_interval=0` disables
periodic replacements, but still recomputes at entry, when the
recursive estimate reaches tolerance, and on termination.

Neither policy promises universal convergence or certifies the SPD
property of an arbitrary user callback.

## Failure codes

`CHEAP_PCG_BREAKDOWN_CURVATURE`: observed `p^T A p <= 0`.
`CHEAP_PCG_BREAKDOWN_NONFINITE`: NaN/Inf in vectors or arithmetic.
`CHEAP_PCG_PRECOND_INVALID`: failed preconditioner callback or
nonpositive preconditioned inner product.
`CHEAP_PCG_CALLBACK_FAILED`: original operator callback failed.
`CHEAP_PCG_MAX_ITER`: iteration budget exhausted.
`CHEAP_PCG_INVALID`: invalid input contract (legacy value retained).

All iteration scratch is preallocated in `cheap_pcg_workspace`.
The solve loop contains no malloc or FFT planning.
