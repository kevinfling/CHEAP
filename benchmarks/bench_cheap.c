/*
 * bench_cheap.c — micro-benchmarks for the CHEAP library.
 * Core benchmarks: apply, forward/inverse, Sinkhorn, Toeplitz, RFF
 *
 * Build (standalone):
 *   gcc -std=c99 -pedantic -Wall -Wextra -Werror -march=native -O3 \
 *       -D_POSIX_C_SOURCE=199309L \
 *       bench_cheap.c -o bench_cheap -lfftw3 -lm
 *
 * Output columns (tab-separated, '#'-prefixed header for grep):
 *   algo  N  wall_ms  ticks
 */

#include "cheap.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* =========================================================================
 * Portable tick counter
 * ========================================================================= */
static inline uint64_t bench_ticks(void)
{
#if defined(__x86_64__) || defined(__i386__)
    return cheap_rdtsc();
#elif defined(__aarch64__)
    uint64_t v;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static double wall_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#define WARMUP_ITERS  3
#define BENCH_ITERS  10

#define CHECK_RC(expr) do { \
    int _rc = (expr); \
    if (_rc != CHEAP_OK) { \
        fprintf(stderr, "%s:%d: %s failed (rc=%d)\n", __FILE__, __LINE__, #expr, _rc); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static void run_bench(void (*bench_fn)(void *), void *state,
                      double *out_wall_ms, uint64_t *out_ticks)
{
    for (int i = 0; i < WARMUP_ITERS; ++i) bench_fn(state);
    double  t0 = wall_seconds();
    uint64_t c0 = bench_ticks();
    for (int i = 0; i < BENCH_ITERS; ++i) bench_fn(state);
    uint64_t c1 = bench_ticks();
    double  t1 = wall_seconds();
    *out_wall_ms = (t1 - t0) * 1e3 / BENCH_ITERS;
    *out_ticks   = (c1 - c0) / (uint64_t)BENCH_ITERS;
}

static void print_result(const char *algo, int n,
                         double wall_ms, uint64_t ticks)
{
    printf("%-24s %6d   %10.6f   %12" PRIu64 "\n",
           algo, n, wall_ms, ticks);
}

/* =========================================================================
 * Core algorithm benchmarks
 * ========================================================================= */

typedef struct { cheap_ctx ctx; double *input; double *weights; double *output; } apply_state;
static void bench_apply(void *p) {
    apply_state *s = (apply_state *)p;
    cheap_apply(&s->ctx, s->input, s->weights, s->output);
}

typedef struct { cheap_ctx ctx; double *input; double *output; } fwd_inv_state;
static void bench_forward(void *p) {
    fwd_inv_state *s = (fwd_inv_state *)p;
    cheap_forward(&s->ctx, s->input);
}
static void bench_inverse(void *p) {
    fwd_inv_state *s = (fwd_inv_state *)p;
    cheap_inverse(&s->ctx, s->output);
}

typedef struct { cheap_ctx ctx; double *a; double *b; double *f; double *g; } sinkhorn_state;
static void bench_sinkhorn(void *p) {
    sinkhorn_state *s = (sinkhorn_state *)p;
    cheap_sinkhorn(&s->ctx, s->a, s->b, 0.5, 50, 1e-15, s->f, s->g);
}

/* =========================================================================
 * Toeplitz benchmarks
 * ========================================================================= */

typedef struct { cheap_ctx ctx; double *lam; double *x; double *y; } toep_state;
static void bench_toeplitz_matvec_pre(void *p) {
    toep_state *s = (toep_state *)p;
    cheap_apply(&s->ctx, s->x, s->lam, s->y);
}
static void bench_toeplitz_solve(void *p) {
    toep_state *s = (toep_state *)p;
    cheap_toeplitz_solve_precomp(&s->ctx, s->lam, s->x, 1e-3, s->y);
}

/* =========================================================================
 * RFF benchmarks
 * ========================================================================= */

typedef struct { cheap_rff_ctx rctx; double *x_in; double *z_out; } rff_state;
static void bench_rff_map(void *p) {
    rff_state *s = (rff_state *)p;
    cheap_rff_map(&s->rctx, s->x_in, s->z_out);
}

typedef struct { cheap_rff_ctx rctx; double *X_in; double *Z_out; int N; } rff_batch_state;
static void bench_rff_map_batch(void *p) {
    rff_batch_state *s = (rff_batch_state *)p;
    cheap_rff_map_batch(&s->rctx, s->X_in, s->N, s->Z_out);
}

/* =========================================================================
 * Benchmark runners
 * ========================================================================= */

static void run_core_benchmarks(int n)
{
    const double H = 0.7;
    if (n >= 8192) fprintf(stderr, "  Planning FFTW for n=%d ...\n", n);

    /* cheap_apply with KRR weights */
    {
        apply_state *s = (apply_state *)malloc(sizeof(*s));
        CHECK_RC(cheap_init(&s->ctx, n, H));
        s->input   = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->weights = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->output  = (double *)fftw_malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; ++i) s->input[i] = sin(2.0 * M_PI * i / n) + 1.0;
        for (int k = 0; k < n; ++k) {
            double denom = s->ctx.lambda[k] + 1e-3;
            s->weights[k] = 1.0 / denom;
        }
        double wms; uint64_t tk;
        run_bench(bench_apply, s, &wms, &tk);
        print_result("apply_krr", n, wms, tk);
        cheap_destroy(&s->ctx);
        fftw_free(s->input); fftw_free(s->weights); fftw_free(s->output); free(s);
    }
    /* cheap_apply with sqrt_lambda (reparam) */
    {
        apply_state *s = (apply_state *)malloc(sizeof(*s));
        CHECK_RC(cheap_init(&s->ctx, n, H));
        s->input   = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->weights = s->ctx.sqrt_lambda;  /* borrow — don't free */
        s->output  = (double *)fftw_malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; ++i) s->input[i] = 1.0;
        double wms; uint64_t tk;
        run_bench(bench_apply, s, &wms, &tk);
        print_result("apply_reparam", n, wms, tk);
        cheap_destroy(&s->ctx);
        fftw_free(s->input); fftw_free(s->output); free(s);
    }
    /* cheap_forward */
    {
        fwd_inv_state *s = (fwd_inv_state *)malloc(sizeof(*s));
        CHECK_RC(cheap_init(&s->ctx, n, H));
        s->input  = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->output = (double *)fftw_malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; ++i) s->input[i] = sin(2.0 * M_PI * i / n);
        double wms; uint64_t tk;
        run_bench(bench_forward, s, &wms, &tk);
        print_result("forward", n, wms, tk);
        cheap_destroy(&s->ctx);
        fftw_free(s->input); fftw_free(s->output); free(s);
    }
    /* cheap_inverse */
    {
        fwd_inv_state *s = (fwd_inv_state *)malloc(sizeof(*s));
        CHECK_RC(cheap_init(&s->ctx, n, H));
        s->input  = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->output = (double *)fftw_malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; ++i) s->input[i] = sin(2.0 * M_PI * i / n);
        cheap_forward(&s->ctx, s->input);
        double wms; uint64_t tk;
        run_bench(bench_inverse, s, &wms, &tk);
        print_result("inverse", n, wms, tk);
        cheap_destroy(&s->ctx);
        fftw_free(s->input); fftw_free(s->output); free(s);
    }
    /* Sinkhorn */
    {
        sinkhorn_state *s = (sinkhorn_state *)malloc(sizeof(*s));
        s->a = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->b = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->f = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->g = (double *)fftw_malloc((size_t)n * sizeof(double));
        CHECK_RC(cheap_init(&s->ctx, n, H));
        for (int i = 0; i < n; ++i) s->a[i] = s->b[i] = 1.0 / (double)n;
        double wms; uint64_t tk;
        run_bench(bench_sinkhorn, s, &wms, &tk);
        print_result("sinkhorn_50", n, wms, tk);
        cheap_destroy(&s->ctx);
        fftw_free(s->a); fftw_free(s->b); fftw_free(s->f); fftw_free(s->g);
        free(s);
    }
}

static void run_toeplitz_benchmarks(int n)
{
    if (n >= 8192) fprintf(stderr, "  Planning FFTW for n=%d ...\n", n);
    toep_state *s = (toep_state *)malloc(sizeof(*s));
    CHECK_RC(cheap_init(&s->ctx, n, 0.5));

    double *t = (double *)calloc((size_t)n, sizeof(double));
    t[0] = 4.0; t[1] = -1.0;

    s->lam = (double *)fftw_malloc((size_t)n * sizeof(double));
    s->x = (double *)fftw_malloc((size_t)n * sizeof(double));
    s->y = (double *)fftw_malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i) s->x[i] = sin(2.0 * M_PI * i / n) + 1.0;
    cheap_toeplitz_eigenvalues(&s->ctx, t, s->lam);

    double wms; uint64_t tk;

    run_bench(bench_toeplitz_matvec_pre, s, &wms, &tk);
    print_result("toeplitz_matvec_pre", n, wms, tk);

    run_bench(bench_toeplitz_solve, s, &wms, &tk);
    print_result("toeplitz_solve_pre", n, wms, tk);

    cheap_destroy(&s->ctx);
    free(t); fftw_free(s->lam); fftw_free(s->x); fftw_free(s->y);
    free(s);
}

static void run_rff_benchmarks(void)
{
    int D_vals[] = {64, 256, 1024};
    double wms; uint64_t tk;

    for (int di = 0; di < 3; ++di) {
        int D = D_vals[di];
        rff_state *s = (rff_state *)malloc(sizeof(*s));
        cheap_rff_init(&s->rctx, D, 1, 1.0, 42);
        s->x_in = (double *)malloc(sizeof(double));
        s->z_out = (double *)malloc((size_t)D * sizeof(double));
        s->x_in[0] = 0.5;

        run_bench(bench_rff_map, s, &wms, &tk);
        print_result("rff_map", D, wms, tk);

        free(s->x_in); free(s->z_out);
        cheap_rff_destroy(&s->rctx);
        free(s);
    }

    int N_vals[] = {1024, 8192};
    for (int ni = 0; ni < 2; ++ni) {
        int N = N_vals[ni];
        rff_batch_state *s = (rff_batch_state *)malloc(sizeof(*s));
        cheap_rff_init(&s->rctx, 256, 1, 1.0, 42);
        s->N = N;
        s->X_in = (double *)malloc((size_t)N * sizeof(double));
        s->Z_out = (double *)malloc((size_t)(N * 256) * sizeof(double));
        for (int i = 0; i < N; ++i) s->X_in[i] = (double)i * 0.001;

        run_bench(bench_rff_map_batch, s, &wms, &tk);
        print_result("rff_map_batch_256", N, wms, tk);

        free(s->X_in); free(s->Z_out);
        cheap_rff_destroy(&s->rctx);
        free(s);
    }
}

/* =========================================================================
 * Spectral weight constructor benchmarks
 * ========================================================================= */

typedef struct { int n; double* lam; double* w; } weight_state;

static void bench_weights_fractional(void *p) {
    weight_state *s = (weight_state *)p;
    cheap_weights_fractional(s->n, 0.4, s->w);
}
static void bench_weights_wiener(void *p) {
    weight_state *s = (weight_state *)p;
    cheap_weights_wiener(s->n, 1.0, s->w);
}
static void bench_weights_specnorm(void *p) {
    weight_state *s = (weight_state *)p;
    cheap_weights_specnorm(s->n, 1e-3, s->w);
}
static void bench_weights_mandelbrot(void *p) {
    weight_state *s = (weight_state *)p;
    cheap_weights_mandelbrot(s->n, 0.7, s->w);
}
static void bench_weights_rmt_shrink(void *p) {
    weight_state *s = (weight_state *)p;
    cheap_weights_rmt_shrink(s->lam, s->n, 1.0, 0.5, s->w);
}

static void run_weight_benchmarks(int n)
{
    weight_state *s = (weight_state *)malloc(sizeof(*s));
    s->n = n;
    s->lam = (double *)fftw_malloc((size_t)n * sizeof(double));
    s->w   = (double *)fftw_malloc((size_t)n * sizeof(double));
    /* Fill with dummy eigenvalues for RMT */
    for (int i = 0; i < n; ++i)
        s->lam[i] = 1.0 + 5.0 * (double)i / (double)n;

    double wms; uint64_t tk;

    run_bench(bench_weights_fractional, s, &wms, &tk);
    print_result("wt_fractional", n, wms, tk);

    run_bench(bench_weights_wiener, s, &wms, &tk);
    print_result("wt_wiener", n, wms, tk);

    run_bench(bench_weights_specnorm, s, &wms, &tk);
    print_result("wt_specnorm", n, wms, tk);

    run_bench(bench_weights_mandelbrot, s, &wms, &tk);
    print_result("wt_mandelbrot", n, wms, tk);

    /* Re-fill for RMT (needs eigenvalues above threshold) */
    double sc = sqrt(0.5);
    double lp = (1.0 + sc) * (1.0 + sc);
    for (int i = 0; i < n; ++i)
        s->lam[i] = lp + 1.0 + 5.0 * (double)i / (double)n;
    run_bench(bench_weights_rmt_shrink, s, &wms, &tk);
    print_result("wt_rmt_shrink", n, wms, tk);

    fftw_free(s->lam); fftw_free(s->w);
    free(s);
}

/* End-to-end: weight computation + apply */
typedef struct { cheap_ctx ctx; double *input; double *weights; double *output; int n; } apply_wt_state;
static void bench_apply_wiener(void *p) {
    apply_wt_state *s = (apply_wt_state *)p;
    cheap_weights_wiener(s->n, 1.0, s->weights);
    cheap_apply(&s->ctx, s->input, s->weights, s->output);
}

static void run_apply_weight_benchmarks(int n)
{
    apply_wt_state *s = (apply_wt_state *)malloc(sizeof(*s));
    CHECK_RC(cheap_init(&s->ctx, n, 0.7));
    s->n = n;
    s->input   = (double *)fftw_malloc((size_t)n * sizeof(double));
    s->weights = (double *)fftw_malloc((size_t)n * sizeof(double));
    s->output  = (double *)fftw_malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i)
        s->input[i] = sin(2.0 * M_PI * i / n) + 1.0;

    double wms; uint64_t tk;
    run_bench(bench_apply_wiener, s, &wms, &tk);
    print_result("apply_wiener_e2e", n, wms, tk);

    cheap_destroy(&s->ctx);
    fftw_free(s->input); fftw_free(s->weights); fftw_free(s->output);
    free(s);
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(void)
{
    static const int sizes[] = {1024, 8192, 65536};
    static const int nsizes  = 3;

    printf("# %-22s %6s   %10s   %12s\n",
           "algo", "N", "wall_ms", "ticks");

    /* Core algorithms */
    for (int i = 0; i < nsizes; ++i)
        run_core_benchmarks(sizes[i]);

    /* Toeplitz */
    for (int i = 0; i < nsizes; ++i)
        run_toeplitz_benchmarks(sizes[i]);

    /* RFF */
    run_rff_benchmarks();

    /* Spectral weight constructors */
    for (int i = 0; i < nsizes; ++i)
        run_weight_benchmarks(sizes[i]);

    /* End-to-end apply with weight computation */
    for (int i = 0; i < nsizes; ++i)
        run_apply_weight_benchmarks(sizes[i]);

    return 0;
}
