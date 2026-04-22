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

#define WARMUP_ITERS  10
#define BENCH_ITERS   1000

#define CHECK_RC(expr) do { \
    int _rc = (expr); \
    if (_rc != CHEAP_OK) { \
        fprintf(stderr, "%s:%d: %s failed (rc=%d)\n", __FILE__, __LINE__, #expr, _rc); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

typedef struct {
    double   wall_ms_min, wall_ms_median, wall_ms_mean, wall_ms_stddev;
    uint64_t ticks_min, ticks_median;
    double   ticks_mean;
    double   cycles_per_el;   /* ticks_median / n */
} bench_stats;

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static int cmp_dbl(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void run_bench(void (*bench_fn)(void *), void *state, int n,
                      bench_stats *out)
{
    static double   wall_samples[BENCH_ITERS];
    static uint64_t tick_samples[BENCH_ITERS];

    for (int i = 0; i < WARMUP_ITERS; ++i) bench_fn(state);
    for (int i = 0; i < BENCH_ITERS; ++i) {
        uint64_t c0 = bench_ticks();
        double   t0 = wall_seconds();
        bench_fn(state);
        double   t1 = wall_seconds();
        uint64_t c1 = bench_ticks();
        wall_samples[i] = (t1 - t0) * 1e3;
        tick_samples[i] = c1 - c0;
    }

    /* Mean + stddev (uses full sample) */
    double sum = 0.0;
    double tick_sum = 0.0;
    for (int i = 0; i < BENCH_ITERS; ++i) {
        sum      += wall_samples[i];
        tick_sum += (double)tick_samples[i];
    }
    double mean = sum / (double)BENCH_ITERS;
    double var = 0.0;
    for (int i = 0; i < BENCH_ITERS; ++i) {
        double d = wall_samples[i] - mean;
        var += d * d;
    }
    var /= (double)(BENCH_ITERS - 1);

    /* Sort for min + median (uses scratch copies) */
    static double   wall_sorted[BENCH_ITERS];
    static uint64_t tick_sorted[BENCH_ITERS];
    memcpy(wall_sorted, wall_samples, sizeof(wall_samples));
    memcpy(tick_sorted, tick_samples, sizeof(tick_samples));
    qsort(wall_sorted, BENCH_ITERS, sizeof(double),   cmp_dbl);
    qsort(tick_sorted, BENCH_ITERS, sizeof(uint64_t), cmp_u64);

    out->wall_ms_min    = wall_sorted[0];
    out->wall_ms_median = wall_sorted[BENCH_ITERS / 2];
    out->wall_ms_mean   = mean;
    out->wall_ms_stddev = sqrt(var);
    out->ticks_min      = tick_sorted[0];
    out->ticks_median   = tick_sorted[BENCH_ITERS / 2];
    out->ticks_mean     = tick_sum / (double)BENCH_ITERS;
    out->cycles_per_el  = (double)out->ticks_median / (double)n;
}

static void print_header(void)
{
    printf("# %-22s %6s   %10s %10s %10s %10s   %12s %12s   %10s\n",
           "algo", "N",
           "wall_min", "wall_med", "wall_mean", "wall_stdd",
           "tick_min", "tick_med",
           "cyc/el");
}

static void print_result(const char *algo, int n, const bench_stats *s)
{
    printf("%-24s %6d   %10.6f %10.6f %10.6f %10.6f   %12" PRIu64 " %12" PRIu64 "   %10.2f\n",
           algo, n,
           s->wall_ms_min, s->wall_ms_median, s->wall_ms_mean, s->wall_ms_stddev,
           s->ticks_min, s->ticks_median,
           s->cycles_per_el);
}

/* =========================================================================
 * Core algorithm benchmarks
 * ========================================================================= */

typedef struct { cheap_ctx ctx; double *input; double *weights; double *output; } apply_state;
static void bench_apply(void *p) {
    apply_state *s = (apply_state *)p;
    cheap_apply(&s->ctx, s->input, s->weights, s->output);
}

typedef struct { cheap_ctx ctx; double *input; double *weights; } apply_inplace_state;
static void bench_apply_inplace(void *p) {
    apply_inplace_state *s = (apply_inplace_state *)p;
    /* Re-seed workspace each iter so the measurement is independent of prior
     * iterations' output (cheap_apply_inplace overwrites workspace). The memcpy
     * matches the one baked into cheap_apply — this is a fair apples-to-apples
     * comparison against "apply_krr" above, isolating the SIMD kernel + elimination
     * of the final scale-copy pass. */
    memcpy(s->ctx.workspace, s->input, (size_t)s->ctx.n * sizeof(double));
    cheap_apply_inplace(&s->ctx, s->weights);
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
        bench_stats st;
        run_bench(bench_apply, s, n, &st);
        print_result("apply_krr", n, &st);
        cheap_destroy(&s->ctx);
        fftw_free(s->input); fftw_free(s->weights); fftw_free(s->output); free(s);
    }
    /* cheap_apply_inplace with KRR weights (same workload as apply_krr) */
    {
        apply_inplace_state *s = (apply_inplace_state *)malloc(sizeof(*s));
        CHECK_RC(cheap_init(&s->ctx, n, H));
        s->input   = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->weights = (double *)fftw_malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; ++i) s->input[i] = sin(2.0 * M_PI * i / n) + 1.0;
        for (int k = 0; k < n; ++k) s->weights[k] = 1.0 / (s->ctx.lambda[k] + 1e-3);
        bench_stats st;
        run_bench(bench_apply_inplace, s, n, &st);
        print_result("apply_krr_inplace", n, &st);
        cheap_destroy(&s->ctx);
        fftw_free(s->input); fftw_free(s->weights); free(s);
    }
    /* cheap_apply with sqrt_lambda (reparam) */
    {
        apply_state *s = (apply_state *)malloc(sizeof(*s));
        CHECK_RC(cheap_init(&s->ctx, n, H));
        s->input   = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->weights = s->ctx.sqrt_lambda;  /* borrow — don't free */
        s->output  = (double *)fftw_malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; ++i) s->input[i] = 1.0;
        bench_stats st;
        run_bench(bench_apply, s, n, &st);
        print_result("apply_reparam", n, &st);
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
        bench_stats st;
        run_bench(bench_forward, s, n, &st);
        print_result("forward", n, &st);
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
        bench_stats st;
        run_bench(bench_inverse, s, n, &st);
        print_result("inverse", n, &st);
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
        bench_stats st;
        run_bench(bench_sinkhorn, s, n, &st);
        print_result("sinkhorn_50", n, &st);
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

    bench_stats st;

    run_bench(bench_toeplitz_matvec_pre, s, n, &st);
    print_result("toeplitz_matvec_pre", n, &st);

    run_bench(bench_toeplitz_solve, s, n, &st);
    print_result("toeplitz_solve_pre", n, &st);

    cheap_destroy(&s->ctx);
    free(t); fftw_free(s->lam); fftw_free(s->x); fftw_free(s->y);
    free(s);
}

static void run_rff_benchmarks(void)
{
    int D_vals[] = {64, 256, 1024};
    bench_stats st;

    for (int di = 0; di < 3; ++di) {
        int D = D_vals[di];
        rff_state *s = (rff_state *)malloc(sizeof(*s));
        cheap_rff_init(&s->rctx, D, 1, 1.0, 42);
        s->x_in = (double *)malloc(sizeof(double));
        s->z_out = (double *)malloc((size_t)D * sizeof(double));
        s->x_in[0] = 0.5;

        run_bench(bench_rff_map, s, D, &st);
        print_result("rff_map", D, &st);

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

        run_bench(bench_rff_map_batch, s, N, &st);
        print_result("rff_map_batch_256", N, &st);

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

    bench_stats st;

    run_bench(bench_weights_fractional, s, n, &st);
    print_result("wt_fractional", n, &st);

    run_bench(bench_weights_wiener, s, n, &st);
    print_result("wt_wiener", n, &st);

    run_bench(bench_weights_specnorm, s, n, &st);
    print_result("wt_specnorm", n, &st);

    run_bench(bench_weights_mandelbrot, s, n, &st);
    print_result("wt_mandelbrot", n, &st);

    /* Re-fill for RMT (needs eigenvalues above threshold) */
    double sc = sqrt(0.5);
    double lp = (1.0 + sc) * (1.0 + sc);
    for (int i = 0; i < n; ++i)
        s->lam[i] = lp + 1.0 + 5.0 * (double)i / (double)n;
    run_bench(bench_weights_rmt_shrink, s, n, &st);
    print_result("wt_rmt_shrink", n, &st);

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

    bench_stats st;
    run_bench(bench_apply_wiener, s, n, &st);
    print_result("apply_wiener_e2e", n, &st);

    cheap_destroy(&s->ctx);
    fftw_free(s->input); fftw_free(s->weights); fftw_free(s->output);
    free(s);
}

/* =========================================================================
 * 2D / 3D core benchmarks
 * ========================================================================= */

typedef struct { cheap_ctx_2d ctx; double *input; double *weights; double *output; } apply_2d_state;
static void bench_apply_2d(void *p) {
    apply_2d_state *s = (apply_2d_state *)p;
    cheap_apply_2d(&s->ctx, s->input, s->weights, s->output);
}
static void bench_apply_2d_inplace(void *p) {
    apply_2d_state *s = (apply_2d_state *)p;
    memcpy(s->ctx.workspace, s->input, (size_t)s->ctx.n * sizeof(double));
    cheap_apply_inplace_2d(&s->ctx, s->weights);
}

typedef struct { cheap_ctx_2d ctx; double *input; double *output; } fwd_inv_2d_state;
static void bench_forward_2d(void *p) {
    fwd_inv_2d_state *s = (fwd_inv_2d_state *)p;
    cheap_forward_2d(&s->ctx, s->input);
}
static void bench_inverse_2d(void *p) {
    fwd_inv_2d_state *s = (fwd_inv_2d_state *)p;
    cheap_inverse_2d(&s->ctx, s->output);
}

static void run_core_benchmarks_2d(int nx, int ny)
{
    const int n = nx * ny;
    const double H = 0.7;

    /* apply_2d with KRR weights */
    {
        apply_2d_state *s = (apply_2d_state *)malloc(sizeof(*s));
        CHECK_RC(cheap_init_2d(&s->ctx, nx, ny, H, H));
        s->input   = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->weights = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->output  = (double *)fftw_malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; ++i) s->input[i] = sin(2.0 * M_PI * i / n) + 1.0;
        for (int k = 0; k < n; ++k) s->weights[k] = 1.0 / (s->ctx.lambda[k] + 1e-3);
        bench_stats st;
        run_bench(bench_apply_2d, s, n, &st);
        print_result("apply_krr_2d", n, &st);
        cheap_destroy_2d(&s->ctx);
        fftw_free(s->input); fftw_free(s->weights); fftw_free(s->output); free(s);
    }
    /* apply_2d_inplace */
    {
        apply_2d_state *s = (apply_2d_state *)malloc(sizeof(*s));
        CHECK_RC(cheap_init_2d(&s->ctx, nx, ny, H, H));
        s->input   = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->weights = (double *)fftw_malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; ++i) s->input[i] = sin(2.0 * M_PI * i / n) + 1.0;
        for (int k = 0; k < n; ++k) s->weights[k] = 1.0 / (s->ctx.lambda[k] + 1e-3);
        bench_stats st;
        run_bench(bench_apply_2d_inplace, s, n, &st);
        print_result("apply_krr_2d_inplace", n, &st);
        cheap_destroy_2d(&s->ctx);
        fftw_free(s->input); fftw_free(s->weights); free(s);
    }
    /* forward_2d */
    {
        fwd_inv_2d_state *s = (fwd_inv_2d_state *)malloc(sizeof(*s));
        CHECK_RC(cheap_init_2d(&s->ctx, nx, ny, H, H));
        s->input  = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->output = (double *)fftw_malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; ++i) s->input[i] = sin(2.0 * M_PI * i / n);
        bench_stats st;
        run_bench(bench_forward_2d, s, n, &st);
        print_result("forward_2d", n, &st);
        cheap_destroy_2d(&s->ctx);
        fftw_free(s->input); fftw_free(s->output); free(s);
    }
    /* inverse_2d */
    {
        fwd_inv_2d_state *s = (fwd_inv_2d_state *)malloc(sizeof(*s));
        CHECK_RC(cheap_init_2d(&s->ctx, nx, ny, H, H));
        s->input  = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->output = (double *)fftw_malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; ++i) s->input[i] = sin(2.0 * M_PI * i / n);
        cheap_forward_2d(&s->ctx, s->input);
        bench_stats st;
        run_bench(bench_inverse_2d, s, n, &st);
        print_result("inverse_2d", n, &st);
        cheap_destroy_2d(&s->ctx);
        fftw_free(s->input); fftw_free(s->output); free(s);
    }
}

typedef struct { cheap_ctx_3d ctx; double *input; double *weights; double *output; } apply_3d_state;
static void bench_apply_3d(void *p) {
    apply_3d_state *s = (apply_3d_state *)p;
    cheap_apply_3d(&s->ctx, s->input, s->weights, s->output);
}
static void bench_apply_3d_inplace(void *p) {
    apply_3d_state *s = (apply_3d_state *)p;
    memcpy(s->ctx.workspace, s->input, (size_t)s->ctx.n * sizeof(double));
    cheap_apply_inplace_3d(&s->ctx, s->weights);
}

typedef struct { cheap_ctx_3d ctx; double *input; double *output; } fwd_inv_3d_state;
static void bench_forward_3d(void *p) {
    fwd_inv_3d_state *s = (fwd_inv_3d_state *)p;
    cheap_forward_3d(&s->ctx, s->input);
}
static void bench_inverse_3d(void *p) {
    fwd_inv_3d_state *s = (fwd_inv_3d_state *)p;
    cheap_inverse_3d(&s->ctx, s->output);
}

static void run_core_benchmarks_3d(int nx, int ny, int nz)
{
    const int n = nx * ny * nz;
    const double H = 0.7;

    /* apply_3d with KRR weights */
    {
        apply_3d_state *s = (apply_3d_state *)malloc(sizeof(*s));
        CHECK_RC(cheap_init_3d(&s->ctx, nx, ny, nz, H, H, H));
        s->input   = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->weights = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->output  = (double *)fftw_malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; ++i) s->input[i] = sin(2.0 * M_PI * i / n) + 1.0;
        for (int k = 0; k < n; ++k) s->weights[k] = 1.0 / (s->ctx.lambda[k] + 1e-3);
        bench_stats st;
        run_bench(bench_apply_3d, s, n, &st);
        print_result("apply_krr_3d", n, &st);
        cheap_destroy_3d(&s->ctx);
        fftw_free(s->input); fftw_free(s->weights); fftw_free(s->output); free(s);
    }
    /* apply_3d_inplace */
    {
        apply_3d_state *s = (apply_3d_state *)malloc(sizeof(*s));
        CHECK_RC(cheap_init_3d(&s->ctx, nx, ny, nz, H, H, H));
        s->input   = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->weights = (double *)fftw_malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; ++i) s->input[i] = sin(2.0 * M_PI * i / n) + 1.0;
        for (int k = 0; k < n; ++k) s->weights[k] = 1.0 / (s->ctx.lambda[k] + 1e-3);
        bench_stats st;
        run_bench(bench_apply_3d_inplace, s, n, &st);
        print_result("apply_krr_3d_inplace", n, &st);
        cheap_destroy_3d(&s->ctx);
        fftw_free(s->input); fftw_free(s->weights); free(s);
    }
    /* forward_3d */
    {
        fwd_inv_3d_state *s = (fwd_inv_3d_state *)malloc(sizeof(*s));
        CHECK_RC(cheap_init_3d(&s->ctx, nx, ny, nz, H, H, H));
        s->input  = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->output = (double *)fftw_malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; ++i) s->input[i] = sin(2.0 * M_PI * i / n);
        bench_stats st;
        run_bench(bench_forward_3d, s, n, &st);
        print_result("forward_3d", n, &st);
        cheap_destroy_3d(&s->ctx);
        fftw_free(s->input); fftw_free(s->output); free(s);
    }
    /* inverse_3d */
    {
        fwd_inv_3d_state *s = (fwd_inv_3d_state *)malloc(sizeof(*s));
        CHECK_RC(cheap_init_3d(&s->ctx, nx, ny, nz, H, H, H));
        s->input  = (double *)fftw_malloc((size_t)n * sizeof(double));
        s->output = (double *)fftw_malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; ++i) s->input[i] = sin(2.0 * M_PI * i / n);
        cheap_forward_3d(&s->ctx, s->input);
        bench_stats st;
        run_bench(bench_inverse_3d, s, n, &st);
        print_result("inverse_3d", n, &st);
        cheap_destroy_3d(&s->ctx);
        fftw_free(s->input); fftw_free(s->output); free(s);
    }
}

/* =========================================================================
 * 2D / 3D weight constructor benchmarks
 * ========================================================================= */

typedef struct { int nx, ny; double* w; } weight_2d_state;
static void bench_weights_laplacian_2d(void *p) {
    weight_2d_state *s = (weight_2d_state *)p;
    cheap_weights_laplacian_2d(s->nx, s->ny, s->w);
}

static void run_weight_benchmarks_2d(int nx, int ny)
{
    weight_2d_state *s = (weight_2d_state *)malloc(sizeof(*s));
    s->nx = nx; s->ny = ny;
    s->w = (double *)fftw_malloc((size_t)(nx * ny) * sizeof(double));
    bench_stats st;
    run_bench(bench_weights_laplacian_2d, s, nx * ny, &st);
    print_result("wt_laplacian_2d", nx * ny, &st);
    fftw_free(s->w); free(s);
}

typedef struct { int nx, ny, nz; double* w; } weight_3d_state;
static void bench_weights_laplacian_3d(void *p) {
    weight_3d_state *s = (weight_3d_state *)p;
    cheap_weights_laplacian_3d(s->nx, s->ny, s->nz, s->w);
}

static void run_weight_benchmarks_3d(int nx, int ny, int nz)
{
    weight_3d_state *s = (weight_3d_state *)malloc(sizeof(*s));
    s->nx = nx; s->ny = ny; s->nz = nz;
    s->w = (double *)fftw_malloc((size_t)(nx * ny * nz) * sizeof(double));
    bench_stats st;
    run_bench(bench_weights_laplacian_3d, s, nx * ny * nz, &st);
    print_result("wt_laplacian_3d", nx * ny * nz, &st);
    fftw_free(s->w); free(s);
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(void)
{
    static const int sizes[] = {1024, 8192, 65536};
    static const int nsizes  = 3;

    print_header();

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

    /* 2D core */
    {
        int sizes_2d[][2] = {{64, 64}, {128, 128}, {256, 256}, {512, 512}};
        for (size_t i = 0; i < sizeof(sizes_2d)/sizeof(sizes_2d[0]); ++i)
            run_core_benchmarks_2d(sizes_2d[i][0], sizes_2d[i][1]);
    }

    /* 3D core */
    {
        int sizes_3d[][3] = {{16, 16, 16}, {32, 32, 32}, {64, 64, 64}};
        for (size_t i = 0; i < sizeof(sizes_3d)/sizeof(sizes_3d[0]); ++i)
            run_core_benchmarks_3d(sizes_3d[i][0], sizes_3d[i][1], sizes_3d[i][2]);
    }

    /* 2D / 3D weight constructors */
    run_weight_benchmarks_2d(128, 128);
    run_weight_benchmarks_3d(32, 32, 32);

    return 0;
}
