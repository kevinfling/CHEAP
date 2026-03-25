/*
 * bench_cheap_stats.c — Statistical benchmarking for CHEAP
 *
 * Uses FFTW_PATIENT planning and repeated measurements to report
 * mean, stddev, min, max, and confidence intervals.
 *
 * Build:
 *   gcc -std=c99 -O3 -march=native -D_POSIX_C_SOURCE=199309L \
 *       bench_cheap_stats.c -o bench_cheap_stats -lfftw3 -lm
 */

#include "cheap.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* =========================================================================
 * Configuration
 * ========================================================================= */
#define WARMUP_TRIALS       3       /* Initial warmup runs (discarded) */
#define MEASUREMENT_TRIALS  30      /* Trials for statistics */
#define MIN_BENCH_TIME_MS   100.0   /* Minimum benchmark duration per trial */

/* =========================================================================
 * Statistics structure
 * ========================================================================= */
typedef struct {
    double mean;
    double stddev;
    double min;
    double max;
    double median;
    double p5;      /* 5th percentile */
    double p95;     /* 95th percentile */
    double cv;      /* Coefficient of variation (stddev/mean) */
    int n;
} stats_t;

/* =========================================================================
 * Portable timing
 * ========================================================================= */
static inline uint64_t read_ticks(void)
{
#if defined(__aarch64__)
    uint64_t v;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#elif defined(__x86_64__)
    return cheap_rdtsc();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static double wall_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Get timer frequency (ticks per second) */
static double get_tick_frequency(void)
{
#if defined(__aarch64__)
    /* ARM generic timer typically runs at 1-50 MHz */
    uint64_t freq;
    __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(freq));
    return (double)freq;
#else
    /* x86: estimate via clock comparison */
    double t0 = wall_time();
    uint64_t c0 = read_ticks();
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 100000000; /* 100ms */
    nanosleep(&ts, NULL);
    double t1 = wall_time();
    uint64_t c1 = read_ticks();
    return (double)(c1 - c0) / (t1 - t0);
#endif
}

/* =========================================================================
 * Statistics computation
 * ========================================================================= */
static int compare_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static stats_t compute_stats(double *values, int n)
{
    stats_t s;
    s.n = n;

    /* Sort for min/max/median/percentiles */
    double *sorted = malloc(n * sizeof(double));
    memcpy(sorted, values, n * sizeof(double));
    qsort(sorted, n, sizeof(double), compare_double);

    s.min = sorted[0];
    s.max = sorted[n - 1];
    s.median = sorted[n / 2];
    s.p5 = sorted[(int)(n * 0.05)];
    s.p95 = sorted[(int)(n * 0.95)];

    /* Mean */
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += values[i];
    s.mean = sum / n;

    /* Stddev (sample) */
    double sq_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = values[i] - s.mean;
        sq_sum += d * d;
    }
    s.stddev = sqrt(sq_sum / (n - 1));
    s.cv = s.stddev / s.mean;

    free(sorted);
    return s;
}

/* =========================================================================
 * Benchmark infrastructure
 * ========================================================================= */
typedef void (*bench_func_t)(void *);

typedef void *(*setup_func_t)(int);
typedef void (*teardown_func_t)(void*);

typedef struct {
    const char *name;
    bench_func_t func;
    setup_func_t setup;
    teardown_func_t teardown;
} bench_def_t;

/* Apply (KRR weights) benchmark */
typedef struct { cheap_ctx ctx; double *input; double *weights; double *output; } apply_state_t;
static void *krr_setup(int n) {
    apply_state_t *s = calloc(1, sizeof(*s));
    cheap_init(&s->ctx, n, 0.7);
    s->input = fftw_malloc((size_t)n * sizeof(double));
    s->weights = fftw_malloc((size_t)n * sizeof(double));
    s->output = fftw_malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i) s->input[i] = sin(2.0 * M_PI * i / n) + 1.0;
    for (int k = 0; k < n; ++k) {
        double denom = s->ctx.lambda[k] + 1e-3;
        s->weights[k] = 1.0 / denom;
    }
    return s;
}
static void krr_bench(void *p) {
    apply_state_t *s = p;
    cheap_apply(&s->ctx, s->input, s->weights, s->output);
}
static void krr_teardown(void *p) {
    apply_state_t *s = p;
    cheap_destroy(&s->ctx);
    fftw_free(s->input); fftw_free(s->weights); fftw_free(s->output);
    free(s);
}

/* Apply (reparam weights) benchmark */
static void *reparam_setup(int n) {
    apply_state_t *s = calloc(1, sizeof(*s));
    cheap_init(&s->ctx, n, 0.7);
    s->input = fftw_malloc((size_t)n * sizeof(double));
    s->weights = s->ctx.sqrt_lambda;  /* borrow — don't free */
    s->output = fftw_malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i) s->input[i] = 1.0;
    return s;
}
static void reparam_bench(void *p) {
    apply_state_t *s = p;
    cheap_apply(&s->ctx, s->input, s->weights, s->output);
}
static void reparam_teardown(void *p) {
    apply_state_t *s = p;
    cheap_destroy(&s->ctx);
    fftw_free(s->input); fftw_free(s->output);
    /* weights is ctx.sqrt_lambda — already freed by cheap_destroy */
    free(s);
}

/* Sinkhorn benchmark */
typedef struct { cheap_ctx ctx; double *a; double *b; double *f; double *g; } sinkhorn_state_t;
static void *sinkhorn_setup(int n) {
    sinkhorn_state_t *s = calloc(1, sizeof(*s));
    cheap_init(&s->ctx, n, 0.6);
    s->a = fftw_malloc(n * sizeof(double));
    s->b = fftw_malloc(n * sizeof(double));
    s->f = fftw_malloc(n * sizeof(double));
    s->g = fftw_malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) s->a[i] = s->b[i] = 1.0 / (double)n;
    return s;
}
static void sinkhorn_bench(void *p) {
    sinkhorn_state_t *s = p;
    cheap_sinkhorn(&s->ctx, s->a, s->b, 0.5, 50, 1e-15, s->f, s->g);
}
static void sinkhorn_teardown(void *p) {
    sinkhorn_state_t *s = p;
    cheap_destroy(&s->ctx);
    fftw_free(s->a); fftw_free(s->b); fftw_free(s->f); fftw_free(s->g);
    free(s);
}

/* Toeplitz matvec benchmark */
typedef struct {
    cheap_ctx ctx;
    double *t;
    double *lambda;
    double *x;
    double *y;
} toep_state_t;
static void *toep_setup(int n) {
    toep_state_t *s = calloc(1, sizeof(*s));
    cheap_init(&s->ctx, n, 0.5);
    s->t = calloc(n, sizeof(double));
    s->lambda = fftw_malloc(n * sizeof(double));
    s->x = fftw_malloc(n * sizeof(double));
    s->y = fftw_malloc(n * sizeof(double));
    s->t[0] = 4.0; s->t[1] = -1.0;
    for (int i = 0; i < n; ++i) s->x[i] = sin(2.0 * M_PI * i / n) + 1.0;
    cheap_toeplitz_eigenvalues(&s->ctx, s->t, s->lambda);
    return s;
}
static void toep_matvec_bench(void *p) {
    toep_state_t *s = p;
    cheap_apply(&s->ctx, s->x, s->lambda, s->y);
}
static void toep_teardown(void *p) {
    toep_state_t *s = p;
    cheap_destroy(&s->ctx);
    free(s->t); fftw_free(s->lambda); fftw_free(s->x); fftw_free(s->y);
    free(s);
}

/* =========================================================================
 * Statistical benchmark runner
 * ========================================================================= */
static void run_statistical_benchmark(const char *name,
                                       void *(*setup)(int),
                                       void (*bench)(void*),
                                       void (*teardown)(void*),
                                       int n,
                                       double tick_freq)
{
    /* Warmup: FFTW_PATIENT planning happens here */
    fprintf(stderr, "  [%s n=%d] Warmup + planning...\n", name, n);
    void *state = setup(n);
    for (int i = 0; i < WARMUP_TRIALS; ++i) {
        bench(state);
    }

    /* Measurements */
    double times_ms[MEASUREMENT_TRIALS];
    uint64_t ticks_arr[MEASUREMENT_TRIALS];

    fprintf(stderr, "  [%s n=%d] Running %d trials...\n", name, n, MEASUREMENT_TRIALS);

    for (int trial = 0; trial < MEASUREMENT_TRIALS; ++trial) {
        /* Ensure minimum runtime for fast functions */
        int inner_iters = 1;
        double start_time = wall_time();
        uint64_t start_ticks = read_ticks();

        do {
            for (int i = 0; i < inner_iters; ++i) {
                bench(state);
            }
            double elapsed_ms = (wall_time() - start_time) * 1000.0;
            if (elapsed_ms >= MIN_BENCH_TIME_MS || inner_iters >= 100) {
                uint64_t end_ticks = read_ticks();
                times_ms[trial] = elapsed_ms / inner_iters;
                ticks_arr[trial] = (end_ticks - start_ticks) / (uint64_t)inner_iters;
                break;
            }
            inner_iters *= 2;
        } while (1);
    }

    teardown(state);

    /* Compute statistics */
    stats_t t_stats = compute_stats(times_ms, MEASUREMENT_TRIALS);
    stats_t c_stats;
    if (tick_freq > 0) {
        /* Convert ticks to comparable units */
        double tick_times_ms[MEASUREMENT_TRIALS];
        for (int i = 0; i < MEASUREMENT_TRIALS; ++i) {
            tick_times_ms[i] = (ticks_arr[i] / tick_freq) * 1000.0;
        }
        c_stats = compute_stats(tick_times_ms, MEASUREMENT_TRIALS);
    }

    /* Output results */
    printf("\n%-20s N=%-6d\n", name, n);
    printf("%-20s %12s %12s %12s %12s %12s\n",
           "Metric", "Mean", "StdDev", "Min", "Max", "CV%");
    printf("%-20s %12.6f %12.6f %12.6f %12.6f %12.2f\n",
           "Time (ms):", t_stats.mean, t_stats.stddev,
           t_stats.min, t_stats.max, t_stats.cv * 100.0);
    if (tick_freq > 0) {
        printf("%-20s %12.6f %12.6f %12.6f %12.6f %12.2f\n",
               "Clock (ms):", c_stats.mean, c_stats.stddev,
               c_stats.min, c_stats.max, c_stats.cv * 100.0);
    }
    printf("%-20s %12.6f %12.6f %12s %12s %12s\n",
           "Percentiles (5/50/95):", t_stats.p5, t_stats.median,
           "", "", "");
    printf("%-20s %12.6f\n", "P95:", t_stats.p95);

    /* Stability assessment */
    if (t_stats.cv < 0.05) {
        printf("  [STABLE] CV = %.2f%%\n", t_stats.cv * 100.0);
    } else if (t_stats.cv < 0.10) {
        printf("  [MODERATE] CV = %.2f%%\n", t_stats.cv * 100.0);
    } else {
        printf("  [NOISY] CV = %.2f%% - consider more trials\n", t_stats.cv * 100.0);
    }
}

/* =========================================================================
 * Main
 * ========================================================================= */
int main(void)
{
    printf("=== CHEAP Statistical Benchmarks ===\n");
    printf("Architecture: ");
#if defined(__aarch64__)
    printf("ARM64\n");
#elif defined(__x86_64__)
    printf("x86_64\n");
#else
    printf("Generic\n");
#endif

    double tick_freq = get_tick_frequency();
    printf("Timer frequency: %.3f MHz\n", tick_freq / 1e6);
    printf("Trials per benchmark: %d (warmup: %d)\n",
           MEASUREMENT_TRIALS, WARMUP_TRIALS);
    printf("\n");

    /* Test sizes */
    int sizes[] = {1024, 4096, 16384, 65536};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int si = 0; si < nsizes; ++si) {
        int n = sizes[si];
        printf("\n%s\n", "=================================================================");
        printf("N = %d\n", n);
        printf("%s\n", "=================================================================");

        run_statistical_benchmark("krr_solve",
                                   krr_setup,
                                   krr_bench,
                                   krr_teardown,
                                   n, tick_freq);

        run_statistical_benchmark("reparam",
                                   reparam_setup,
                                   reparam_bench,
                                   reparam_teardown,
                                   n, tick_freq);

        run_statistical_benchmark("sinkhorn_50",
                                   sinkhorn_setup,
                                   sinkhorn_bench,
                                   sinkhorn_teardown,
                                   n, tick_freq);

        run_statistical_benchmark("toeplitz_matvec",
                                   toep_setup,
                                   toep_matvec_bench,
                                   toep_teardown,
                                   n, tick_freq);
    }

    printf("\n=== Benchmarks Complete ===\n");
    return 0;
}
