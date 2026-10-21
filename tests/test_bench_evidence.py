#!/usr/bin/env python3
"""Check harness CLI, failure reporting and statistics against raw solve logs.

Usage: python3 tests/test_bench_evidence.py <binary> [<other-backend-binary>]
No timing or iteration-speedup assertions; numerical library tests are separate.
"""
import csv
import io
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


def check_rows(rows, samples, n, rho):
    """Also used by the sweep runner to validate each archived result."""
    assert len(samples) == 82, len(samples)
    bnorm = math.sqrt(sum((math.sin(0.1 * i) + 0.5) ** 2 for i in range(n)))
    solve_rows = [r for r in rows if r['backend'] in ('pcg', 'dct_pcg')]
    assert len(solve_rows) == 2
    for row in solve_rows:
        group = [s for s in samples if s['operator'] == row['operator']]
        assert len(group) == 41
        assert [int(s['repetition']) for s in group] == list(range(-1, 40))
        assert group[0]['phase'] == 'first'
        assert all(s['phase'] == 'steady' for s in group[1:])
        assert row['result'] == row['first_result'] == row['failed_solves'] == '0'
        assert row['residual_valid'] == '1'
        assert row['planner'] == 'FFTW_ESTIMATE'
        for sample in group:
            assert sample['reason'] == '0' and sample['accepted'] == '1'
            assert sample['residual_valid'] == '1'
            assert sample['configured_backend'] == row['configured_backend']
            assert int(sample['n']) == int(row['n']) == n
            assert float(sample['rho']) == float(row['rho']) == rho
            assert math.isfinite(float(sample['elapsed_ms']))
            assert float(sample['elapsed_ms']) >= 0
            assert 0 <= float(sample['true_residual']) <= 1e-10 + 1e-8 * bnorm
            for key in ('relative_residual', 'independent_relative_residual'):
                value = float(sample[key])
                assert math.isfinite(value) and 0 <= value <= 1e-8 + 1e-10 / bnorm
            assert math.isclose(float(sample['relative_residual']),
                                float(sample['true_residual']) / bnorm, rel_tol=1e-12,
                                abs_tol=1e-16)
            iterations = int(sample['iterations'])
            assert int(sample['operator_calls']) >= iterations + 2
            assert int(sample['preconditioner_calls']) == (
                iterations if row['backend'] == 'dct_pcg' else 0)
        times = sorted(float(s['elapsed_ms']) for s in group[1:])
        expected = {
            'first_ms': float(group[0]['elapsed_ms']),
            'median_ms': statistics.median(times),
            'p95_ms': times[37],
            'iqr_ms': times[29] - times[9],
            'true_residual': float(group[0]['true_residual']),
            'relative_residual': float(group[0]['relative_residual']),
            'independent_relative_residual': float(group[0]['independent_relative_residual']),
            'max_relative_residual': max(float(s['relative_residual']) for s in group),
            'max_independent_relative_residual': max(
                float(s['independent_relative_residual']) for s in group),
        }
        for key, value in expected.items():
            assert math.isclose(float(row[key]), value, rel_tol=5e-8, abs_tol=1e-7
                                if key.endswith('_ms') else 1e-18), (key, row[key], value)
        for key in ('iterations', 'operator_calls', 'preconditioner_calls'):
            assert row[key] == group[-1][key]
        assert int(row['min_iterations']) == min(int(s['iterations']) for s in group)
        assert int(row['max_iterations']) == max(int(s['iterations']) for s in group)
        assert math.isclose(float(row['setup_ms']), float(row['common_setup_ms']) +
                            float(row['preconditioner_setup_ms']), rel_tol=5e-8, abs_tol=1e-7)
        assert float(row['preconditioner_setup_ms']) >= 0
    assert solve_rows[0]['common_setup_ms'] == solve_rows[1]['common_setup_ms']
    assert float(solve_rows[0]['preconditioner_setup_ms']) == 0


def check_binary(binary):
    for args in (['bad'], ['2x'], ['1'], ['65537'], ['17', 'nan'],
                 ['17', 'inf'], ['17', '-0.1'], ['17', '1'], ['17', '.9x'],
                 ['17', '.9', 'bad-mode'], ['17', '.9', 'pcg', 'a', 'extra']):
        run = subprocess.run([binary, *args], capture_output=True, text=True)
        assert run.returncode == 2 and 'usage:' in run.stderr, (args, run.stderr)
    with tempfile.TemporaryDirectory(prefix='cheap-bench-check-') as directory:
        log = Path(directory) / 'samples.csv'
        for n, rho, mode, count, status in ((2, 0.0, 'pcg', 2, 0),
                                           (17, .99, 'pcg', 2, 0),
                                           (17, .3, 'all', 4, 0),
                                           (2, .99, 'all', 3, 1)):
            run = subprocess.run([binary, str(n), str(rho), mode, str(log)],
                                 capture_output=True, text=True)
            assert run.returncode == status, run.stderr
            if status:
                assert 'circulant sampler setup' in run.stderr
            else:
                assert not run.stderr, run.stderr
            rows = list(csv.DictReader(io.StringIO(run.stdout)))
            assert len(rows) == count
            check_rows(rows, list(csv.DictReader(log.open())), n, rho)
        run = subprocess.run([binary, '17', '.9', 'pcg',
                              str(Path(directory) / 'missing' / 'samples.csv')],
                             capture_output=True, text=True)
        assert run.returncode == 2 and run.stderr and not run.stdout
    print(f'PASS: {binary}: CLI, failed embedding retention, all/pcg modes, '
          'per-solve residuals, call counts, setup accounting and raw statistics')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for argument in sys.argv[1:]:
        check_binary(str(Path(argument).resolve()))
