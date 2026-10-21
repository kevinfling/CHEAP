#!/usr/bin/env python3
"""Check isolated preconditioner benchmark output against raw timings/plans."""
import csv
import io
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

OPERATIONS = ['preconditioner_apply', 'cheap_apply', 'validate_copy', 'dct2_execute',
              'weight_multiply', 'dct3_execute', 'scale_copy']


def check_rows(rows, samples, plans, backend, n):
    assert len(rows) == 7 and len(samples) == 287
    assert [r['operation'] for r in rows] == OPERATIONS
    assert f'backend={backend} n={n}' in plans
    assert 'DCT-II/REDFT10:' in plans and 'DCT-III/REDFT01:' in plans
    if backend == 'NOTORIOUS':
        direct = int(bool(n & (n - 1)))
        assert plans.count(f'direct_r2r={direct}') == 2
        assert plans.count(f'fft_present={1-direct}') == 2
    for row in rows:
        assert row['backend'] == backend and int(row['n']) == n
        assert row['result'] == '0' and row['planner'] == 'FFTW_ESTIMATE'
        assert row['repetitions'] == '40'
        group = [s for s in samples if s['operation'] == row['operation']]
        assert len(group) == 41
        assert [int(s['repetition']) for s in group] == list(range(-1, 40))
        for s in group:
            assert s['backend'] == backend and int(s['n']) == n
            assert float(s['rho']) == float(row['rho'])
            assert s['result'] == '0'
            assert s['phase'] == ('first' if s['repetition'] == '-1' else 'steady')
            assert math.isfinite(float(s['elapsed_ms'])) and float(s['elapsed_ms']) >= 0
            assert math.isfinite(float(s['relative_error']))
            assert 0 <= float(s['relative_error']) <= 1e-9
        times = sorted(float(s['elapsed_ms']) for s in group[1:])
        expected = {'first_ms': float(group[0]['elapsed_ms']),
                    'median_ms': statistics.median(times), 'p95_ms': times[37],
                    'iqr_ms': times[29] - times[9],
                    'max_relative_error': max(float(s['relative_error']) for s in group)}
        for key, value in expected.items():
            assert math.isclose(float(row[key]), value, rel_tol=1e-8,
                                abs_tol=1e-10 if key.endswith('_ms') else 1e-20), (key, row[key], value)
    assert len({r['setup_ms'] for r in rows}) == 1


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage: test_bench_preconditioner.py BINARY [BINARY]')
    for argument in sys.argv[1:]:
        binary = str(Path(argument).resolve())
        for args in ([], ['bad', 'x'], ['1', 'x'], ['4097', 'x'],
                     ['17', 'x', 'nan'], ['17', 'x', '1'],
                     ['17', 'x', '-.1'], ['17', 'x', '.99', 'extra']):
            run = subprocess.run([binary, *args], capture_output=True, text=True)
            assert run.returncode == 2 and 'usage:' in run.stderr
        with tempfile.TemporaryDirectory(prefix='cheap-isolated-check-') as directory:
            for n in (17, 64):
                prefix = Path(directory) / str(n)
                run = subprocess.run([binary, str(n), str(prefix)], capture_output=True, text=True)
                assert run.returncode == 0 and not run.stderr, run.stderr
                rows = list(csv.DictReader(io.StringIO(run.stdout)))
                with Path(str(prefix) + '-samples.csv').open() as stream:
                    samples = list(csv.DictReader(stream))
                check_rows(rows, samples, Path(str(prefix) + '-plans.txt').read_text(),
                           rows[0]['backend'], n)
            run = subprocess.run([binary, '17', str(Path(directory) / 'missing' / 'x')],
                                 capture_output=True, text=True)
            assert run.returncode == 2 and run.stderr and not run.stdout
        print(f'PASS: {binary}: CLI, stage reference checks, raw statistics and plan diagnostics')
