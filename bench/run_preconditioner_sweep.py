#!/usr/bin/env python3
"""Archive isolated DCT preconditioner/stage timings and selected plans."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from run_pcg_sweep import ROOT, capture, snapshot, utc, write_json
sys.path.insert(0, str(ROOT / 'tests'))
from test_bench_preconditioner import check_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fftw', type=Path, default=ROOT / 'build-jetson-gcc-FFTW3/cheap_bench_preconditioner')
    parser.add_argument('--notorious', type=Path, default=ROOT / 'build-jetson-gcc-NOTORIOUS/cheap_bench_preconditioner')
    args = parser.parse_args()
    binaries = {'FFTW3': args.fftw.resolve(), 'NOTORIOUS': args.notorious.resolve()}
    for binary in binaries.values():
        if not binary.is_file():
            parser.error(f'missing binary: {binary}')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    commands = [['uname', '-a'], ['lscpu'], ['/usr/bin/gcc', '--version'],
                ['cmake', '--version'], ['pkg-config', '--modversion', 'fftw3'],
                ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                ['git', '-C', str(ROOT), 'status', '--short'], ['cat', '/etc/nv_tegra_release']]
    write_json(out / 'platform.json', {'utc': utc(), 'commands': [capture(c) for c in commands]})
    patch = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--', 'CMakeLists.txt'])
    (out / 'cmake.patch.gz').write_bytes(gzip.compress(patch, mtime=0))
    sources = [ROOT / name for name in ('CMakeLists.txt', 'bench/bench_preconditioner.c',
               'bench/run_preconditioner_sweep.py', 'bench/run_pcg_sweep.py',
               'tests/test_bench_preconditioner.py')]
    sources += sorted((ROOT / 'include').glob('*')) + sorted((ROOT / 'src').glob('*'))
    write_json(out / 'source-hashes.json', {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources if p.is_file()})
    builds = {}
    for backend, binary in binaries.items():
        build = binary.parent
        entries = json.loads((build / 'compile_commands.json').read_text())
        builds[backend] = {
            'binary': str(binary), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            'compile_commands': [e for e in entries if Path(e['file']).name in
                                 ('bench_preconditioner.c', 'cheap_notorious_fft.c')],
            'link_command': (build / 'CMakeFiles/cheap_bench_preconditioner.dir/link.txt').read_text()}
        if backend == 'NOTORIOUS':
            dependency = build / '_deps/notorious_fft-src'
            builds[backend]['dependency_commit'] = subprocess.check_output(
                ['git', '-C', str(dependency), 'rev-parse', 'HEAD'], text=True).strip()
            header = (dependency / 'include/notorious_fft.h').read_text().splitlines()
            # Preserve source evidence for the measured runtime plan fields.
            excerpt = []
            for start, stop in ((4308, 4342), (4386, 4390), (4524, 4531), (5042, 5055), (5661, 5675)):
                excerpt.extend(f'{i+1}: {header[i]}' for i in range(start-1, stop))
                excerpt.append('')
            (out / 'notorious-path.txt').write_text('\n'.join(excerpt) + '\n')
    write_json(out / 'builds.json', builds)
    power = [['sudo', '-n', 'nvpmodel', '-q'], ['sudo', '-n', 'jetson_clocks', '--show']]
    write_json(out / 'power-before.json', {'utc': utc(), 'commands': [capture(c) for c in power]})
    records = []
    for n in (256, 257, 258, 511, 512, 513):
        for backend, binary in binaries.items():
            prefix = out / f'{backend}-{n}'
            command = [str(binary), str(n), str(prefix)]
            record = {'backend': backend, 'n': n, 'rho': .99, 'command': command, 'before': snapshot()}
            start = time.monotonic()
            with prefix.with_suffix('.csv').open('w') as stdout:
                try:
                    run = subprocess.run(command, stdout=stdout, stderr=subprocess.PIPE, text=True, timeout=120)
                    record.update(exit_status=run.returncode, stderr=run.stderr)
                except subprocess.TimeoutExpired as error:
                    record.update(exit_status=124, stderr=str(error))
            record.update(elapsed_seconds=time.monotonic()-start, after=snapshot())
            try:
                with prefix.with_suffix('.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                with Path(str(prefix) + '-samples.csv').open() as stream:
                    samples = list(csv.DictReader(stream))
                plans = Path(str(prefix) + '-plans.txt').read_text()
                assert record['exit_status'] == 0 and not record['stderr'], record['stderr']
                check_rows(rows, samples, plans, backend, n)
                record['validation'] = 'passed'
            except (AssertionError, OSError, KeyError, ValueError) as error:
                record.update(validation='failed', validation_error=repr(error))
            records.append(record)
            write_json(out / 'runs.json', records)
            print(f'{backend}-{n}: exit={record["exit_status"]}, validation={record["validation"]}, '
                  f'elapsed={record["elapsed_seconds"]:.3f}s', flush=True)
    write_json(out / 'power-after.json', {'utc': utc(), 'commands': [capture(c) for c in power]})
    return int(any(r['validation'] != 'passed' for r in records))


if __name__ == '__main__':
    sys.exit(main())
