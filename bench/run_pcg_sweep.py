#!/usr/bin/env python3
"""Run sequential AR(1) CG/DCT-PCG pairs and archive measurements/provenance.

Build both cheap_bench_evidence binaries first. Use a new --output directory.
Power/clock commands only query the prepared Jetson; no settings are changed.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'tests'))
from test_bench_evidence import check_rows


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def capture(command):
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {'command': command, 'exit_status': run.returncode,
                'stdout': run.stdout, 'stderr': run.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'command': command, 'error': str(error)}


def snapshot():
    result = {'utc': utc(), 'thermal': {}, 'cpufreq': {}}

    def read(path):
        # os.read reports EAGAIN correctly for inactive Jetson CV sensors.
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                return os.read(fd, 4096).decode().strip()
            finally:
                os.close(fd)
        except OSError as error:
            return {'unavailable': str(error)}

    for zone in sorted(Path('/sys/class/thermal').glob('thermal_zone*')):
        result['thermal'][zone.name] = {'type': read(zone / 'type'),
                                        'millidegrees_c': read(zone / 'temp')}
    for policy in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*')):
        result['cpufreq'][policy.name] = {
            name: read(policy / name) for name in
            ('scaling_governor', 'scaling_min_freq', 'scaling_max_freq', 'scaling_cur_freq')}
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fftw', type=Path,
                        default=ROOT / 'build-jetson-gcc-FFTW3/cheap_bench_evidence')
    parser.add_argument('--notorious', type=Path,
                        default=ROOT / 'build-jetson-gcc-NOTORIOUS/cheap_bench_evidence')
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
                ['git', '-C', str(ROOT), 'status', '--short'],
                ['cat', '/etc/nv_tegra_release']]
    write_json(out / 'platform.json', {'utc': utc(), 'commands': [capture(c) for c in commands]})
    patch = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--', 'bench/bench_matched.c'])
    (out / 'harness.patch').write_bytes(patch)
    files = [ROOT / 'bench/bench_matched.c', Path(__file__).resolve(),
             ROOT / 'tests/test_bench_evidence.py', ROOT / 'CMakeLists.txt',
             *sorted((ROOT / 'include').glob('*')), *sorted((ROOT / 'src').glob('*'))]
    write_json(out / 'source-hashes.json', {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files if p.is_file()})
    builds = {}
    for backend, binary in binaries.items():
        directory = binary.parent
        entries = json.loads((directory / 'compile_commands.json').read_text())
        builds[backend] = {
            'binary': str(binary), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            'compile_commands': [e for e in entries if Path(e['file']).name in
                                 ('bench_matched.c', 'cheap_notorious_fft.c')],
            'link_command': (directory / 'CMakeFiles/cheap_bench_evidence.dir/link.txt').read_text(),
        }
        if backend == 'NOTORIOUS':
            builds[backend]['dependency_commit'] = subprocess.check_output(
                ['git', '-C', str(directory / '_deps/notorious_fft-src'), 'rev-parse', 'HEAD'],
                text=True).strip()
    write_json(out / 'builds.json', builds)
    power = [['sudo', '-n', 'nvpmodel', '-q'], ['sudo', '-n', 'jetson_clocks', '--show']]
    write_json(out / 'power-before.json', {'utc': utc(), 'commands': [capture(c) for c in power]})
    records = []
    for rho in (.3, .9, .99):
        for n in (17, 64, 257, 512, 1024, 4096):
            for backend, binary in binaries.items():
                name = f'{backend}-rho{rho:g}-{n}'
                sample_path = out / f'{name}-samples.csv'
                command = [str(binary), str(n), str(rho), 'pcg', str(sample_path)]
                record = {'backend': backend, 'n': n, 'rho': rho,
                          'command': command, 'before': snapshot()}
                start = time.monotonic()
                with (out / f'{name}.csv').open('w') as stdout:
                    try:
                        run = subprocess.run(command, stdout=stdout, stderr=subprocess.PIPE,
                                             text=True, timeout=120)
                        record.update(exit_status=run.returncode, stderr=run.stderr)
                    except subprocess.TimeoutExpired as error:
                        record.update(exit_status=124, stderr=str(error))
                record.update(elapsed_seconds=time.monotonic() - start, after=snapshot())
                try:
                    with (out / f'{name}.csv').open() as stream:
                        rows = list(csv.DictReader(stream))
                    with sample_path.open() as stream:
                        samples = list(csv.DictReader(stream))
                    assert record['exit_status'] == 0 and not record['stderr'], record['stderr']
                    assert len(rows) == 2
                    check_rows(rows, samples, n, rho)
                    record['validation'] = 'passed'
                except (AssertionError, OSError, KeyError, ValueError) as error:
                    record['validation'] = 'failed'
                    record['validation_error'] = repr(error)
                records.append(record)
                write_json(out / 'runs.json', records)
                print(f'{name}: exit={record["exit_status"]}, '
                      f'validation={record["validation"]}, '
                      f'elapsed={record["elapsed_seconds"]:.3f}s', flush=True)
    write_json(out / 'power-after.json', {'utc': utc(), 'commands': [capture(c) for c in power]})
    return int(any(r['validation'] != 'passed' for r in records))


if __name__ == '__main__':
    sys.exit(main())
