#!/usr/bin/env python3
"""Build pinned SIMD FFTW and headless GNU Radio into build/deps. No system installs."""
import hashlib
import os
import pathlib
import platform
import subprocess
import sys
import tarfile
import urllib.request

if sys.version_info < (3, 12):
    raise SystemExit('Use Python 3.12+ (on this Mac: /opt/homebrew/bin/python3 scripts/bootstrap.py)')

root = pathlib.Path(__file__).resolve().parents[1]
vendor = root / 'build/vendor'
prefix = root / 'build/deps'
vendor.mkdir(parents=True, exist_ok=True)
jobs = str(min(8, os.cpu_count() or 2))
environment = os.environ.copy()
environment['PKG_CONFIG_PATH'] = str(prefix / 'lib/pkgconfig') + os.pathsep + environment.get('PKG_CONFIG_PATH', '')
environment['FFTW3_DIR'] = str(prefix)

def fetch(name, url, checksum, directory):
    archive = vendor / (name + '.tar.gz')
    if not archive.exists():
        print('Downloading', url, flush=True)
        urllib.request.urlretrieve(url, archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != checksum:
        raise RuntimeError('Checksum mismatch: ' + str(archive))
    target = vendor / directory
    if not target.exists():
        with tarfile.open(archive) as src:
            src.extractall(vendor, filter='data')
    return target

def build(source, name, options):
    directory = vendor / (name + '-build')
    commands = [
        ['cmake', '-S', str(source), '-B', str(directory), '-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_INSTALL_PREFIX=' + str(prefix), '-DCMAKE_PREFIX_PATH=' + str(prefix) + ';/opt/homebrew'] + options,
        ['cmake', '--build', str(directory), '-j', jobs],
        ['cmake', '--install', str(directory)],
    ]
    for command in commands:
        subprocess.run(command, check=True, env=environment)

# FFTW's CMake build does not expose NEON. Use its supported configure build;
# the installed system package may otherwise provide only scalar codelets.
fftw = fetch('fftw-3.3.11', 'https://www.fftw.org/fftw-3.3.11.tar.gz',
    '5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1', 'fftw-3.3.11')
fftw_build = vendor / 'fftw-build'
fftw_build.mkdir(exist_ok=True)
simd = ['--enable-neon'] if platform.machine().lower() in ('arm64', 'aarch64') else []
if platform.machine().lower() in ('x86_64', 'amd64'):
    simd = ['--enable-sse2', '--enable-avx2']  # FFTW checks CPU support at runtime.
for command in [
    [str(fftw / 'configure'), '--prefix=' + str(prefix), '--enable-float',
     '--enable-shared', '--enable-threads', '--disable-static', '--disable-fortran',
     '--disable-doc', 'CFLAGS=-O3'] + simd,
    ['make', '-j', jobs], ['make', 'install'],
]:
    subprocess.run(command, cwd=fftw_build, check=True, env=environment)

spdlog = fetch('spdlog', 'https://github.com/gabime/spdlog/archive/refs/tags/v1.15.3.tar.gz',
    '15a04e69c222eb6c01094b5c7ff8a249b36bb22788d72519646fb85feb267e67', 'spdlog-1.15.3')
build(spdlog, 'spdlog', ['-DSPDLOG_BUILD_SHARED=ON', '-DSPDLOG_BUILD_EXAMPLE=OFF'])
gnuradio = fetch('gnuradio', 'https://github.com/gnuradio/gnuradio/archive/refs/tags/v3.10.12.0.tar.gz',
    'fe78ad9f74c8ebf93d5c8ad6fa2c13236af330f3c67149d91a0647b3dc6f3958', 'gnuradio-3.10.12.0')
build(gnuradio, 'gnuradio', ['-DENABLE_DEFAULT=OFF', '-DENABLE_GNURADIO_RUNTIME=ON',
    '-UFFTW3f_*', '-U*PC_FFTW3f*',
    '-DENABLE_GR_BLOCKS=ON', '-DENABLE_GR_FFT=ON', '-DENABLE_GR_FILTER=ON',
    '-DENABLE_GR_ANALOG=ON', '-DENABLE_GR_DIGITAL=ON', '-DENABLE_GR_DTV=ON',
    '-DENABLE_GR_FEC=ON', '-DENABLE_PYTHON=OFF', '-DENABLE_EXAMPLES=OFF',
    '-DENABLE_TESTING=OFF', '-DENABLE_GRC=OFF', '-DENABLE_PERFORMANCE_COUNTERS=OFF'])
print('Installed project-local dependencies to', prefix)
