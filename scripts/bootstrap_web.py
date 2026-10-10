#!/usr/bin/env python3
"""Install the pinned WebAssembly build tools and local viewer server; no global installs."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

if sys.version_info < (3, 12):
    raise SystemExit('Use Python 3.12+ (on this Mac: /opt/homebrew/bin/python3)')
root=Path(__file__).resolve().parents[1]
tools=root/'build/toolchains'
tools.mkdir(parents=True,exist_ok=True)
archive=tools/'emsdk.tar.gz'
if not archive.exists():
    urllib.request.urlretrieve('https://github.com/emscripten-core/emsdk/archive/refs/tags/4.0.23.tar.gz',archive)
if hashlib.sha256(archive.read_bytes()).hexdigest()!='a91a4c1f42dbb0345faac093161e27d43e9b6964840d8c8d80976ab8d3eaf2d3':
    raise SystemExit('Emscripten SDK archive checksum mismatch')
sdk=tools/'emsdk-4.0.23'
if not sdk.exists():
    with tarfile.open(archive) as src:src.extractall(tools,filter='data')
for action in ('install','activate'):
    subprocess.run([sys.executable,str(sdk/'emsdk.py'),action,'4.0.23'],check=True,cwd=root)
subprocess.run([sys.executable,str(sdk/'upstream/emscripten/emcmake'),'cmake','-S','web','-B','build/web','-DCMAKE_BUILD_TYPE=Release'],check=True,cwd=root)
subprocess.run(['cmake','--build','build/web','-j',str(min(8,os.cpu_count() or 2))],check=True,cwd=root)
venv=root/'build/web-venv'
if not venv.exists():subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
subprocess.run([str(venv/'bin/python'),'-m','pip','install','-r','web/requirements.txt'],check=True,cwd=root)
print('Run: build/web-venv/bin/python web/serve.py YOUR_OUTPUT_DIRECTORY')
