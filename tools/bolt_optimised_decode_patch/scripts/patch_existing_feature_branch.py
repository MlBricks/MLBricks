#!/usr/bin/env python3
"""Patch an EXISTING checked-out BOLT feature branch locally (never GitHub).

This runner adds the previously GPU-validated split kernel if absent, then
applies the per-width dispatch + AMP wrapper and executes CPU-only unit tests.
The target must not be on main/master. CUDA compilation and GPU parity are
separate, required release gates.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
SPLIT_PATCH=PACKAGE/'experimental_split_context_cuda.patch'


def cmd(args, *, cwd=None, env=None):
    print('RUN', ' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),check=True,cwd=cwd,env=env)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True,type=Path)
    args=parser.parse_args()
    repo=args.repo.expanduser().resolve()
    if not (repo/'.git').exists():
        raise RuntimeError('Pass an actual Git repository feature-branch checkout, not a zip or site-packages install')
    branch=subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()
    if not branch or branch in ('main','master'):
        raise RuntimeError('Refusing to patch main/master or detached HEAD; checkout a new feature branch')
    native=repo/'mlbricks/bolt/native_backend.py'
    if not native.exists():raise FileNotFoundError(native)
    if 'decode_append_split' not in native.read_text(encoding='utf-8'):
        cmd(['git','apply','--check',str(SPLIT_PATCH)],cwd=repo)
        cmd(['git','apply',str(SPLIT_PATCH)],cwd=repo)
    cmd([sys.executable,str(HERE/'apply_bolt_dispatch_patch.py'),'--repo',str(repo)])
    env=os.environ.copy()
    env['MLBRICKS_REPO']=str(repo)
    cmd([sys.executable,'-m','unittest','discover','-s',str(PACKAGE/'tests'),'-v'],cwd=repo,env=env)
    cmd(['git','diff','--check'],cwd=repo)
    print('\nLOCAL PATCH COMPLETE. No commit/push/merge performed.')
    print('Review using: git diff -- mlbricks/bolt/native_backend.py mlbricks/bolt/presets.py')
    print('GPU compile, native V2 Strict numerical parity, trained generation, CUDA Graph and benchmark still required.')


if __name__=='__main__':main()
