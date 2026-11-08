#!/usr/bin/env python3
"""Apply the parity-preserving BOLT cached-decode dispatch & dtype patch.

Target: MLBricks feature branch with experimental `decode_append_split` present.
The patch fails closed if the required source anchors are absent, and is idempotent.
No GitHub actions, CUDA compilation or pip installation are performed here.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess
import sys

PATCH_MARKER = "# MLBRICKS_BOLT_WIDTH_DISPATCH_V1"

DISPATCH_SRC = '''# MLBRICKS_BOLT_WIDTH_DISPATCH_V1
# Conservative experimental policy from two-GPU synthetic + Wikipedia runs.
# These widths are PER HEAD latent dimensions, not total attention/AVQ width.
_BOLT_SPLIT_THRESHOLDS = {16: 1536, 32: 896}


def split_read_threshold(latent_dim: int) -> int:
    """Resolve native split-reader crossover; explicit env override wins.

    MLBRICKS_BOLT_SPLIT_MIN_TOKENS='auto' or unset: measured per-width policy.
    Any positive integer: manual global threshold for controlled benchmarking.
    Unknown dimensions keep the old 8192-token conservative threshold.
    """
    latent_dim = int(latent_dim)
    setting = os.getenv('MLBRICKS_BOLT_SPLIT_MIN_TOKENS', 'auto').strip().lower()
    if setting in ('auto', ''):
        return _BOLT_SPLIT_THRESHOLDS.get(latent_dim, 8192)
    try:
        n = int(setting)
    except ValueError as exc:
        raise ValueError('MLBRICKS_BOLT_SPLIT_MIN_TOKENS must be auto or a positive integer') from exc
    if n <= 0:
        raise ValueError('MLBRICKS_BOLT_SPLIT_MIN_TOKENS must be positive')
    return n


def decode_append(q, c_now, cache, scratch, position, scale):
    """Decode with existing native kernels; optionally use per-width split read.

    Default remains LEGACY when MLBRICKS_BOLT_SPLIT_CONTEXT_READ is unset/0.
    With opt-in 1, R=16 -> split from 1536; R=32 -> split from 896.
    Does not alter AVQ recurrence, CUDA kernels, cache dtype, or public API.
    """
    global _SPLIT_READ_CALLS
    ext = get_extension()
    position = int(position)
    scale = float(scale)
    enabled = os.getenv('MLBRICKS_BOLT_SPLIT_CONTEXT_READ', '0').strip().lower()
    if enabled in ('1', 'true', 'yes', 'on'):
        # Q shape in the native Direct-Q reader is [B, H, 1, R].
        # R is the 16/32 cache latent width, NOT the AVQ state width.
        if q.ndim != 4 or q.shape[-2] != 1:
            raise ValueError('split attention requires query [batch, heads, 1, latent_dim]')
        threshold = split_read_threshold(q.shape[-1])
        if position + 1 >= threshold:
            if ext is None or not hasattr(ext, 'decode_append_split'):
                raise RuntimeError('Native split-context BOLT CUDA extension unavailable')
            try:
                chunk = int(os.getenv('MLBRICKS_BOLT_SPLIT_CHUNK_TOKENS', '512'))
            except ValueError as exc:
                raise ValueError('MLBRICKS_BOLT_SPLIT_CHUNK_TOKENS must be a positive integer') from exc
            if chunk <= 0:
                raise ValueError('MLBRICKS_BOLT_SPLIT_CHUNK_TOKENS must be positive')
            # Backend already checks capacity, cache type, and supported R.
            result = ext.decode_append_split(q.contiguous(), c_now.contiguous(),
                                             cache, scratch, position, scale, chunk)
            _SPLIT_READ_CALLS += 1
            return result
    if ext is None or not hasattr(ext, 'decode_append'):
        raise RuntimeError('Native BOLT decode extension unavailable')
    return ext.decode_append(q.contiguous(), c_now.contiguous(),
                             cache, scratch, position, scale)
'''


def locate_native(text: str) -> str:
    """Require the opt-in CUDA split implementation to be present."""
    if PATCH_MARKER in text:
        return text
    if '_SPLIT_READ_CALLS' not in text or 'decode_append_split' not in text:
        raise RuntimeError('Missing split-context patch in native_backend.py. Apply the validated split-context patch first.')
    begin = text.find('def decode_append(q,c_now,cache,scratch,position,scale):')
    if begin < 0:
        begin = text.find('def decode_append(q, c_now, cache, scratch, position, scale):')
    if begin < 0:
        raise RuntimeError('Cannot locate native_backend.decode_append; inspect current branch before patching')
    end = text.find('\n__all__', begin)
    if end < 0 or 'decode_append_split' not in text[begin:end]:
        raise RuntimeError('Unexpected native backend layout; patch aborted')
    return text[:begin] + DISPATCH_SRC + '\n' + text[end:]


def locate_presets(text: str) -> str:
    """Wrap inference methods in FP16 AMP without rounding FP32 residual inputs.

    This matches the successful Wikipedia notebook's external FP16 autocast.
    Unlike converting `o` to FP16, AMP preserves the FP32 residual and the
    native V2 Strict FP32 state update, only casting supported GEMM inputs.
    """
    if '# MLBRICKS_BOLT_INFERENCE_AMP_V2' in text:
        return text
    import re
    patterns = (
        (r'(?m)^    def step\(self,\s*o,\s*state(?:,\s*\*,[^)]*)?\):',
         'self.project.weight', 'o'),
        (r'(?m)^    def decode_step\(self,\s*x,\s*cache\):',
         'self.out_proj.weight', 'x'),
        (r'(?m)^    def decode_with_cache\(self,\s*x,\s*cache[^\n]*\):',
         'self.out_proj.weight', 'x'),
    )
    for signature, weight, input_name in patterns:
        match = re.search(signature, text)
        if match is None:
            raise RuntimeError('Unsupported BOLT cached decoder signature: '+signature)
        first_line_end = text.find('\n', match.end())
        if first_line_end < 0:raise RuntimeError('Malformed decoder source')
        # The body ends at the next method/decorator, or at the next class.
        next_method = re.search(r'(?m)^(?: {4}(?=\S)|class )',text[first_line_end+1:])
        if next_method is None:raise RuntimeError('Cannot locate end of decoder method')
        body_end = first_line_end+1+next_method.start()
        body = text[first_line_end+1:body_end]
        if not body.strip():raise RuntimeError('Empty decoder body')
        if any(line and not line.startswith('        ') for line in body.splitlines()):
            raise RuntimeError('Unexpected indentation in decoder method')
        scope = ("        # MLBRICKS_BOLT_INFERENCE_AMP_V2\n"
                 "        # FP16 projection GEMMs only; FP32 V2 Strict state/residual unchanged.\n"
                 f"        with torch.autocast(device_type={input_name}.device.type, "
                 "dtype=torch.float16,\n"
                 f"                            enabled=({weight}.dtype == torch.float16 "
                 f"and {input_name}.device.type in ('cuda','cpu'))):\n")
        # Only re-indent existing non-empty body lines; trailing blank lines stay blank.
        indented = ''.join(('    '+line if line.strip() else line)
                           for line in body.splitlines(keepends=True))
        text = text[:first_line_end+1]+scope+indented+text[body_end:]
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True,help='Path to the checked-out MLBricks repository')
    parser.add_argument('--check',action='store_true',help='Validate compatibility without writing files')
    args=parser.parse_args()
    root=args.repo.expanduser().resolve()
    if (root/'.git').exists():
        branch=subprocess.run(['git','-C',str(root),'branch','--show-current'],
                              text=True,capture_output=True,check=True).stdout.strip()
        if branch in ('main','master') and not args.check:
            raise RuntimeError('Refusing to patch main/master. Create a feature branch first.')
    native=root/'mlbricks/bolt/native_backend.py'
    presets=root/'mlbricks/bolt/presets.py'
    for f in (native,presets):
        if not f.is_file():raise FileNotFoundError(str(f))
    n0=native.read_text(encoding='utf-8')
    p0=presets.read_text(encoding='utf-8')
    n1=locate_native(n0)
    p1=locate_presets(p0)
    ast.parse(n1,filename=str(native))
    ast.parse(p1,filename=str(presets))
    if args.check:
        print('CHECK PASS: patched source syntax/anchors',root)
        return 0
    for path,old,new in ((native,n0,n1),(presets,p0,p1)):
        if new!=old:
            path.write_text(new,encoding='utf-8')
            print('PATCHED',path)
        else:print('ALREADY PATCHED',path)
    return 0

if __name__=='__main__':
    sys.exit(main())
