"""CPU-only regression tests for opt-in BOLT native dispatch and dtype safety.

Usage:
  MLBRICKS_REPO=/path/to/MLBricks python -m unittest discover -s tests -v

The native extension is mocked; REAL GPU parity must be separately qualified.
"""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

REPO = Path(os.environ.get('MLBRICKS_REPO', Path(__file__).resolve().parents[2])).resolve()
NATIVE = REPO / 'mlbricks/bolt/native_backend.py'

class FakeExtension:
    def __init__(self):
        self.calls=[]
    def decode_append(self,q,c,cache,scratch,position,scale):
        self.calls.append(('legacy',position))
        return torch.ones_like(q)
    def decode_append_split(self,q,c,cache,scratch,position,scale,chunk):
        self.calls.append(('split',position,chunk))
        return torch.ones_like(q)


def load_native():
    assert NATIVE.is_file(), f'Could not find native_backend.py: {NATIVE}'
    spec=importlib.util.spec_from_file_location('bolt_backend_dispatch_test',NATIVE)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


class TestBOLTWidthPolicy(unittest.TestCase):
    def setUp(self):
        self.backend=load_native()
        self.ext=FakeExtension()
        self.ext_patch=patch.object(self.backend,'get_extension',return_value=self.ext)
        self.ext_patch.start()
        self.addCleanup(self.ext_patch.stop)
        self.env_patch=patch.dict(os.environ, {
            'MLBRICKS_BOLT_SPLIT_CONTEXT_READ':'1',
            'MLBRICKS_BOLT_SPLIT_MIN_TOKENS':'auto',
            'MLBRICKS_BOLT_SPLIT_CHUNK_TOKENS':'512',
        })
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def call(self,width,position):
        q=torch.zeros(1,2,1,width,dtype=torch.float32)
        c=torch.zeros(1,2,1,width,dtype=torch.float16)
        cache=torch.empty(1,2,max(position+2,1),width,dtype=torch.float16)
        scratch=torch.empty(1,2,max(position+2,1),dtype=torch.float32)
        return self.backend.decode_append(q,c,cache,scratch,position,0.1)

    def test_16_boundary(self):
        self.assertEqual(self.backend.split_read_threshold(16),1536)
        self.call(16,1534)
        self.call(16,1535)
        self.assertEqual([c[0] for c in self.ext.calls],['legacy','split'])

    def test_32_boundary(self):
        self.assertEqual(self.backend.split_read_threshold(32),896)
        self.call(32,894)
        self.call(32,895)
        self.assertEqual([c[0] for c in self.ext.calls],['legacy','split'])

    def test_short_context_default_legacy(self):
        os.environ['MLBRICKS_BOLT_SPLIT_CONTEXT_READ']='0'
        self.call(16,20000)
        self.assertEqual(self.ext.calls[-1][0],'legacy')

    def test_unknown_width_keeps_conservative_threshold(self):
        self.assertEqual(self.backend.split_read_threshold(64),8192)

    def test_explicit_override(self):
        os.environ['MLBRICKS_BOLT_SPLIT_MIN_TOKENS']='4096'
        self.call(16,4094)
        self.call(16,4095)
        self.assertEqual([c[0] for c in self.ext.calls],['legacy','split'])

    def test_invalid_threshold_fails_closed(self):
        os.environ['MLBRICKS_BOLT_SPLIT_MIN_TOKENS']='broken'
        with self.assertRaises(ValueError):self.call(16,1000)
        os.environ['MLBRICKS_BOLT_SPLIT_MIN_TOKENS']='-1'
        with self.assertRaises(ValueError):self.call(16,1000)

    def test_invalid_chunk_fails_closed(self):
        os.environ['MLBRICKS_BOLT_SPLIT_CHUNK_TOKENS']='0'
        with self.assertRaises(ValueError):self.call(16,1535)

    def test_split_counter_tracks_actual_kernel(self):
        before=self.backend.split_read_calls()
        self.call(16,1534)
        self.call(16,1535)
        self.assertEqual(self.backend.split_read_calls()-before,1)


class TestBOLTInferenceDtype(unittest.TestCase):
    def test_avq_projection_accepts_native_fp32_with_fp16_weights(self):
        # CPU stand-in for the CUDA reader returning FP32 O. No GPU kernels involved.
        sys.path.insert(0,str(REPO))
        from mlbricks.bolt.presets import BoltStateMemory
        layer=BoltStateMemory(input_dim=32,width=32,device='cpu',backend='pytorch',
                              return_map='identity').eval().half()
        state=torch.zeros(1,32,dtype=torch.float32)
        raw=torch.randn(1,1,32,dtype=torch.float32)
        with torch.no_grad():
            result,updated=layer.step(raw,state)
        self.assertEqual(result.dtype,torch.float32)  # FP32 residual retained
        self.assertEqual(updated.dtype,torch.float32)
        with torch.no_grad(), torch.autocast('cpu',dtype=torch.float16):
            expected,expected_state=layer.step(raw,state)
        self.assertTrue(torch.equal(result,expected))
        self.assertTrue(torch.equal(updated,expected_state))
        self.assertTrue(torch.isfinite(result).all().item())
        self.assertTrue(torch.isfinite(updated).all().item())
    def test_cached_bolt_fp16_projection_fp32_native_output_no_external_amp(self):
        from mlbricks.bolt.presets import BoltCache,BoltPreset
        from mlbricks.bolt import native_backend
        for mode in ('quality','fast'):
            model=BoltPreset(64,2,mode=mode,cache='tight',backend='pytorch').eval().half()
            model._native_enabled=lambda x:True  # emulate CUDA native reader selection
            x=torch.randn(1,1,64,dtype=torch.float16)
            c=BoltCache.allocate(1,2,10,16,32 if mode=='quality' else 0,
                                  device=x.device,dtype=x.dtype)
            with patch.object(native_backend,'get_extension',return_value=object()),\
                 patch.object(native_backend,'decode_append',
                       side_effect=lambda q,c_now,cache,scratch,pos,scale:
                           torch.ones_like(q,dtype=torch.float32)):
                with torch.no_grad():
                    actual=model.decode_with_cache(x,c,position=0)
            self.assertEqual(actual.dtype,torch.float16)
            self.assertTrue(torch.isfinite(actual).all().item())
            if c.state is not None:
                self.assertEqual(c.state.dtype,torch.float32)


class TestPatchV2StrictCompatibility(unittest.TestCase):
    def test_v2_method_signature_keeps_amp_wrapping_native_strict_update(self):
        import ast, subprocess
        source = subprocess.check_output(['git','-C',str(REPO),'show','HEAD:mlbricks/bolt/presets.py'],text=True)
        # MLBRICKS_COMMITTED_V2_TEST_FIX
        # HEAD is already patched on optimised-decode; check idempotency there.
        if '# MLBRICKS_BOLT_INFERENCE_AMP_V2' in source:
            scripts=REPO/'tools/bolt_optimised_decode_patch/scripts'
            sys.path.insert(0,str(scripts))
            from apply_bolt_dispatch_patch import locate_presets
            self.assertIn('_v2_strict.maybe_step',source)
            self.assertEqual(source,locate_presets(source))
            ast.parse(source)
            return
        source=source.replace('def step(self,o,state):',
                              'def step(self,o,state,*,_v2_cache_inplace=False):',1)
        needle='        a,v,r=self.components(o)\n        if self.use_native and o.is_cuda'
        self.assertIn(needle,source)
        source=source.replace(needle,"""        if self.use_native and o.is_cuda:
            candidate = _v2_strict.maybe_step(self, o, state, inplace_cache=_v2_cache_inplace)
            if candidate is not None:return candidate
        a,v,r=self.components(o)
        if self.use_native and o.is_cuda""",1)
        scripts=REPO/'tools/bolt_optimised_decode_patch/scripts'
        sys.path.insert(0,str(scripts))
        from apply_bolt_dispatch_patch import locate_presets
        after=locate_presets(source)
        ast.parse(after)
        self.assertIn('with torch.autocast',after)
        self.assertLess(after.index('MLBRICKS_BOLT_INFERENCE_AMP_V2'),
                        after.index('_v2_strict.maybe_step'))
        self.assertEqual(after,locate_presets(after))



class TestBOLTPrecisionDefault(unittest.TestCase):
    def test_public_default_fp32_and_explicit_auto_supported(self):
        from mlbricks import BoltConfig
        self.assertEqual(BoltConfig(vocab_size=50257).precision,'fp32')
        self.assertEqual(BoltConfig(vocab_size=50257,precision='auto').precision,'auto')
        self.assertEqual(BoltConfig(vocab_size=50257,precision='fp16').precision,'fp16')

if __name__=='__main__': unittest.main()
