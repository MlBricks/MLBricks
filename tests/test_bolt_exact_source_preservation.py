"""Guard source-specific V2/FP16 behavior while adding optional graph dispatch."""
import inspect
from unittest.mock import patch
import torch
from mlbricks import Bolt, BoltConfig, BoltModel
from mlbricks.bolt import native_backend
from mlbricks.bolt.presets import BoltPreset, BoltCache


def test_uploaded_fp16_native_cache_fix_still_active():
    m = BoltPreset(64, 4, mode='fast', cache='tight', backend='pytorch').eval()
    x = torch.randn(1, 1, 64)
    cache = BoltCache.allocate(1, 4, 5, 16, 0, device='cpu', dtype=torch.float16)
    q = torch.randn(1, 4, 1, 16, dtype=torch.float32)
    current = torch.randn_like(q)
    captured = []
    def mock_append(query, now, history, scratch, position, scale):
        captured.append((query.dtype, now.dtype, history.dtype))
        history[:,:,position:position+1].copy_(now)
        return torch.zeros_like(query)
    m._native_enabled = lambda x: True
    m._project = lambda x,start_pos=0: (q,current,current)
    with patch.object(native_backend, 'get_extension', return_value=object()), \
         patch.object(native_backend, 'decode_append', side_effect=mock_append):
        out = m.decode_with_cache(x, cache, position=0)
    assert captured == [(torch.float32, torch.float16, torch.float16)]
    assert out.isfinite().all() and cache.length == 1


def test_cache_session_reset_clears_capture_policy_counters():
    c=BoltCache.allocate(1,4,128,16,64,device='cpu',dtype=torch.float32)
    c._mlbricks_graph_calls = 91
    c._mlbricks_graph_unavailable = 'old failure'
    c.reset()
    assert c._mlbricks_graph_calls == 0
    assert not hasattr(c, '_mlbricks_graph_unavailable')
    c.load_prefix(torch.zeros(1,4,1,16), torch.zeros(1,64))
    assert c._mlbricks_graph_calls == 0


def test_model_and_layer_auto_signature_and_checkpoint():
    assert inspect.signature(Bolt).parameters['graph'].default == 'auto'
    model = BoltModel(BoltConfig(vocab_size=32,context=12,layers=2,dim=32,heads=4,device='cpu'))
    before=set(model.state_dict())
    assert model.graph == 'auto'
    model.set_graph(False)
    assert model.graph == 'off'
    assert set(model.state_dict()) == before
    assert all(b.attn.graph == 'off' for b in model.blocks)
