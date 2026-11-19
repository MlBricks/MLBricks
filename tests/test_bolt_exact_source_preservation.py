"""Guard source-specific V2/FP16 behavior while adding optional graph dispatch."""
import inspect
from unittest.mock import patch
import torch
from mlbricks import Bolt, BoltConfig, BoltModel
from mlbricks.bolt import native_backend
from mlbricks.bolt.enhanced_mqa import EnhancedMQACache






def test_model_and_layer_auto_signature_and_checkpoint():
    assert inspect.signature(Bolt).parameters['graph'].default == 'auto'
    model = BoltModel(BoltConfig(vocab_size=32,context=12,layers=2,dim=32,heads=4,device='cpu'))
    before=set(model.state_dict())
    assert model.graph == 'auto'
    model.set_graph(False)
    assert model.graph == 'off'
    assert set(model.state_dict()) == before
    assert all(b.attn.graph == 'off' for b in model.blocks)

def test_modern_cache_reset_owns_only_fp32_state():
    c=EnhancedMQACache.allocate(1,128,16,64,device='cpu',dtype=torch.float16)
    c.load_prefix(torch.zeros(1,1,4,16),torch.zeros(1,1,4,16),torch.ones(1,64))
    c.reset()
    assert c.length == 0 and c.state.dtype == torch.float32 and torch.count_nonzero(c.state)==0
