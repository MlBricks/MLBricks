"""Production BOLT causal language model with automatic CUDA placement."""
from __future__ import annotations
import contextlib
import torch
from ..gaussian import Gaussian, GaussianConfig
from .config import BoltConfig


class BoltModel(Gaussian):
    """Complete BOLT language model.

    The constructor automatically places model weights on CUDA when available.
    Input token IDs and targets are transferred to the model's current device.
    ``precision='auto'`` uses autocast on CUDA and full precision on CPU.
    """
    def __init__(self, config: BoltConfig | GaussianConfig | None = None, **kwargs):
        if config is None:
            aliases = {'block':'context','n_layer':'layers','embd':'dim','head':'heads'}
            for old,new in aliases.items():
                if old in kwargs and new not in kwargs:
                    kwargs[new] = kwargs.pop(old)
            config = BoltConfig(**kwargs)
        elif kwargs:
            raise TypeError('Pass a config or keyword arguments, not both')
        if not isinstance(config, GaussianConfig):
            raise TypeError('config must be BoltConfig or GaussianConfig')
        super().__init__(config)
        requested = getattr(config,'device','auto')
        if str(requested).lower() == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(requested)
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA was explicitly requested but is unavailable')
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def _autocast(self):
        precision = getattr(self.config,'precision','fp32')
        if self.device.type != 'cuda' or precision == 'fp32':
            return contextlib.nullcontext()
        if precision == 'auto':
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif precision == 'bf16':
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError('Requested BF16 precision is unsupported by this CUDA GPU')
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        return torch.autocast('cuda',dtype=dtype)

    def _on_device(self, data):
        if data is None:
            return None
        return data.to(self.device, non_blocking=True) if data.device != self.device else data

    def forward(self, input_ids, *, targets=None):
        input_ids = self._on_device(input_ids)
        targets = self._on_device(targets)
        with self._autocast():
            return super().forward(input_ids, targets=targets)

    @torch.no_grad()
    def prefill(self,input_ids,*,cache_buffers=None):
        with self._autocast():
            return super().prefill(self._on_device(input_ids), cache_buffers=cache_buffers)

    @torch.no_grad()
    def decode_step(self,input_ids,caches,*,position):
        with self._autocast():
            return super().decode_step(self._on_device(input_ids),caches,position=position)

    @torch.no_grad()
    def generate(self,input_ids,max_new_tokens,*,temperature=1.,top_k=None):
        # Parent implementation calls self.prefill/decode_step, which handle autocast.
        return super().generate(self._on_device(input_ids),max_new_tokens,
                                temperature=temperature,top_k=top_k)
