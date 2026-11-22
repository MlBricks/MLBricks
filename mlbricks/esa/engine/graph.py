"""Opt-in, device-isolated CUDA Graph greedy decoding for fastesa models.

Captures N sequential recurrent next-byte/token steps with GPU-resident state
and next-token feedback. No Python/model launch per emitted token on replay.
Call AFTER training and BEFORE benchmarking; the model must stay on the same
GPU with the same parameters and tensor shapes for the lifetime of the graph.
"""
from __future__ import annotations

import torch
import math

from .api import forward_tokens


class GraphDecoder:
    """Greedy CUDA Graph decoder for a fixed batch size and unroll length.

    Example::

        from fastesa import CUDAGreedyGraph, make_model
        model = make_model('quality').to('cuda:1').eval()
        prompt = torch.tensor([[79, 110, 99, 101]], device='cuda:1')
        decoder = CUDAGreedyGraph(model, prompt, unroll=8)
        decoder.check_parity()
        tokens = decoder.generate(128)  # [B,128] on GPU

    Optimizer steps or weight loads invalidate the captured graph.
    No seed/prompt changes after capture; create a new graph for a new prompt.
    Training uses normal forward/backward, not this class.
    """

    @torch.inference_mode()
    def __init__(self, model, prompt: torch.Tensor, *, unroll: int = 8,
                 amp: bool = True, dtype: torch.dtype = torch.float16):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDAGreedyGraph requires CUDA')
        if prompt.ndim != 2 or prompt.shape[1] < 1:
            raise ValueError('prompt must have shape [batch, positive sequence length]')
        if prompt.dtype != torch.long or not prompt.is_cuda:
            raise ValueError('prompt must be torch.long on CUDA')
        if not isinstance(unroll, int) or unroll < 1:
            raise ValueError('unroll must be a positive integer')
        if model.training:
            raise ValueError('Run model.eval() before CUDA Graph capture')
        try:
            param_dev = next(model.parameters()).device
        except StopIteration as e:
            raise ValueError('model has no parameters') from e
        if param_dev != prompt.device:
            raise ValueError('Model and prompt must be on the same GPU')
        self.model, self.device, self.unroll = model, prompt.device, unroll
        self.amp, self.dtype = amp, dtype
        with torch.cuda.device(self.device):
            # Prefill before capture. Also lazily compiles the native quality
            # decode kernel via warmups, avoiding compiles inside capture.
            with torch.autocast('cuda', dtype=self.dtype, enabled=self.amp):
                first, initial = forward_tokens(model, prompt)
            self._start_token = first.argmax(-1).view(-1, 1).contiguous().clone()
            self._start_states = [s.detach().contiguous().clone() for s in initial]
            self._token_input = self._start_token.clone()
            self._state_inputs = [s.clone() for s in self._start_states]
            self._out = torch.empty((prompt.size(0), unroll),
                                    dtype=torch.long, device=self.device)

            # Correct PyTorch graph warmup: do it on a separate GPU-specific
            # stream, then make the capture stream wait for completion.
            warm_stream = torch.cuda.Stream(device=self.device)
            warm_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(warm_stream):
                with torch.autocast('cuda', dtype=self.dtype, enabled=self.amp):
                    for _ in range(3):
                        token = self._start_token.clone()
                        states = [s.clone() for s in self._start_states]
                        for _j in range(unroll):
                            logits, states = forward_tokens(model, token, states)
                            token = logits.argmax(-1).view(-1, 1)
            torch.cuda.current_stream(self.device).wait_stream(warm_stream)
            torch.cuda.synchronize(self.device)
            # IMPORTANT: implicit torch.cuda.graph() capture stream can be
            # reused across cuda:0 and cuda:1. Always set a per-device stream.
            self._capture_stream = torch.cuda.Stream(device=self.device)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, stream=self._capture_stream):
                with torch.autocast('cuda', dtype=self.dtype, enabled=self.amp):
                    token = self._token_input
                    states = self._state_inputs
                    outputs = []
                    for _ in range(unroll):
                        outputs.append(token)
                        logits, states = forward_tokens(model, token, states)
                        token = logits.argmax(-1).view(-1, 1)
                    self._out.copy_(torch.cat(outputs, dim=1))
                    self._token_input.copy_(token)
                    for dst, src in zip(self._state_inputs, states):
                        dst.copy_(src)
            self.reset()

    @torch.inference_mode()
    def reset(self):
        """Restore initial token and recurrent states (no new prompt prefill)."""
        with torch.cuda.device(self.device):
            self._token_input.copy_(self._start_token)
            for dst, src in zip(self._state_inputs, self._start_states):
                dst.copy_(src)

    @torch.inference_mode()
    def step(self):
        """Replay one unroll, returning a new GPU tensor of emitted tokens."""
        with torch.cuda.device(self.device):
            self._graph.replay()
            return self._out.clone()

    @torch.inference_mode()
    def generate(self, tokens: int, *, reset: bool = True) -> torch.Tensor:
        """Generate exactly `tokens` new greedy tokens, leaving state on GPU.

        Token count must be a multiple of unroll to avoid advancing the recurrent
        state past the returned output. Return tensor [B,tokens] on GPU.
        """
        if not isinstance(tokens, int) or tokens < 1 or tokens % self.unroll:
            raise ValueError(f'tokens must be a positive multiple of {self.unroll}')
        if reset:
            self.reset()
        return torch.cat([self.step() for _ in range(tokens // self.unroll)], dim=1)

    @torch.inference_mode()
    def check_parity(self, *, chunks: int = 2, state_atol: float = 0.003) -> float:
        """Assert graph vs eager greedy-token and state parity on same GPU.

        Returns maximum final-state difference; graph remains reset afterward.
        """
        if chunks < 1:
            raise ValueError('chunks must be >=1')
        with torch.cuda.device(self.device):
            token = self._start_token.clone()
            states = [s.clone() for s in self._start_states]
            expected = []
            with torch.autocast('cuda', dtype=self.dtype, enabled=self.amp):
                for _ in range(chunks * self.unroll):
                    expected.append(token)
                    logits, states = forward_tokens(self.model, token, states)
                    token = logits.argmax(-1).view(-1, 1)
            expected = torch.cat(expected, dim=1)
            actual = self.generate(chunks * self.unroll)
            torch.cuda.synchronize(self.device)
            if not torch.equal(actual, expected):
                self.reset()
                raise AssertionError('CUDA Graph token mismatch against eager decode')
            diff = max((a - b).abs().max().item() for a, b in zip(self._state_inputs, states))
            if not torch.equal(self._token_input, token) or not math.isfinite(diff) or diff > state_atol:
                self.reset()
                raise AssertionError(f'CUDA Graph state/next-token mismatch: max_diff={diff}')
            self.reset()
            return diff


CUDAGreedyGraph = GraphDecoder  # compatibility alias
