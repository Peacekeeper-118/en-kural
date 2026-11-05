"""Inference speed-ups for the vendored Seed-VC / BigVGAN modules.

The vendored code stays untouched; these helpers patch module instances at
load time and are all mathematically equivalent to the originals (verified by
tests/test_fast_ops.py):

* ``fuse_bigvgan_activations`` – BigVGAN's anti-aliased activation
  (2× upsample → Snake → 2× downsample, ~70 % of vocoder time) rewritten as
  polyphase shifted sums.  No depthwise convolutions or padded intermediates
  remain, so ``torch.compile`` fuses it into a couple of Triton kernels — the
  job BigVGAN's optional CUDA kernel does, without needing nvcc.
* ``FlashAttentionSwitch`` – the DiT passes an all-true padding mask to
  ``scaled_dot_product_attention`` even when nothing is padded, which rules out
  the flash-attention kernel.  The switch drops the mask while the engine runs
  unpadded batches.
"""
from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import List, Optional

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- BigVGAN
def _taps(filt: torch.Tensor) -> List[float]:
    return [float(v) for v in filt.reshape(-1).tolist()]


def anti_alias_snake(x: torch.Tensor, h: List[float], g: List[float], alpha: torch.Tensor,
                     inv_beta: torch.Tensor) -> torch.Tensor:
    """Activation1d(Snake/SnakeBeta) with 2× up/down sampling and 12-tap filters.

    Upsampling by 2 with filter ``h`` splits into an even and an odd phase of
    6 taps each; the Snake runs on both phases; the stride-2 low-pass ``g``
    reads the phases back with 6 taps each.  Replicate padding at the edges is
    reproduced exactly (see the derivation in tests/test_fast_ops.py).
    """
    T = x.shape[-1]
    xp = F.pad(x, (3, 3), mode="replicate")
    e = 2.0 * sum(h[2 * q + 1] * xp[..., 5 - q: 5 - q + T] for q in range(6))
    o = 2.0 * sum(h[2 * q] * xp[..., 6 - q: 6 - q + T] for q in range(6))
    e = e + inv_beta * torch.sin(e * alpha).pow(2)
    o = o + inv_beta * torch.sin(o * alpha).pow(2)
    first, last = e[..., :1], o[..., -1:]
    E = torch.cat([first, first, e, last, last, last], dim=-1)          # e[i+q-2], edges replicated
    O = torch.cat([first, first, first, o, last, last], dim=-1)         # o[i+q-3]
    return (sum(g[2 * q + 1] * E[..., q: q + T] for q in range(6))
            + sum(g[2 * q] * O[..., q: q + T] for q in range(6)))


_compiled_act = None
_compile_lock = threading.Lock()


def _act_fn(compile_: bool):
    global _compiled_act
    if not compile_:
        return anti_alias_snake
    with _compile_lock:
        if _compiled_act is None:
            _compiled_act = torch.compile(anti_alias_snake, dynamic=True)
        return _compiled_act


def fuse_bigvgan_activations(vocoder: torch.nn.Module, compile_: bool = True) -> int:
    """Replace every anti-aliased Snake activation of a (weight-norm-removed,
    eval-mode) BigVGAN with the polyphase version; returns how many."""
    from modules.bigvgan import activations
    from modules.bigvgan.alias_free_activation.torch.act import Activation1d

    fn = _act_fn(compile_)
    n = 0
    for mod in vocoder.modules():
        if not isinstance(mod, Activation1d):
            continue
        up, down, act = mod.upsample, mod.downsample, mod.act
        if (up.ratio, up.kernel_size, down.ratio, down.kernel_size) != (2, 12, 2, 12):
            continue  # derivation assumes BigVGAN's 2× / 12-tap setup
        with torch.no_grad():
            alpha = act.alpha.detach()[None, :, None]
            if isinstance(act, activations.SnakeBeta):
                beta = act.beta.detach()[None, :, None]
            elif isinstance(act, activations.Snake):
                beta = alpha
            else:
                continue
            if act.alpha_logscale:
                alpha, beta = torch.exp(alpha), torch.exp(beta)
            inv_beta = 1.0 / (beta + act.no_div_by_zero)
        h, g = _taps(up.filter), _taps(down.lowpass.filter)
        mod.forward = (lambda x, _h=h, _g=g, _a=alpha, _b=inv_beta: fn(x, _h, _g, _a, _b))
        n += 1
    return n


# --------------------------------------------------------------------------- DiT attention
class FlashAttentionSwitch:
    """Stands in for ``torch.nn.functional`` inside ``modules.diffusion_transformer``.

    While ``unpadded()`` is active, the padding mask handed to
    ``scaled_dot_product_attention`` is dropped (it is all true then), so
    PyTorch can pick the flash-attention kernel.  Everything else is forwarded.
    """

    def __init__(self, functional) -> None:
        self._f = functional
        # a plain flag (not thread-local) so torch.compile can guard on it; the
        # engine lock already serialises conversions
        self.active = False

    def __getattr__(self, name):
        return getattr(self._f, name)

    def scaled_dot_product_attention(self, q, k, v, attn_mask=None, **kw):
        if self.active:
            attn_mask = None
        return self._f.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, **kw)

    @contextlib.contextmanager
    def unpadded(self):
        self.active = True
        try:
            yield
        finally:
            self.active = False


_switch: Optional[FlashAttentionSwitch] = None


def flash_attention_switch() -> FlashAttentionSwitch:
    """Install (once) and return the switch for the vendored DiT module."""
    global _switch
    import modules.diffusion_transformer as dt

    if _switch is None:
        _switch = FlashAttentionSwitch(F)
    dt.F = _switch
    return _switch


def compile_enabled(device) -> bool:
    return getattr(device, "type", str(device)) == "cuda" and os.environ.get("ENKURAL_COMPILE", "1") != "0"
