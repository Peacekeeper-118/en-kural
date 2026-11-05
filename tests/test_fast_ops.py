"""Equivalence tests for the inference speed-ups (CPU, float64 where possible).

Polyphase derivation checked here: with x replicate-padded, BigVGAN's 2×
upsampler (12-tap h, pad 5, crop 15) gives
    u[2j]   = 2 Σ_q h[2q+1] x[clamp(j+2-q)],   u[2j+1] = 2 Σ_q h[2q] x[clamp(j+3-q)]
and the stride-2 low-pass (12-tap g, pad 5/6) gives
    y[i] = Σ_q g[2q+1] s[clamp(2(i+q-2))] + Σ_q g[2q] s[clamp(2(i+q-3)+1)]
where s = snake(u) and clamping at the ends reads s[0] = e[0] or s[2T-1] = o[T-1].
"""
import copy

import pytest
import torch

from enkural import config
from enkural.vc import fast_ops
from enkural.vc.seedvc_v1 import LEGACY_SAMPLER, SAMPLER, SeedVCv1, time_grid

pytestmark = pytest.mark.skipif(not (config.SEEDVC_DIR / "modules").exists(), reason="vendored Seed-VC missing")


@pytest.fixture(scope="module")
def seedvc():
    from enkural.vc.hf import setup_seedvc

    setup_seedvc()


@pytest.mark.parametrize("kind", ["snakebeta_log", "snakebeta", "snake"])
def test_polyphase_activation_matches_original(seedvc, kind):
    from modules.bigvgan import activations
    from modules.bigvgan.alias_free_activation.torch.act import Activation1d

    torch.manual_seed(0)
    C = 16
    if kind == "snake":
        act = activations.Snake(C, alpha_logscale=False)
    else:
        act = activations.SnakeBeta(C, alpha_logscale=kind.endswith("log"))
    with torch.no_grad():
        act.alpha.copy_(torch.randn(C) * 0.5 + (0 if act.alpha_logscale else 1.5))
        if hasattr(act, "beta"):
            act.beta.copy_(torch.randn(C) * 0.5 + (0 if act.alpha_logscale else 1.5))
    ref = Activation1d(activation=act).double().eval()
    fast = copy.deepcopy(ref)
    assert fast_ops.fuse_bigvgan_activations(torch.nn.ModuleList([fast]), compile_=False) == 1
    for T in (1, 2, 3, 7, 64, 513):
        x = torch.randn(2, C, T, dtype=torch.float64) * 3
        with torch.no_grad():
            assert torch.allclose(fast(x), ref(x), atol=1e-12, rtol=0), T


def test_flash_switch_only_drops_mask_when_active():
    calls = []

    class FakeF:
        def scaled_dot_product_attention(self, q, k, v, attn_mask=None, **kw):
            calls.append(attn_mask)
            return q

        def silu(self, x):
            return "forwarded"

    sw = fast_ops.FlashAttentionSwitch(FakeF())
    mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    sw.scaled_dot_product_attention(1, 2, 3, attn_mask=mask)
    with sw.unpadded():
        sw.scaled_dot_product_attention(1, 2, 3, attn_mask=mask)
    sw.scaled_dot_product_attention(1, 2, 3, attn_mask=mask)
    assert calls[0] is mask and calls[1] is None and calls[2] is mask
    assert sw.silu(0) == "forwarded"


def test_flash_switch_is_exact_for_unpadded_attention():
    import torch.nn.functional as F

    q, k, v = (torch.randn(2, 4, 10, 8, dtype=torch.float64) for _ in range(3))
    mask = torch.ones(2, 1, 10, 10, dtype=torch.bool)
    sw = fast_ops.FlashAttentionSwitch(F)
    with sw.unpadded():
        fast = sw.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(fast, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-12)


def test_time_grids():
    lin, sway = time_grid(16), time_grid(16, "sway", -1.0)
    for t in (lin, sway):
        assert float(t[0]) == 0.0 and abs(float(t[-1]) - 1.0) < 1e-6
        assert bool((t[1:] > t[:-1]).all())
    assert float(sway[1]) < float(lin[1])          # sway front-loads the step budget
    assert torch.equal(time_grid(16, "sway", 0.0), lin)
    assert SAMPLER["steps"]["standard"] < LEGACY_SAMPLER["steps"]["standard"]


def test_legacy_sampler_reproduces_upstream(seedvc):
    """Our sampler with the legacy settings must equal upstream CFM.inference
    bit for bit (same random draws), on a randomly initialised DiT."""
    import yaml
    from modules.commons import build_model, recursive_munch

    from enkural.vc.hf import hf_download
    from enkural.vc.seedvc_v1 import BASE_CKPT, REPO

    try:
        cfg_path = hf_download(REPO, BASE_CKPT[1])
    except Exception:
        pytest.skip("Seed-VC config not downloaded")
    params = recursive_munch(yaml.safe_load(open(cfg_path))["model_params"])
    torch.manual_seed(0)
    model = build_model(params, stage="DiT")
    cfm = model.cfm.eval()
    cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=256)
    T, P = 48, 12
    mu = torch.randn(1, T, params.DiT.content_dim)
    prompt = torch.randn(1, params.DiT.in_channels, P)
    style = torch.randn(1, 192)

    fake = type("E", (), {})()
    fake.sampler, fake._flash = copy.deepcopy(LEGACY_SAMPLER), None
    with torch.inference_mode():
        torch.manual_seed(1)
        ref = cfm.inference(mu.clone(), torch.LongTensor([T]), prompt, style, None, 6, inference_cfg_rate=0.7)
        torch.manual_seed(1)
        ours = SeedVCv1._sample(fake, cfm, mu.clone(), prompt, style, None, 6, 0.7)
        assert torch.equal(ours, ref)
        # the tuned sampler runs the same machinery, just on a different grid / window
        fake.sampler = copy.deepcopy(SAMPLER)
        torch.manual_seed(1)
        tuned = SeedVCv1._sample(fake, cfm, mu.clone(), prompt, style, None, 6, 0.7)
    assert tuned.shape == ref.shape and torch.isfinite(tuned).all()
