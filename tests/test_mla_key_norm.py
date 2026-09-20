"""The content key norm is opt-in, and a checkpoint that has one must say so.

DeepSeek puts one norm on the key/value path -- `kv_a_norm`, on the latent. This fork
also normalized the content half of the key after the up-projection, which neither parent
does: Qwen norms the whole head, DeepSeek norms the latent, and a norm over the content
half alone is a third thing. Dropping it changes the function rather than the layout, so
the flag stays for checkpoints trained with it while the default follows the reference.
"""
import torch

from distillkit.models.qwen35.mla import Qwen35LatentAttention

from test_csa2_routing import tiny_config


def build(**overrides):
    torch.manual_seed(0)
    return Qwen35LatentAttention(tiny_config(**overrides), layer_idx=1).eval()


def test_the_default_follows_deepseek_and_has_no_content_key_norm():
    assert build().k_norm is None
    assert "k_norm.weight" not in dict(build().named_parameters())


def test_the_flag_restores_the_norm_and_its_parameter():
    module = build(mla_content_key_norm=True)
    assert module.k_norm is not None
    assert "k_norm.weight" in dict(module.named_parameters())


def test_the_norm_changes_the_output_rather_than_only_its_layout():
    # Qwen3_5RMSNorm applies (1 + weight), so a zero gain is still a normalize: the two
    # variants have to disagree even before anything is trained.
    plain = build()
    normed = build(mla_content_key_norm=True)
    shared = dict(normed.named_parameters())
    with torch.no_grad():
        for name, parameter in plain.named_parameters():
            shared[name].copy_(parameter)
        states = torch.randn(1, 6, plain.config.hidden_size)
        width = int(plain.head_dim * 0.25)
        position = (torch.ones(1, 6, width), torch.zeros(1, 6, width))
        first = plain(states, position)[0]
        second = normed(states, position)[0]
    assert not torch.allclose(first, second, atol=1e-5)
