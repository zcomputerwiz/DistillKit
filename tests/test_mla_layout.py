"""Which columns of ``kv_b_proj`` belong to which head, and to the key or the value.

This is the invariant a conversion has to fit against, and getting it wrong is invisible
in every number a conversion prints. The fit solves a least squares against its own
target, so it reports a high r2 whatever order that target is in; only the forward knows
that the up-projection is read as ``view(..., num_heads, content_dim + head_dim)`` and
split per head. A target laid out as two blocks -- every head's key, then every head's
value -- satisfies the same least squares and then hands head 1's key to head 0 as its
value. The 2B converted 6.38 nats above its source that way while reporting key r2 0.97.

So pin the layout from the model's side: build an up-projection whose output is a known
pattern, run the forward, and check each head received its own key and its own value.
"""
import torch

from distillkit.models.qwen35.mla import Qwen35LatentAttention

from test_csa2_routing import tiny_config


def build():
    torch.manual_seed(0)
    return Qwen35LatentAttention(tiny_config(), layer_idx=1).eval()


def test_the_up_projection_is_read_per_head_not_as_two_blocks():
    """Each head's key and value sit together, so the target must interleave them."""
    module = build()
    heads, content, head_dim = module.num_heads, module.content_dim, module.head_dim
    assert module.kv_b_proj.out_features == heads * (content + head_dim)

    # One latent direction per head, carrying a value this head alone should see.
    marks = torch.arange(1, heads + 1, dtype=torch.float32)
    weight = torch.zeros(heads * (content + head_dim), module.latent)
    for head in range(heads):
        base = head * (content + head_dim)
        weight[base:base + content, 0] = marks[head]
        weight[base + content:base + content + head_dim, 0] = -marks[head]
    with torch.no_grad():
        module.kv_b_proj.weight.copy_(weight)
        latent = torch.zeros(1, 4, module.latent)
        latent[..., 0] = 1.0
        projected = module.kv_b_proj(latent).view(1, 4, heads, content + head_dim)
        key, value = torch.split(projected, [content, head_dim], dim=-1)

    for head in range(heads):
        assert torch.allclose(key[0, :, head], torch.full((4, content), marks[head]))
        assert torch.allclose(value[0, :, head],
                              torch.full((4, head_dim), -marks[head]))


def test_a_two_block_target_would_scramble_the_heads():
    """The layout the conversion used to build, shown failing on purpose.

    Without this the mistake is unfalsifiable: two-block and per-head targets are the same
    numbers in a different order, and every metric a conversion computes is order blind.
    """
    module = build()
    heads, content, head_dim = module.num_heads, module.content_dim, module.head_dim
    keys = torch.arange(1, heads + 1, dtype=torch.float32).repeat_interleave(content)
    values = -torch.arange(1, heads + 1, dtype=torch.float32).repeat_interleave(head_dim)
    blocks = torch.cat([keys, values])

    read = blocks.view(heads, content + head_dim)
    key, value = torch.split(read, [content, head_dim], dim=-1)
    # Head 0 still finds its own key, because both layouts start there.
    assert torch.allclose(key[0], torch.full((content,), 1.0))
    # Its value is not its value, and what it actually holds is head 1's key.
    assert not torch.allclose(value[0], torch.full((head_dim,), -1.0))
    assert torch.allclose(value[0][:content], torch.full((content,), 2.0)), \
        "head 0 should be reading head 1's key where its own value belongs"

    interleaved = torch.cat([
        keys.view(heads, content), values.view(heads, head_dim)], dim=-1).reshape(-1)
    read = interleaved.view(heads, content + head_dim)
    key, value = torch.split(read, [content, head_dim], dim=-1)
    for head in range(heads):
        assert torch.allclose(key[head], torch.full((content,), float(head + 1)))
        assert torch.allclose(value[head], torch.full((head_dim,), -float(head + 1)))
