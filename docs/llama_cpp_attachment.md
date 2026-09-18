# How a standard part could attach to llama.cpp

Findings from reading `zcomputerwiz/llama.cpp` at `4fea119de`, recorded because the
answers were not what was assumed going in and the line numbers will drift. Nothing here
is a commitment. No part has been trained, nothing has been shown to displace anything,
and the program should not adopt a format before it has two examples to fit it against.

Read this as "what the runtime already does", not "what we will build".

## Why the question came up

The standard-parts program claims a module is trained once and reused across backbones.
That claim is about a portable artifact, so it eventually has to survive contact with a
runtime that loads artifacts. The question asked was narrow: can a module be contained in
the model file, or does it need a new architecture.

Neither, as it turns out.

## The load-time rule that shaped the first answer

`llama-model-loader.cpp:1392`:

```cpp
if (n_created < n_tensors) {
    if (!partial) {
        throw std::runtime_error(format("%s: wrong number of tensors; expected %d, got %d", ...));
```

`n_tensors` counts what is in the file, `n_created` counts what the graph builder asked
for. The only call site (`llama-model.cpp:1719`) uses the default `partial = false`.

**A tensor inside a model GGUF that the graph builder does not know by name is a hard load
error, not a warning.** So extra tensors in the backbone's own file are not free, and
"absent equals stock" only works in one direction: a patched build reads a plain file,
but a stock build cannot read a module-bearing one.

`qwen35.cpp:36` shows how the codebase handles an optional in-file component anyway, using
MTP as the example:

```cpp
const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;
```

`TENSOR_NOT_REQUIRED` means may-be-absent, `TENSOR_SKIP` means present-and-deliberately-
ignored -- the latter still increments `n_created`, which is what keeps the count
balanced. The builder must name the tensor either way.

## The finding that made the rule irrelevant

Adapters do not go through `llama_model_loader` at all.

`llama-adapter.cpp:158`, LoRA reading its own file:

```cpp
gguf_context_ptr ctx_gguf { gguf_init_from_file_ptr(file, meta_gguf_params) };
```

`llama-adapter.cpp:41-88`, the control vector building its own contexts and buffers and
taking device placement from the host model:

```cpp
ggml_backend_buffer_type_t buft = model.select_buft(il);
ggml_context * ctx = ctx_for_buft(buft);
...
ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
```

The adapter owns `ctxs` and `bufs`. **A component can own GGUF-loaded tensors, placed on
the correct backend device, without being a model and without entering the model loader's
accounting.** The tensor-count error applies only to the model's own file.

Consequence, if a module is ever shipped as its own file: the backbone GGUF stays
byte-identical and stock-loadable, and the module is genuinely additive. That is a better
match for "train once, reuse across backbones" than embedding module tensors into every
backbone checkpoint, because the reusable artifact becomes one separately versioned file
rather than something re-embedded per model.

## Four attachment points the runtime already distinguishes

| attachment | tensors | mechanism | precedent |
| --- | --- | --- | --- |
| input, per-position | yes | own loader, a `build_inp_*`, a connector | mmproj; cvec for placement |
| output, in-graph | yes | own loader, `llm_graph_input_sampling` | sampling-in-graph |
| output, CPU only | no | sampler chain | grammar sampler |
| per-layer, position-independent | yes | control vector, unchanged llama.cpp | `llama_adapter_cvec` |

Two of these are worth separating carefully.

**Control vectors are the closest fit and not close enough.** `llama-adapter.h:17` gives
`llama_adapter_cvec` a per-layer tensor list and an `apply_to(ctx, cur, il)` that every
graph builder already calls through `build_cvec` (`llama-graph.cpp:1508`). Separate file,
per-layer additive, zero changes to llama.cpp. But a control vector is **static** -- the
same vector added at every position. Anything context-dependent, which the repetition
index is by definition, cannot use it.

That is a genuine fork in the road for the candidate-parts table: a part whose output
depends only on position-independent state ships today with no runtime changes at all. A
part whose output depends on context does not.

**Sampling happens inside the graph.** `llama-graph.h:745`:

```cpp
class llm_graph_input_sampling : public llm_graph_input_i {
    std::map<llama_seq_id, llama_sampler *> samplers;
```

and `samplers` sits in the graph params beside `cvec`, `loras`, `mctx`, `cross`. So an
output-end part with weights has somewhere to live on-device. Meanwhile `llama-sampler.cpp`
holds no tensor loading whatsoever -- one `llama_get_model` call for vocab access and
nothing else -- so a sampler-based part is CPU algorithm only. That splits the output-end
candidates cleanly: n-gram prior and copy distribution want in-graph, constraint mask
wants the sampler chain.

## The suffix-history problem is already solved upstream

The worry was that a repetition index needs accumulated per-sequence state, and that
state would have to track the KV cache through copy, rollback and defragmentation.

It does not, because the token history is already retained and already has an accessor
built for a near-identical purpose.

`llama-kv-cells.h:20` -- each cell stores `llama_token tok`. `llama-kv-cells.h:321`:

```cpp
// the token of the cell of sequence seq_id at the largest position <= p
// note: used by n-gram input embeddings to recover the tokens preceding a ubatch
llama_token seq_pos_tok_le(llama_seq_id seq_id, llama_pos p) const {
```

with the batch-level primitive at `llama-kv-cache.cpp:1836`:

```cpp
void llama_kv_cache::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const
```

It already handles the cases that would have caused silent wrongness: M-RoPE position
gaps, embd batches where position does not encode token order, and -- flagged in its own
`TODO` -- a token belonging to several sequences having ambiguous history, which n-gram
architectures must reject.

Retention is opt-in per architecture, `llama-kv-cache.cpp:1831`:

```cpp
bool llama_kv_cache::has_cell_ext() const {
    // M-RoPE needs the 2D position, the PLE n-gram hash needs the token id
    return hparams.n_pos_per_embd() > 1 || hparams.ple_n_heads > 0;
}
```

Qwen3.5's PLE n-gram hash is already one of the consumers. A repetition index is the same
category of consumer, which means **the index is a cache rather than state** -- always
recomputable from history the runtime keeps, so rollback and sequence copy need no special
handling beyond rebuilding.

By contrast the recurrent state machinery is the wrong shape and was the wrong thing to
reach for. `llama-memory-recurrent.h` holds `std::vector<ggml_tensor *> r_l, s_l, p_l` --
fixed-size per-layer tensors in backend buffers, sized by `size_r_bytes()` and friends.
Size-rigid and device-resident. A growing hash table does not go there.

## DFlash, as a packaging example

Checked because it taps several points of a running model and that sounded relevant. It is
relevant, but not for the tap.

DFlash is a separate model with its own `LLM_ARCH_DFLASH`, its own GGUF, loaded alongside
the target and linked through `cparams.ctx_other` (`llama-context.cpp:155`). Target hidden
states cross as an embd batch at encoder width (`n_embd_inp_enc`) and are injected into the
draft's KV by a pass that stores K/V without attending -- `llama-graph.cpp:475` notes the
mask is deliberately left unallocated for it.

The transferable detail is that the taps are declared as metadata rather than hardcoded:

```
{ LLM_KV_TARGET_LAYERS, "%s.target_layers" },
```

with `llama_model_target_layer_ids()` and `_n()` exposed in `llama-ext.h:126`. If a part
ever reads intermediate activations instead of token ids, this is the established idiom and
there is no reason to invent another.

The wider point is that DFlash demonstrates a component shipping as its own arch and its own
file, linked to a host at runtime. Whether a standard part should be packaged that way is
open; it is heavier than an adapter and lighter than a fork.

## Not verified

- Whether the CPU cost of the index at decode time is acceptable. It was measured at
  **0.118 s per training step** at batch 64 x 1024 (21% of a 0.56 s step, hideable behind
  the GPU with a prefetch), but decode is a different access pattern and was not measured.
- Whether a module GGUF can reuse the adapter loading route directly or would need its own,
  since `llama_adapter_lora` assumes a particular tensor-pair layout.
- Whether `ple_n_heads` gating of `has_cell_ext()` is the right hook for a different
  consumer, or whether that needs its own hparam.
- Any of this end to end. Nothing was built or run.

## What this does not settle

The architecture question is downstream of the science question. If the copy experiment's
decisive cell -- ablated module, copy probe -- shows the backbone built the function anyway,
there is no module to package and none of the above matters. The order stays: displacement
first, reuse second, packaging last.

Recorded so the reading does not have to be repeated, not to fix a design.
