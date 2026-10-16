# Qwen3-VL MegatronMIMO

Qwen3-VL Dense and MoE providers expose the `language` and `images` components
used by the generic MegatronMIMO converter. The image encoder is named
`qwen_visual`. Checkpoint parameter names are unchanged from standard Qwen3-VL.

The MIMO image boundary carries both the primary image embeddings and all
DeepStack features. The language component restores the visual mask and injects
the auxiliary features into the decoder. Discarding those features changes the
model even when a checkpoint round-trip succeeds.

This integration requires Megatron Core's `language_model_kwargs` support in
`MimoModel` (see the `feat/mimo-language-model-kwargs` branch of
[kota-row/Megatron-LM](https://github.com/kota-row/Megatron-LM/tree/feat/mimo-language-model-kwargs)).
The image payload is unpacked on the first language stage and passed explicitly
to the native Qwen3-VL language model for each forward call. No microbatch
features are stored on the model instance. Extra arguments are rank-local;
MIMO does not communicate or shard them automatically.

## Dense conversion

Run import and export as separate processes to exercise checkpoint reload:

```bash
uv run python -m torch.distributed.run --nproc_per_node=2 \
  examples/conversion/convert_megatron_mimo.py import \
  --hf-model Qwen/Qwen3-VL-2B-Instruct \
  --megatron-path /workspace/models/qwen3-vl-2b-mimo \
  --component language=tp=1,dp=1,rank_offset=0 \
  --component images=tp=1,dp=1,rank_offset=1 \
  --torch-dtype bfloat16

uv run python -m torch.distributed.run --nproc_per_node=2 \
  examples/conversion/convert_megatron_mimo.py export \
  --hf-model Qwen/Qwen3-VL-2B-Instruct \
  --megatron-path /workspace/models/qwen3-vl-2b-mimo \
  --hf-path /workspace/models/qwen3-vl-2b-export \
  --component language=tp=1,dp=1,rank_offset=0 \
  --component images=tp=1,dp=1,rank_offset=1 \
  --torch-dtype bfloat16
```

## MoE configuration

For `Qwen/Qwen3-VL-30B-A3B-Instruct`, use four language ranks and one image
rank on an allocation with at least five GPUs:

```text
language=tp=1,pp=1,cp=1,dp=4,ep=4,etp=1,rank_offset=0
images=tp=1,pp=1,cp=1,dp=1,rank_offset=4
```

EP subdivides the language data-parallel ranks; it does not multiply the
component's rank count. The Qwen3.5 MIMO finetuning entry point accepts this
model's HF configuration and the component `ep` option. Select the matching
processor and pretrained MIMO checkpoint explicitly.

## Limits

DeepStack MIMO currently requires language TP=CP=1 and sequence parallelism
disabled. The first language pipeline stage's first chunk must contain all
DeepStack injection layers (the first three decoder layers for the published
Qwen3-VL configurations). This is checked against the configured layer partition,
including uneven and virtual pipeline splits. Distributed PP training is not
validated by this example. Expert parallelism is supported within the language
DP group. Use non-colocated components. Packed sequences, video, and MTP are not validated
by this example. Standard non-MIMO providers keep their existing behavior.
