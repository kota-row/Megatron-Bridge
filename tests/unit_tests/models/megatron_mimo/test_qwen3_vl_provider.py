# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MIMO contracts shared by Qwen3-VL Dense and MoE."""

from unittest.mock import MagicMock

import pytest
import torch
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.megatron_mimo.conversion import get_mimo_conversion_spec, supports_mimo_conversion
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
from megatron.bridge.models.qwen_vl import qwen3_vl_provider as provider_module
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.mimo import Qwen3VLMIMOGPTModel as Qwen3VLGPTModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.mimo import Qwen3VLMIMOVisionModel as Qwen3VLVisionModel
from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import Qwen3VLBridge, Qwen3VLMoEBridge
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider, Qwen3VLMoEModelProvider


pytestmark = pytest.mark.unit


@pytest.fixture(
    params=[(Qwen3VLModelProvider, Qwen3VLBridge), (Qwen3VLMoEModelProvider, Qwen3VLMoEBridge)], ids=["dense", "moe"]
)
def model_pair(request):
    provider_cls, bridge_cls = request.param
    kwargs = dict(
        num_layers=2, hidden_size=128, num_attention_heads=4, num_query_groups=4, kv_channels=32, vocab_size=256
    )
    if provider_cls is Qwen3VLMoEModelProvider:
        kwargs.update(num_moe_experts=8, moe_router_topk=2, moe_ffn_hidden_size=64)
    return provider_cls(**kwargs), bridge_cls


def test_metadata_and_default_conversion(model_pair):
    provider, bridge_cls = model_pair
    assert provider.modality_keys == {"images": "qwen_visual"}
    assert provider.special_token_ids == {"images": provider.image_token_id}
    assert bridge_cls.mimo_source_prefixes == {"language": "language_model.", "images": "vision_model."}
    assert supports_mimo_conversion(bridge_cls)
    assert callable(get_mimo_conversion_spec(bridge_cls))


def test_language_spec_selects_dense_or_grouped_experts(model_pair, monkeypatch):
    provider, _ = model_pair
    factory = MagicMock(return_value=ModuleSpec(module=object))
    monkeypatch.setattr(provider_module, "get_gpt_layer_with_transformer_engine_spec", factory)
    assert provider.build_language_spec() is factory.return_value
    factory.assert_called_once_with(
        num_experts=provider.num_moe_experts,
        moe_grouped_gemm=provider.moe_grouped_gemm if provider.num_moe_experts is not None else False,
        qk_layernorm=provider.qk_layernorm,
        fp8=False,
    )


def test_language_spec_preserves_qwen_configuration(model_pair, monkeypatch):
    provider, _ = model_pair
    provider.logit_dtype = torch.float32
    provider.fp16_lm_cross_entropy = True
    provider.image_token_id = 42
    layer = ModuleSpec(module=object)
    monkeypatch.setattr(provider, "build_language_spec", MagicMock(return_value=layer))
    spec = provider.build_language_model_spec(pp_rank=1)
    assert spec.module is Qwen3VLGPTModel
    assert spec.params["config"] is provider
    assert spec.params["transformer_layer_spec"] is layer
    assert spec.params["logit_dtype"] is torch.float32
    assert spec.params["fp16_lm_cross_entropy"] is True
    assert spec.params["position_embedding_type"] == "mrope"
    assert spec.params["rotary_base"] == provider.rotary_base
    assert spec.params["max_sequence_length"] == provider.language_max_sequence_length
    assert spec.params["share_embeddings_and_output_weights"] == provider.share_embeddings_and_output_weights
    assert spec.params["scatter_embedding_sequence_parallel"] is False
    assert spec.params["mtp_block_spec"] is None
    assert provider.special_token_ids == {"images": 42}
    provider.build_language_spec.assert_called_once_with(pp_rank=1)
    assert "pg_collection" not in spec.params


def test_native_vision_spec_has_independent_pipeline(model_pair):
    provider, _ = model_pair
    spec = provider.build_vision_encoder_spec()
    assert spec.module is Qwen3VLVisionModel
    config = spec.params["transformer_config"]
    assert config.pipeline_model_parallel_size == 1
    assert config.first_pipeline_num_layers is None
    assert config.hidden_size == provider.vision_config.hidden_size
    assert config.deepstack_visual_indexes == provider.vision_config.deepstack_visual_indexes
    assert spec.params["pre_process"] and spec.params["post_process"]
    assert "pg_collection" not in spec.params


def test_hf_vision_is_rejected(model_pair):
    provider, _ = model_pair
    provider.use_hf_vision_model = True
    with pytest.raises(ValueError, match="use_hf_vision_model"):
        provider.build_vision_encoder_spec()


def test_default_routes_and_factory(model_pair):
    standard, bridge_cls = model_pair
    source = bridge_cls()
    source.provider_bridge = MagicMock(return_value=standard)
    parallelism = MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=1),
            "images": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=1, rank_offset=1),
        }
    )
    hf_source = object()
    provider, routes = get_mimo_conversion_spec(bridge_cls)(source, hf_source, parallelism)
    assert isinstance(provider, MegatronMIMOProvider)
    assert provider.standard_provider is standard
    assert provider.language_model_spec.module is Qwen3VLGPTModel
    assert provider.special_token_ids == {"images": standard.image_token_id}
    assert (
        provider.modality_submodules_spec["images"].submodules["encoders"]["qwen_visual"].module is Qwen3VLVisionModel
    )
    assert {(r.source_prefix, r.target_module_path) for r in routes} == {
        ("language_model.", "language_model"),
        ("vision_model.", "modality_submodules.images.encoders.qwen_visual"),
    }
    source.provider_bridge.assert_called_once_with(hf_source)


@pytest.mark.parametrize(
    "field", ["tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"]
)
def test_deepstack_rejects_unimplemented_sharding(model_pair, field):
    provider, _ = model_pair
    assert hasattr(provider, field)
    setattr(provider, field, 2)
    with pytest.raises(ValueError, match="DeepStack MIMO"):
        provider.build_language_model_spec()


def test_deepstack_payload_preserves_features_and_gradients(monkeypatch):
    from types import SimpleNamespace

    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.mimo import Qwen3VLMIMOModel, Qwen3VLMIMOVisionModel
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel as NativeVision

    primary = torch.randn(2, 4, requires_grad=True)
    auxiliary = [torch.randn(2, 4, requires_grad=True) for _ in range(2)]
    monkeypatch.setattr(NativeVision, "forward", lambda *args, **kwargs: (primary, auxiliary))
    encoder = Qwen3VLMIMOVisionModel.__new__(Qwen3VLMIMOVisionModel)
    torch.nn.Module.__init__(encoder)
    payload = encoder()
    model = Qwen3VLMIMOModel.__new__(Qwen3VLMIMOModel)
    torch.nn.Module.__init__(model)
    model.language_model = torch.nn.Identity()
    model.mimo_config = SimpleNamespace(
        language_model_spec=ModuleSpec(
            module=object,
            params={
                "config": SimpleNamespace(
                    hidden_size=4, vision_config=SimpleNamespace(deepstack_visual_indexes=[0, 1])
                )
            },
        )
    )
    ids = torch.tensor([[3, 9, 9, 4]])
    text = torch.randn(2, 4, requires_grad=True)
    result = model.align_embeddings_by_token_positions({"images": payload, "text": text}, ids, {"images": 9})
    torch.testing.assert_close(result[:, 0], torch.stack([text[0], primary[0], primary[1], text[1]]))
    mask, received = model.language_model._mimo_visual_inputs
    assert torch.equal(mask, ids == 9)
    for actual, expected in zip(received, auxiliary):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (result.sum() + sum(t.sum() for t in received)).backward()
    for tensor in [primary, text, *auxiliary]:
        assert torch.equal(tensor.grad, torch.ones_like(tensor))


def test_deepstack_kwargs_are_consumed_once(monkeypatch):
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.mimo import Qwen3VLMIMOGPTModel
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel as NativeGPT

    calls = []
    monkeypatch.setattr(NativeGPT, "forward", lambda self, **kwargs: calls.append(kwargs) or torch.ones(1))
    language = Qwen3VLMIMOGPTModel.__new__(Qwen3VLMIMOGPTModel)
    torch.nn.Module.__init__(language)
    mask = torch.tensor([[True, False]])
    features = [torch.randn(1, 4)]
    language._mimo_visual_inputs = (mask, features)
    language.forward(input_ids=None)
    language.forward(input_ids=None)
    assert calls[0]["visual_pos_masks"] is mask
    assert calls[0]["deepstack_visual_embeds"] is features
    assert "visual_pos_masks" not in calls[1]
    assert language._mimo_visual_inputs is None
