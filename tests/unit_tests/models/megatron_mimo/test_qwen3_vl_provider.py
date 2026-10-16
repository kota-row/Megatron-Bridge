# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MIMO contracts shared by Qwen3-VL Dense and MoE."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import yaml
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.megatron_mimo.conversion import (
    get_mimo_conversion_spec,
    supports_mimo_conversion,
    validate_route_table,
)
from megatron.bridge.models.megatron_mimo.conversion.mimo_model_io import _clear_derived_spec_fields
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
from megatron.bridge.models.qwen_vl import qwen3_vl_provider as provider_module
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.mimo import Qwen3VLMIMOModel, Qwen3VLMIMOVisionModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel as NativeVision
from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import Qwen3VLBridge, Qwen3VLMoEBridge
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider, Qwen3VLMoEModelProvider
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.instantiate_utils import instantiate
from megatron.bridge.utils.yaml_utils import safe_yaml_representers


pytestmark = pytest.mark.unit


def _make_language_provider(provider_cls=Qwen3VLModelProvider):
    kwargs = dict(
        num_layers=4, hidden_size=128, num_attention_heads=4, num_query_groups=4, kv_channels=32, vocab_size=256
    )
    if provider_cls is Qwen3VLMoEModelProvider:
        kwargs.update(num_moe_experts=8, moe_router_topk=2, moe_ffn_hidden_size=64)
    return provider_cls(**kwargs)


def _make_parallelism_config():
    return MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(tensor_model_parallel_size=1),
            "images": ModuleParallelismConfig(tensor_model_parallel_size=1, rank_offset=1),
        }
    )


@pytest.fixture(
    params=[(Qwen3VLModelProvider, Qwen3VLBridge), (Qwen3VLMoEModelProvider, Qwen3VLMoEBridge)], ids=["dense", "moe"]
)
def model_pair(request):
    provider_cls, bridge_cls = request.param
    return _make_language_provider(provider_cls), bridge_cls


@pytest.fixture
def dense_provider():
    # Dense and MoE install the same DeepStack validation helper.
    return _make_language_provider()


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
    assert spec.module is Qwen3VLMIMOVisionModel
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
    standard.mtp_num_layers = 2
    parallelism = _make_parallelism_config()
    hf_source = object()
    provider, routes = get_mimo_conversion_spec(bridge_cls)(source, hf_source, parallelism)
    assert isinstance(provider, MegatronMIMOProvider)
    assert provider.standard_provider is standard
    assert standard.mtp_num_layers is None
    assert provider.megatron_mimo_parallelism_config is parallelism
    assert provider.language_model_spec.module is Qwen3VLGPTModel
    assert provider.special_token_ids == {"images": standard.image_token_id}
    assert (
        provider.modality_submodules_spec["images"].submodules["encoders"]["qwen_visual"].module
        is Qwen3VLMIMOVisionModel
    )
    assert {(r.source_prefix, r.target_module_path) for r in routes} == {
        ("language_model.", "language_model"),
        ("vision_model.", "modality_submodules.images.encoders.qwen_visual"),
    }
    source.provider_bridge.assert_called_once_with(hf_source)

    validate_route_table(
        routes, parallelism_config=parallelism, modality_submodules_spec=provider.modality_submodules_spec
    )


def test_yaml_round_trip_rebuilds_specs_after_clear(model_pair, tmp_path):
    standard, _ = model_pair
    provider = MegatronMIMOProvider.from_standard_provider(
        standard_provider=standard, megatron_mimo_parallelism_config=_make_parallelism_config()
    )
    _clear_derived_spec_fields(provider)
    model_dict = ConfigContainer._convert_value_to_dict(provider)
    assert model_dict["language_model_spec"] is None
    assert model_dict["modality_submodules_spec"] == {}
    path = tmp_path / "run_config.yaml"
    with safe_yaml_representers():
        path.write_text(yaml.safe_dump({"model": model_dict}))
    loaded = instantiate(yaml.safe_load(path.read_text())["model"])
    assert isinstance(loaded, MegatronMIMOProvider)
    assert type(loaded.standard_provider) is type(standard)
    assert loaded.language_model_spec.module is Qwen3VLGPTModel
    assert loaded.standard_provider.mimo_model_class is Qwen3VLMIMOModel
    assert (
        loaded.modality_submodules_spec["images"].submodules["encoders"]["qwen_visual"].module
        is Qwen3VLMIMOVisionModel
    )
    assert loaded.special_token_ids == standard.special_token_ids


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tensor_model_parallel_size", 2),
        ("context_parallel_size", 2),
        ("sequence_parallel", True),
    ],
)
def test_deepstack_rejects_unimplemented_sharding(dense_provider, field, value):
    provider = dense_provider
    assert hasattr(provider, field)
    setattr(provider, field, value)
    with pytest.raises(ValueError, match="DeepStack MIMO"):
        provider.build_language_model_spec()


@pytest.mark.parametrize(
    ("layers", "first_layers", "vp_size", "embedding_split", "accepted"),
    [
        (6, None, None, False, True),  # Exactly three layers on the first stage.
        (4, None, None, False, False),
        (8, 3, None, False, True),
        (8, 2, None, False, False),
        (12, None, 2, False, True),  # Count the first virtual chunk, not the whole rank.
        (8, None, 2, False, False),
        (7, None, None, True, True),
        (5, None, None, True, False),  # An embedding slot is not a decoder layer.
    ],
)
def test_deepstack_pipeline_first_chunk_capacity(
    dense_provider, monkeypatch, layers, first_layers, vp_size, embedding_split, accepted
):
    provider = dense_provider
    provider.num_layers = layers
    provider.pipeline_model_parallel_size = 2
    provider.num_layers_in_first_pipeline_stage = first_layers
    provider.virtual_pipeline_model_parallel_size = vp_size
    provider.account_for_embedding_in_pipeline_split = embedding_split
    monkeypatch.setattr(provider, "build_language_spec", MagicMock(return_value=ModuleSpec(module=object)))
    if accepted:
        assert provider.build_language_model_spec(pp_rank=1).module is Qwen3VLGPTModel
    else:
        with pytest.raises(ValueError, match="at least 3 decoder layers.*got 2"):
            provider.build_language_model_spec(pp_rank=1)


@pytest.mark.parametrize("first_layers", [2, 3])
def test_deepstack_custom_pipeline_layout(dense_provider, monkeypatch, first_layers):
    from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

    provider = dense_provider
    provider.num_layers = 8
    provider.pipeline_model_parallel_size = 2
    provider.pipeline_model_parallel_layout = PipelineParallelLayerLayout(
        f"Et*{first_layers}|t*{8 - first_layers}L", pipeline_model_parallel_size=2
    )
    monkeypatch.setattr(provider, "build_language_spec", MagicMock(return_value=ModuleSpec(module=object)))
    if first_layers == 3:
        assert provider.build_language_model_spec(pp_rank=1).module is Qwen3VLGPTModel
    else:
        with pytest.raises(ValueError, match="at least 3 decoder layers.*got 2"):
            provider.build_language_model_spec(pp_rank=1)


@pytest.fixture
def deepstack_model(dense_provider):
    model = Qwen3VLMIMOModel.__new__(Qwen3VLMIMOModel)
    torch.nn.Module.__init__(model)
    model.special_token_ids = dense_provider.special_token_ids
    model.role = SimpleNamespace(is_first_stage=lambda name: True)
    model.mimo_config = SimpleNamespace(
        language_model_spec=ModuleSpec(module=Qwen3VLGPTModel, params={"config": dense_provider})
    )
    return model


def test_deepstack_boundary_preserves_features_and_microbatch_gradients(deepstack_model, monkeypatch):
    from megatron.core.models.mimo import MimoModel

    model = deepstack_model
    config = model.mimo_config.language_model_spec.params["config"]
    ids = torch.tensor([[3, config.image_token_id, config.image_token_id, 4]])
    captured = []

    def language_forward(self, ids, positions, attention, loss_mask, labels, inputs, **kwargs):
        extra = kwargs["language_model_kwargs"]
        captured.append((inputs, extra))
        if inputs is None:
            return torch.zeros(()), loss_mask
        return inputs["images"].square().sum() + sum(
            value.square().sum() for value in extra["deepstack_visual_embeds"]
        ), loss_mask

    monkeypatch.setattr(MimoModel, "_forward_language_module", language_forward)
    encoder = Qwen3VLMIMOVisionModel.__new__(Qwen3VLMIMOVisionModel)
    torch.nn.Module.__init__(encoder)
    tensors, outputs = [], []
    caller_kwargs = {"custom_option": 7}
    for _ in range(2):
        primary = torch.randn(2, config.hidden_size, requires_grad=True)
        auxiliary = [
            torch.randn_like(primary, requires_grad=True) for _ in config.vision_config.deepstack_visual_indexes
        ]
        monkeypatch.setattr(NativeVision, "forward", lambda *args, **kwargs: (primary, auxiliary))
        inputs = {"images": encoder()}
        payload = inputs["images"]
        output, _ = model._forward_language_module(
            ids, None, None, None, None, inputs, language_model_kwargs=caller_kwargs
        )
        forwarded, extra = captured[-1]
        assert inputs["images"] is payload
        torch.testing.assert_close(forwarded["images"], primary)
        assert torch.equal(extra["visual_pos_masks"], ids == config.image_token_id)
        for received, expected in zip(extra["deepstack_visual_embeds"], auxiliary, strict=True):
            torch.testing.assert_close(received, expected)
        assert extra["custom_option"] == 7
        tensors.extend([primary, *auxiliary])
        outputs.append(output)
    # A text microbatch on the same instance must not reuse image features.
    model._forward_language_module(torch.tensor([[3, 4]]), None, None, None, None, None)
    assert captured[-1][1] == {}
    assert caller_kwargs == {"custom_option": 7}
    for output in reversed(outputs):
        output.backward()
    for tensor in tensors:
        torch.testing.assert_close(tensor.grad, 2 * tensor)


def test_deepstack_rejects_duplicate_features(deepstack_model):
    config = deepstack_model.mimo_config.language_model_spec.params["config"]
    width = config.hidden_size * (1 + len(config.vision_config.deepstack_visual_indexes))
    with pytest.raises(ValueError, match="supplied twice"):
        deepstack_model._forward_language_module(
            torch.tensor([[config.image_token_id]]),
            None,
            None,
            None,
            None,
            {"images": torch.ones(1, width)},
            language_model_kwargs={"deepstack_visual_embeds": []},
        )


def test_deepstack_rejects_colocated_components(deepstack_model):
    with pytest.raises(NotImplementedError, match="non-colocated"):
        deepstack_model._forward_all_modules()


def test_deepstack_rejects_incorrect_payload_width(deepstack_model):
    with pytest.raises(ValueError, match="payload has width"):
        deepstack_model._unpack_image_inputs(torch.tensor([[3, 4]]), {"images": torch.ones(2, 4)})
