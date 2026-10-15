# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

from megatron.core import parallel_state as mpu

from megatron.bridge.models.megatron_mimo.build_model import _bridge_parallel_state_globals


def test_bridge_parallel_state_uses_mcore_expert_group_fields(monkeypatch):
    groups = SimpleNamespace(
        tp=object(),
        dp=object(),
        dp_cp=object(),
        pp=object(),
        ep=object(),
        expt_tp=object(),
        tp_ep=object(),
        tp_ep_pp=object(),
        cp=object(),
        mp=object(),
    )
    for global_name in (
        "_GLOBAL_MEMORY_BUFFER",
        "_TENSOR_MODEL_PARALLEL_GROUP",
        "_DATA_PARALLEL_GROUP",
        "_DATA_PARALLEL_GROUP_WITH_CP",
        "_PIPELINE_MODEL_PARALLEL_GROUP",
        "_EXPERT_MODEL_PARALLEL_GROUP",
        "_EXPERT_TENSOR_PARALLEL_GROUP",
        "_EXPERT_DATA_PARALLEL_GROUP",
        "_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP",
        "_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
        "_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP",
        "_CONTEXT_PARALLEL_GROUP",
        "_MODEL_PARALLEL_GROUP",
    ):
        monkeypatch.setattr(mpu, global_name, None)

    _bridge_parallel_state_globals(groups)

    assert mpu._EXPERT_MODEL_PARALLEL_GROUP is groups.ep
    assert mpu._EXPERT_TENSOR_PARALLEL_GROUP is groups.expt_tp
    assert mpu._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is groups.tp_ep
    assert mpu._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP is groups.tp_ep_pp


def test_component_globals_are_available_during_model_construction(monkeypatch):
    """Expert layers query MPU globals while the provider constructs its model."""
    import importlib
    from unittest.mock import MagicMock

    module = importlib.import_module("megatron.bridge.models.megatron_mimo.build_model")
    groups = object()
    infra = SimpleNamespace(module_to_grid_map={}, pg_collections={})
    events = []
    provider = MagicMock()
    provider.provide_distributed_model.side_effect = lambda **kwargs: events.append("construct") or [object()]
    monkeypatch.setattr(module, "_set_per_module_random_seeds", lambda *args, **kwargs: None)
    from megatron.bridge.training import megatron_mimo_parallel_utils as parallel_utils

    provider.build_infra.return_value = infra
    monkeypatch.setattr(parallel_utils, "get_active_module_pg", lambda *args: ("language", groups))
    monkeypatch.setattr(module, "_bridge_parallel_state_globals", lambda pg: events.append(("bridge", pg)))
    monkeypatch.setattr(parallel_utils, "validate_no_stub_ranks", lambda *args: None)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    module.build_megatron_mimo_model(provider)
    assert events == [("bridge", groups), "construct"]
