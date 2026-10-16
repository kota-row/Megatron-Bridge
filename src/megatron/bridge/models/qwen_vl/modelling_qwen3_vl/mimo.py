# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preserve Qwen3-VL DeepStack features across a tensor-only MIMO boundary."""

from typing import Any

import torch
from megatron.core.models.mimo import MimoModel

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel


class Qwen3VLMIMOVisionModel(Qwen3VLVisionModel):
    """Pack primary and DeepStack embeddings along the boundary feature axis."""

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Return one tensor without changing any checkpoint parameter names."""
        primary, deepstack = super().forward(*args, **kwargs)
        return torch.cat([primary, *deepstack], dim=-1)


class Qwen3VLMIMOModel(MimoModel):
    """Pass each microbatch's DeepStack features explicitly to the decoder."""

    def _unpack_image_inputs(
        self,
        input_ids: torch.Tensor,
        input_tensors: dict[str, torch.Tensor] | None,
    ) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
        """Return primary inputs and auxiliary kwargs without retaining tensors."""
        images = input_tensors.get("images") if input_tensors else None
        language_kwargs: dict[str, Any] = {}
        if images is not None and images.numel():
            config = self.mimo_config.language_model_spec.params["config"]
            count = len(config.vision_config.deepstack_visual_indexes)
            expected_width = config.hidden_size * (count + 1)
            if images.shape[-1] != expected_width:
                raise ValueError(f"Qwen3-VL image payload has width {images.shape[-1]}, expected {expected_width}.")
            primary, *deepstack = images.split(config.hidden_size, dim=-1)
            input_tensors = dict(input_tensors or {}, images=primary)
            if deepstack:
                language_kwargs = {
                    "visual_pos_masks": input_ids == self.special_token_ids["images"],
                    "deepstack_visual_embeds": deepstack,
                }
        return input_tensors, language_kwargs

    def _forward_language_module(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        loss_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        input_tensors: dict[str, torch.Tensor] | None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, torch.Tensor | None]:
        """Unpack image features on the first language stage for this call only."""
        language_kwargs = dict(kwargs.pop("language_model_kwargs", None) or {})
        if self.role.is_first_stage("language"):
            input_tensors, visual_kwargs = self._unpack_image_inputs(input_ids, input_tensors)
            conflicts = language_kwargs.keys() & visual_kwargs.keys()
            if conflicts:
                raise ValueError(f"DeepStack arguments are supplied twice: {sorted(conflicts)}")
            language_kwargs.update(visual_kwargs)
        return super()._forward_language_module(
            input_ids,
            position_ids,
            attention_mask,
            loss_mask,
            labels,
            input_tensors,
            *args,
            language_model_kwargs=language_kwargs,
            **kwargs,
        )

    def _forward_all_modules(self, *args: Any, **kwargs: Any) -> tuple[Any, torch.Tensor | None]:
        """Require disjoint vision/language components for DeepStack transport."""
        if self.mimo_config.language_model_spec.params["config"].vision_config.deepstack_visual_indexes:
            raise NotImplementedError("Qwen3-VL DeepStack MIMO requires non-colocated components.")
        return super()._forward_all_modules(*args, **kwargs)
