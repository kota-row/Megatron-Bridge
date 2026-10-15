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

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel


class Qwen3VLMIMOVisionModel(Qwen3VLVisionModel):
    """Pack primary and DeepStack embeddings along the boundary feature axis."""

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Return one tensor without changing any checkpoint parameter names."""
        primary, deepstack = super().forward(*args, **kwargs)
        return torch.cat([primary, *deepstack], dim=-1)


class Qwen3VLMIMOGPTModel(Qwen3VLGPTModel):
    """Receive DeepStack kwargs from the MIMO embedding assembly step."""

    _mimo_visual_inputs: tuple[torch.Tensor, list[torch.Tensor]] | None = None

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Consume this microbatch's visual features before running the decoder."""
        visual_inputs = self._mimo_visual_inputs
        self._mimo_visual_inputs = None
        if visual_inputs is not None:
            kwargs["visual_pos_masks"], kwargs["deepstack_visual_embeds"] = visual_inputs
        return super().forward(*args, **kwargs)


class Qwen3VLMIMOModel(MimoModel):
    """Split the image boundary payload before aligning primary embeddings."""

    def align_embeddings_by_token_positions(
        self,
        modality_embeddings: dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        special_token_ids: dict[str, int],
        modality_token_indices: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Route auxiliary features to Qwen's decoder with their visual mask."""
        images = modality_embeddings.get("images")
        visual_inputs = None
        if images is not None and images.numel():
            config = self.mimo_config.language_model_spec.params["config"]
            count = len(config.vision_config.deepstack_visual_indexes)
            expected_width = config.hidden_size * (count + 1)
            if images.shape[-1] != expected_width:
                raise ValueError(f"Qwen3-VL image payload has width {images.shape[-1]}, expected {expected_width}.")
            primary, *deepstack = images.split(config.hidden_size, dim=-1)
            modality_embeddings = dict(modality_embeddings, images=primary)
            if deepstack:
                visual_inputs = (input_ids == special_token_ids["images"], deepstack)
        output = super().align_embeddings_by_token_positions(
            modality_embeddings, input_ids, special_token_ids, modality_token_indices
        )
        language_model = self.language_model
        while hasattr(language_model, "module"):
            language_model = language_model.module
        language_model._mimo_visual_inputs = visual_inputs
        return output
