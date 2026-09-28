# Copyright 2026 CarbonTeq
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
"""Extension point that lets a caller choose the prompts of every rollout dispatch.

A prompt selector replaces the training dataloader as the source of prompts in the synchronous V1
trainer. It is consulted once per dispatch: the step's first batch (``stage="initial_batch"``) or,
with ``algorithm.active_sampling``, every round (``stage="active_sampling_refill"``, rounds counted
from 1). After a dispatch's groups finish, it observes each finished group's per-trajectory metric
values in dispatch order, so a later round can depend on earlier ones at the same policy weights.
Its state is saved into and restored from every ``global_step_*`` checkpoint folder.

Configure it with ``data.prompt_selector.class_path`` (an importable ``module.Class``),
``data.prompt_selector.kwargs`` and ``data.prompt_selector.metric`` (the ``reward_extra_info``
value observed; default ``seq_reward``). The class is constructed as ``cls(dataset=train_dataset,
**kwargs)``.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PromptSelector(Protocol):
    def select(self, num_prompts: int, *, global_steps: int, stage: str, round_index: int | None) -> list[int]:
        """Return ``num_prompts`` training-dataset indices for the next dispatch."""

    def observe(self, groups: list[tuple[int, list[float]]], *, global_steps: int) -> None:
        """Receive ``(dataset_index, metric values)`` for every finished group of a dispatch."""

    def save_checkpoint(self, local_dir: str) -> None:
        """Write selector state into a ``global_step_*`` checkpoint folder."""

    def load_checkpoint(self, local_dir: str) -> None:
        """Restore selector state from a ``global_step_*`` checkpoint folder."""


def load_prompt_selector(selector_config: Any, dataset: Any) -> PromptSelector | None:
    """Instantiate the configured selector, or return ``None`` when none is configured."""
    if selector_config is None:
        return None
    class_path = selector_config.get("class_path", None)
    if not class_path:
        return None
    module_name, _, class_name = str(class_path).rpartition(".")
    if not module_name:
        raise ValueError(f"data.prompt_selector.class_path must be 'module.Class', got {class_path!r}")
    selector_cls = getattr(importlib.import_module(module_name), class_name)
    kwargs = selector_config.get("kwargs", None) or {}
    selector = selector_cls(dataset=dataset, **dict(kwargs))
    if not isinstance(selector, PromptSelector):
        raise TypeError(f"{class_path} does not implement select/observe/save_checkpoint/load_checkpoint")
    return selector
