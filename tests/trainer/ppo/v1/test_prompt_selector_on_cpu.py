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
"""The prompt-selector extension point of the synchronous V1 trainer."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.prompt_selector import PromptSelector, load_prompt_selector
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class RecordingSelector:
    """Choose dataset rows from a script and record every call."""

    def __init__(self, dataset, script=(), prefix=""):
        self.dataset = dataset
        self.script = [list(indices) for indices in script]
        self.prefix = prefix
        self.selected: list[tuple[int, int, str, int | None]] = []
        self.observed: list[tuple[list[tuple[int, list[float]]], int]] = []
        self.saved: list[str] = []
        self.loaded: list[str] = []

    def select(self, num_prompts, *, global_steps, stage, round_index):
        self.selected.append((num_prompts, global_steps, stage, round_index))
        return self.script.pop(0)

    def observe(self, groups, *, global_steps):
        self.observed.append((groups, global_steps))

    def save_checkpoint(self, local_dir):
        self.saved.append(local_dir)

    def load_checkpoint(self, local_dir):
        self.loaded.append(local_dir)


def _dataset():
    return [{"input_ids": torch.tensor([index, index]), "example_id": f"task-{index}"} for index in range(5)]


def test_selector_is_loaded_from_an_importable_class_path():
    config = OmegaConf.create(
        {"class_path": f"{__name__}.RecordingSelector", "kwargs": {"prefix": "p"}, "metric": "seq_reward"}
    )
    dataset = _dataset()
    selector = load_prompt_selector(config, dataset)
    assert isinstance(selector, PromptSelector) and selector.dataset is dataset and selector.prefix == "p"
    assert load_prompt_selector(OmegaConf.create({"class_path": None}), dataset) is None
    with pytest.raises(ValueError, match="module.Class"):
        load_prompt_selector(OmegaConf.create({"class_path": "Selector"}), dataset)


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def _trainer(selector):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {"data": {"train_batch_size": 2, "gen_batch_size": 1, "prompt_selector": {"metric": "seq_reward"}}}
    )
    trainer.train_dataset = _dataset()
    trainer.prompt_selector = selector
    trainer._selected_prompt_indices = {}
    trainer.global_steps = 7
    return trainer


def test_trainer_builds_dispatches_from_selected_rows_and_observes_by_dataset_index():
    selector = RecordingSelector(None, script=[[3, 1], [4]])
    trainer = _trainer(selector)

    first = trainer._next_train_batch(2)
    refill = trainer._next_train_batch(1, stage="active_sampling_refill", round_index=2)

    assert selector.selected == [(2, 7, "initial_batch", None), (1, 7, "active_sampling_refill", 2)]
    assert list(first["example_id"]) == ["task-3", "task-1"]
    assert torch.equal(first["input_ids"], torch.tensor([[3, 3], [1, 1]]))
    assert list(refill["example_id"]) == ["task-4"]
    uids = [str(uid) for uid in first["uid"]] + [str(uid) for uid in refill["uid"]]
    assert len(set(uids)) == 3

    trainer._observe_prompt_groups([(uids[2], [0.0, 1.0]), (uids[0], [1.0, 1.0])])
    assert selector.observed == [([(4, [0.0, 1.0]), (3, [1.0, 1.0])], 7)]


def test_trainer_rejects_a_selector_that_returns_the_wrong_count():
    trainer = _trainer(RecordingSelector(None, script=[[0]]))
    with pytest.raises(RuntimeError, match="returned 1 prompts, expected 2"):
        trainer._next_train_batch(2)


def test_selected_uids_are_numpy_objects_like_dataloader_batches():
    trainer = _trainer(RecordingSelector(None, script=[[0, 2]]))
    batch = trainer._next_train_batch(2)
    assert isinstance(np.asarray(list(batch["uid"])), np.ndarray)
