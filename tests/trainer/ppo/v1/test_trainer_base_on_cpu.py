# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from unittest.mock import MagicMock, patch

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorData, NonTensorStack, TensorDict
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _CustomSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _trainer_with_filter_groups(filter_groups: dict, trainer_mode: str = "sync") -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = trainer_mode
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": filter_groups},
            "data": {"train_batch_size": 64, "gen_batch_size": 8},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "trainer": {
                "v1": {
                    trainer_mode: {},
                    "sampler": {
                        "custom_sampler": None,
                        "max_off_policy_threshold": 1,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                    },
                }
            },
        }
    )
    return trainer


def test_builtin_sampler_class_follows_trainer_mode():
    sync_sampler = _trainer_with_filter_groups({"enable": False}, trainer_mode="sync")._build_replay_buffer()
    async_samplers = [
        _trainer_with_filter_groups({"enable": True, "metric": "acc"}, trainer_mode=mode)._build_replay_buffer()
        for mode in ("colocate_async", "separate_async")
    ]

    assert type(sync_sampler) is ReplayBuffer
    assert all(type(sampler) is ReplayBufferAsync for sampler in async_samplers)
    assert all(sampler.filter_groups_metric == "acc" for sampler in async_samplers)
    assert all(sampler.train_batch_size is None for sampler in async_samplers)
    assert all(sampler.gen_batch_size is None for sampler in async_samplers)


def test_custom_sampler_skips_builtin_filter_groups_validation():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})
    trainer.config.trainer.v1.sampler.custom_sampler = {"path": "custom.py", "name": "CustomSampler"}

    with (
        patch("verl.trainer.ppo.v1.trainer_base.load_extern_type", return_value=_CustomSampler),
        patch.object(trainer, "_resolve_filter_groups_metric") as resolve_filter_groups_metric,
    ):
        sampler = trainer._build_replay_buffer()

    resolve_filter_groups_metric.assert_not_called()
    assert isinstance(sampler, _CustomSampler)
    assert "filter_groups_metric" not in sampler.kwargs
    assert "train_batch_size" not in sampler.kwargs
    assert "gen_batch_size" not in sampler.kwargs
    assert "max_inflight_gen_batches" not in sampler.kwargs
    assert "max_num_gen_batches" not in sampler.kwargs
    assert "sync_refill_failed_groups" not in sampler.kwargs


def test_builtin_filter_groups_uses_default_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})

    sampler = trainer._build_replay_buffer()

    assert sampler.filter_groups_metric == "acc"
    assert sampler.train_batch_size == 64
    assert sampler.gen_batch_size == 1
    assert sampler.max_inflight_gen_batches == 1
    assert sampler.max_num_gen_batches == 0


def test_builtin_filter_groups_forwards_configured_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_inflight_gen_batches": 3})

    sampler = trainer._build_replay_buffer()

    assert sampler.max_inflight_gen_batches == 3


def test_builtin_sync_failure_refill_forces_single_prompt_generation():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True

    sampler = trainer._build_replay_buffer()

    assert sampler.sync_refill_failed_groups is True
    assert sampler.gen_batch_size == 1


def test_sync_failure_refill_overrides_dataloader_generation_batch_size():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True
    trainer.config.data.update(
        {
            "train_files": [],
            "val_files": [],
            "train_max_samples": -1,
            "val_max_samples": -1,
            "dataloader_num_workers": 0,
            "val_batch_size": 1,
            "validation_shuffle": False,
        }
    )
    trainer.config.trainer.total_epochs = 1
    trainer.config.trainer.total_training_steps = None
    trainer.parameter_sync_step = 1
    trainer.tokenizer = None
    trainer.processor = None

    with (
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_dataset", side_effect=[[{}, {}], [{}]]),
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_sampler", return_value=None),
        patch("verl.trainer.ppo.v1.trainer_base.StatefulDataLoader") as dataloader,
        patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning,
    ):
        trainer._init_dataloader()

    assert trainer.config.data.gen_batch_size == 1
    assert dataloader.call_args_list[0].kwargs["batch_size"] == 1
    warning.assert_any_call("data.gen_batch_size=8 is overridden to 1.")


def test_builtin_filter_groups_forwards_total_generation_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_num_gen_batches": 10})

    sampler = trainer._build_replay_buffer()

    assert sampler.max_num_gen_batches == 10


def test_sampo_advantage_fetches_and_forwards_rollout_metadata():
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "sampo",
                "gamma": 1.0,
                "lam": 1.0,
                "use_kl_in_reward": False,
                "norm_adv_by_std_in_grpo": False,
            },
            "actor_rollout_ref": {"rollout": {"n": 2}},
        }
    )
    batch = KVBatchMeta(
        partition_id="train",
        keys=["group_session-a_0_0", "group_session-b_1_0"],
        tags=[{}, {}],
    )
    response_mask = torch.nested.as_nested_tensor(
        [torch.ones(2, dtype=torch.int64), torch.ones(2, dtype=torch.int64)],
        layout=torch.jagged,
    )
    rm_scores = torch.nested.as_nested_tensor(
        [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 2.0])],
        layout=torch.jagged,
    )
    transfer_data = TensorDict(
        {
            # The replay-buffer key, not a potentially trajectory-local stored uid,
            # owns SAMPO prompt-group identity.
            "uid": NonTensorStack.from_list([NonTensorData("trajectory-a"), NonTensorData("trajectory-b")]),
            "response_mask": response_mask,
            "rm_scores": rm_scores,
            "extra_fields": NonTensorStack.from_list(
                [
                    NonTensorData(
                        {
                            "sampo_turn_spans": [[0, 2]],
                            "sampo_anchor_state_keys": ["anchor"],
                            "sampo_step_rewards": [1.0],
                        }
                    ),
                    NonTensorData(
                        {
                            "sampo_turn_spans": [[0, 2]],
                            "sampo_anchor_state_keys": ["anchor"],
                            "sampo_step_rewards": [2.0],
                        }
                    ),
                ]
            ),
        },
        batch_size=[2],
    )
    computed = MagicMock()
    computed.batch = {
        "advantages": torch.ones(2, 2),
        "returns": torch.ones(2, 2),
    }

    with (
        patch("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_get", return_value=transfer_data) as get,
        patch("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_put", return_value=batch),
        patch(
            "verl.trainer.ppo.v1.trainer_base.compute_advantage_for_multi_trajectories",
            return_value=computed,
        ) as compute,
    ):
        trainer._compute_advantage(batch, metrics={})

    selected_fields = get.call_args.kwargs["select_fields"]
    assert selected_fields[-1] == "extra_fields"
    forwarded = compute.call_args.args[0].non_tensor_batch
    assert forwarded["uid"].tolist() == ["group", "group"]
    assert forwarded["sampo_turn_spans"].tolist() == [[[0, 2]], [[0, 2]]]
    assert forwarded["sampo_anchor_state_keys"].tolist() == [["anchor"], ["anchor"]]
    assert forwarded["sampo_step_rewards"].tolist() == [[1.0], [2.0]]
    assert all(value.dtype == np.dtype("O") for value in forwarded.values())
