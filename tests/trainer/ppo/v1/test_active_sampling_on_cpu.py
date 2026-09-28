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
"""Round-based active sampling (TRL GRPO semantics) in the synchronous V1 replay buffer.

Expected round sizes and metrics are TRL 1.12.0.post11 ``_prepare_active_sampling_inputs``
results for the same keep decisions; the consumer (Posttrain) additionally runs both
implementations side by side.
"""

import uuid

import pytest
import torch
import transfer_queue as tq
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.replay_buffer import (
    ActiveSamplingReplayBuffer,
    ActiveSamplingRounds,
    active_sampling_round_capacity_error,
)
from verl.trainer.ppo.v1.trainer_base import PPOTrainer

POLL_INTERVAL = 0.02
GROUP = 2


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


@pytest.fixture
def partition_id():
    return f"active-{uuid.uuid4().hex}"


def _drive(rounds: ActiveSamplingRounds, keep: list[bool]) -> list[int]:
    """Run the round arithmetic over candidate keep decisions; return retained candidate indices."""
    retained: list[int] = []
    while (plan := rounds.next_round()) is not None:
        requested, size = plan
        start = rounds.cursor
        kept = [index for index in range(start, start + size) if keep[index]]
        retained.extend(kept)
        rounds.record(requested, size, len(kept))
    return retained


def test_rounds_refill_exactly_the_missing_groups_without_oversampling():
    rounds = ActiveSamplingRounds(target=3, max_rounds=4)
    retained = _drive(rounds, [True, False, False, False, True, True, True])

    assert rounds.round_log == [(3, 3, 1), (2, 2, 1), (1, 1, 1)]
    assert retained == [0, 4, 5]
    assert rounds.metrics(GROUP) == {
        "active_sampling/generation_rounds": 3,
        "active_sampling/retained_fraction": 3 / 6,
        "active_sampling/generated_rows": 12,
        "active_sampling/candidate_groups_reserved": 24,
        "active_sampling/candidate_groups_generated": 12,
        "active_sampling/candidate_groups_retained": 6,
        "active_sampling/candidate_groups_unused": 12,
    }


def test_rounds_oversample_first_round_and_cap_refills_at_its_size():
    rounds = ActiveSamplingRounds(target=4, max_rounds=3, oversample=2, oversample_refill=5)
    keep = [True, False, False, True, False, False] + [True, True, True, True, False]
    retained = _drive(rounds, keep + [False] * 12)

    # Round 2 wants 2 missing + 5 extra = 7, capped at the first round's 6.
    assert rounds.round_log == [(6, 6, 2), (6, 6, 4)]
    assert retained[:4] == [0, 3, 6, 7]
    metrics = rounds.metrics(GROUP)
    assert metrics["active_sampling/round_1_requested_groups"] == 6
    assert metrics["active_sampling/round_2_generated_groups"] == 6
    assert metrics["active_sampling/round_2_retained_groups"] == 4
    assert metrics["active_sampling/oversampled_groups"] == (6 - 4) + (6 - 2)
    assert metrics["active_sampling/discarded_groups"] == 2


def test_rounds_cut_extra_groups_to_the_pool_but_fail_when_missing_groups_do_not_fit():
    rounds = ActiveSamplingRounds(target=2, max_rounds=2, oversample=1, oversample_refill=1)
    assert rounds.next_round() == (3, 3)
    rounds.record(3, 3, 1)
    # One group missing plus one extra requested; one candidate left in the pool of four.
    assert rounds.next_round() == (2, 1)

    exhausted = ActiveSamplingRounds(target=2, max_rounds=2, oversample=2)
    exhausted.record(4, 4, 0)
    with pytest.raises(RuntimeError, match="exhausted its bounded candidate pool: 2 groups are missing"):
        exhausted.next_round()


def test_rounds_report_a_batch_that_ran_out_of_rounds():
    rounds = ActiveSamplingRounds(target=2, max_rounds=2)
    _drive(rounds, [False] * 4)
    with pytest.raises(RuntimeError, match="exhausted 2 generation rounds.*retained 0 of 2"):
        rounds.require_full()


def test_capacity_error_checks_only_oversampled_first_rounds():
    kwargs = dict(train_batch_size=4, rollout_n=4, engine_max_num_seqs=16, max_concurrent_episodes=32, worker_slots=24)
    assert active_sampling_round_capacity_error(oversample=0, **kwargs) is None
    error = active_sampling_round_capacity_error(oversample=2, **kwargs)
    assert error is not None and "needs 24 concurrent episodes" in error and "at most 16 sequences" in error
    assert "max_concurrent_episodes" not in error


class _Dispatcher:
    """Complete each dispatched round immediately with scripted per-session rewards."""

    def __init__(self, partition_id: str, groups: list[list[float] | str]):
        self.partition_id = partition_id
        self.groups = list(groups)
        self.calls: list[int] = []
        self.rounds: list[int | None] = []
        self.uids: list[str] = []

    def __call__(self, count: int, round_index: int | None = None) -> list[str]:
        self.calls.append(count)
        self.rounds.append(round_index)
        uids = []
        for _ in range(count):
            spec = self.groups.pop(0)
            uid = uuid.uuid4().hex
            if spec != "failure":
                for session, reward in enumerate(spec):
                    tq.kv_put(
                        key=f"{uid}_{session}_0",
                        partition_id=self.partition_id,
                        fields={
                            "input_ids": torch.tensor([1, 2, 3]),
                            "extra_fields": {"reward_extra_info": {"seq_reward": float(reward)}},
                        },
                        tag={"is_prompt": False, "seq_len": 3, "global_steps": 1},
                    )
            tq.kv_put(
                key=uid,
                partition_id=self.partition_id,
                tag={"is_prompt": True, "status": "failure" if spec == "failure" else "finished", "global_steps": 1},
            )
            uids.append(uid)
        self.uids.extend(uids)
        return uids


def _buffer(dispatcher, observe_fn=None, **active) -> ActiveSamplingReplayBuffer:
    return ActiveSamplingReplayBuffer(
        trainer_mode="sync",
        trainer_config={},
        max_off_policy_threshold=1,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=POLL_INTERVAL,
        refill_fn=lambda count: count,
        train_batch_size=2,
        gen_batch_size=1,
        dispatch_fn=dispatcher,
        observe_fn=observe_fn,
        **active,
    )


def test_buffer_keeps_first_informative_groups_in_dispatch_order_and_clears_the_rest(tq_init, partition_id):
    groups = [[0.0, 0.0], [0.0, 1.0], "failure", [1.0, 1.0], [0.5, 0.0], [1.0, 0.0], [0.0, 0.0]]
    dispatcher = _Dispatcher(partition_id, groups)
    buffer = _buffer(dispatcher, active_max_rounds=4, active_oversample=1, active_oversample_refill=1)
    try:
        batch, metrics = buffer.sample(global_steps=1, partition_id=partition_id, batch_size=2)
        # Round 1: 2 + 1 = 3 groups keep one; round 2: 1 missing + 1 extra = 2 groups keep one.
        assert dispatcher.calls == [3, 2]
        selected = {key.split("_")[0] for key in batch.keys}
        assert selected == {dispatcher.uids[1], dispatcher.uids[4]}
        assert len(batch.keys) == 4
        assert metrics["active_sampling/generation_rounds"] == 2
        assert metrics["active_sampling/candidate_groups_generated"] == 10
        assert metrics["active_sampling/round_1_retained_groups"] == 1
        assert metrics["active_sampling/oversampled_groups"] == 2
        assert metrics["active_sampling/discarded_groups"] == 0
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        for rejected in (dispatcher.uids[0], dispatcher.uids[2], dispatcher.uids[3]):
            assert not any(key.split("_")[0] == rejected for key in remaining)
    finally:
        keys = list(tq.kv_list(partition_id=partition_id).get(partition_id, {}).keys())
        if keys:
            tq.kv_clear(keys=keys, partition_id=partition_id)


def test_buffer_discards_surplus_informative_groups(tq_init, partition_id):
    dispatcher = _Dispatcher(partition_id, [[0.0, 1.0]] * 3)
    buffer = _buffer(dispatcher, active_max_rounds=2, active_oversample=1)
    try:
        batch, metrics = buffer.sample(global_steps=1, partition_id=partition_id, batch_size=2)
        assert dispatcher.calls == [3]
        assert {key.split("_")[0] for key in batch.keys} == set(dispatcher.uids[:2])
        assert metrics["active_sampling/discarded_groups"] == 1
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert not any(key.split("_")[0] == dispatcher.uids[2] for key in remaining)
    finally:
        keys = list(tq.kv_list(partition_id=partition_id).get(partition_id, {}).keys())
        if keys:
            tq.kv_clear(keys=keys, partition_id=partition_id)


def test_buffer_fails_after_its_bounded_rounds(tq_init, partition_id):
    dispatcher = _Dispatcher(partition_id, [[1.0, 1.0]] * 4)
    buffer = _buffer(dispatcher, active_max_rounds=2)
    try:
        with pytest.raises(RuntimeError, match="exhausted 2 generation rounds"):
            buffer.sample(global_steps=1, partition_id=partition_id, batch_size=2)
        assert dispatcher.calls == [2, 2]
    finally:
        keys = list(tq.kv_list(partition_id=partition_id).get(partition_id, {}).keys())
        if keys:
            tq.kv_clear(keys=keys, partition_id=partition_id)


def test_buffer_rejects_streaming_filter_groups_and_async_modes():
    with pytest.raises(ValueError, match="separate refill strategies"):
        ActiveSamplingReplayBuffer(
            trainer_mode="sync",
            trainer_config={},
            max_off_policy_threshold=1,
            max_off_policy_strategy="drop",
            sampler_kwargs={},
            refill_fn=lambda count: count,
            filter_groups_metric="seq_reward",
            dispatch_fn=lambda count: [],
        )


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def _trainer(active_sampling: dict, *, max_num_seqs: int = 64) -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = "sync"
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": {"enable": False}, "active_sampling": active_sampling},
            "data": {"train_batch_size": 4},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "actor_rollout_ref": {
                "rollout": {"n": 4, "max_num_seqs": max_num_seqs, "tensor_model_parallel_size": 1, "agent": {}}
            },
            "trainer": {
                "n_gpus_per_node": 1,
                "nnodes": 1,
                "v1": {
                    "sync": {},
                    "sampler": {
                        "custom_sampler": None,
                        "max_off_policy_threshold": 1,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                    },
                },
            },
        }
    )
    return trainer


def test_trainer_builds_the_active_sampling_buffer_from_config():
    trainer = _trainer({"enable": True, "max_candidate_batches": 6, "oversample": 2, "oversample_refill": 1})
    buffer = trainer._build_replay_buffer()

    assert type(buffer) is ActiveSamplingReplayBuffer
    assert (buffer.active_max_rounds, buffer.active_oversample, buffer.active_oversample_refill) == (6, 2, 1)
    assert buffer.gen_batch_size == 1
    assert buffer.dispatch_fn == trainer._dispatch_prompts


def test_trainer_rejects_an_oversampled_first_round_above_engine_capacity():
    trainer = _trainer({"enable": True, "oversample": 2}, max_num_seqs=16)
    with pytest.raises(ValueError, match="needs 24 concurrent episodes"):
        trainer._build_replay_buffer()


def test_buffer_numbers_rounds_and_reports_every_finished_group_in_dispatch_order(tq_init, partition_id):
    groups = [[0.0, 0.0], "failure", [0.0, 1.0], [1.0, 1.0], [0.25, 0.75]]
    dispatcher = _Dispatcher(partition_id, groups)
    observed: list[list[tuple[str, list[float]]]] = []
    buffer = _buffer(dispatcher, observe_fn=observed.append, active_max_rounds=3)
    try:
        buffer.sample(global_steps=1, partition_id=partition_id, batch_size=2)
        assert dispatcher.rounds == [1, 2, 3]
        uids = dispatcher.uids
        # Failed groups are not observed; kept and rejected finished groups are.
        assert observed == [
            [(uids[0], [0.0, 0.0])],
            [(uids[2], [0.0, 1.0]), (uids[3], [1.0, 1.0])],
            [(uids[4], [0.25, 0.75])],
        ]
    finally:
        keys = list(tq.kv_list(partition_id=partition_id).get(partition_id, {}).keys())
        if keys:
            tq.kv_clear(keys=keys, partition_id=partition_id)


def test_buffer_ignores_non_finite_values_like_trl_nan_rewards(tq_init, partition_id):
    nan = float("nan")
    # Group 0 keeps spread only through an excluded (NaN) trajectory; group 1 has spread among finite values.
    dispatcher = _Dispatcher(partition_id, [[0.0, 0.0, nan], [0.0, 1.0, nan], [1.0, nan, nan], [0.5, 0.25, 0.5]])
    buffer = _buffer(dispatcher, active_max_rounds=3)
    try:
        batch, _ = buffer.sample(global_steps=1, partition_id=partition_id, batch_size=2)
        # Group 2 has one finite value (no spread), so a third round is needed.
        assert dispatcher.calls == [2, 1, 1]
        assert {key.split("_")[0] for key in batch.keys} == {dispatcher.uids[1], dispatcher.uids[3]}
    finally:
        keys = list(tq.kv_list(partition_id=partition_id).get(partition_id, {}).keys())
        if keys:
            tq.kv_clear(keys=keys, partition_id=partition_id)
