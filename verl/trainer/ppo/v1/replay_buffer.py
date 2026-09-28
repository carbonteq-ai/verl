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
import logging
import os
import time
from collections import Counter, defaultdict

import numpy as np
import transfer_queue as tq
from omegaconf import DictConfig
from transfer_queue import KVBatchMeta

from verl.utils.skip import SkipManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS = int(os.getenv("VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS", "60"))

DAPO_FILTERED_REWARD_COUNTS_KEY = "_dapo_filtered_reward_counts"


def _accumulate_eviction_metrics(acc: dict, new: dict, stale_count: int) -> None:
    """Merge one poll iteration's eviction metrics into ``acc`` in place.

    ``stale_count`` weights the staleness mean so it stays a true per-sample average across iterations.
    """
    stale_count_key = next((k for k in new if k.endswith("/off_policy/evicted_samples")), None)
    prev_stale_total = acc.get(stale_count_key, 0) if stale_count_key else 0

    for key, value in new.items():
        if key.endswith("/evicted_samples_staleness/mean"):
            denom = prev_stale_total + stale_count
            acc[key] = (acc.get(key, 0.0) * prev_stale_total + value * stale_count) / denom if denom else value
        elif key.endswith("/evicted_samples_staleness/max"):
            acc[key] = max(acc.get(key, value), value)
        elif key.endswith("/evicted_samples_staleness/min"):
            acc[key] = min(acc.get(key, value), value)
        elif key == DAPO_FILTERED_REWARD_COUNTS_KEY:
            # Dict-valued diagnostic: merge {metric_value: count} across poll iterations.
            merged = Counter(acc.get(key, {}))
            merged.update(value)
            acc[key] = dict(merged)
        else:
            acc[key] = acc.get(key, 0) + value


# TODO: Pass custom sampler to TransferQueue:
# https://github.com/Ascend/TransferQueue/blob/main/tutorial/05_custom_sampler.py


class ReplayBuffer:
    """ReplayBuffer is used by trainer to sample trajectories produced during rollout.

    We use [TransferQueue](https://github.com/Ascend/TransferQueue) as kv store to store trajectories.

    ### [Trajectories storage format]
    The key format is `{uid}_{session_id}_{index}`, where:
    - uid: Auto generated unique id when prompt is sampled from dataset.
    - session_id: Session id for GRPO group sampling: [0, n).
    - index: Index of output trajectory in a session.

    There're two types of data associated with each key: tag and value. The tag are arbitrary metadata:
    `{"status": "running", ...}` used to track the status of the trajectory.

    The value is a dictionary containing the following fields:
    - messages/datasource/reward_model/...: fields from dataset.
    - prompt_ids/response_ids/response_mask/...: fields from AgentLoopOutput.

    TransferQueue store tag and value separately, the tag are stored in meta server, while the value is stored
    in storage units.

    ### [GRPO group sampling control]
    Except trajectories, we also store raw prompts in TransferQueue with key `{uid}`, with `status` tag to track
    status of GRPO group sampling.
    - pending: the prompt is sampled from dataset but its sessions are not yet started.
    - running: all sessions of the prompt are running.
    - finished: all sessions of the prompt are finished without error.
    - failure: all sessions of the prompt are finished, but at least one session failed.
    Only prompts with status `finished` or `failure` enter terminal-group handling.

    ### [Terminal-group eviction/refill matrix]
    ``drop`` means off-policy staleness dropping. Both off-policy strategies (``drop`` and the dropless
    ``wait``, which blocks until stale in-flight prompts finish) are only for async trainers; sync sampling
    is on-policy, so ``max_off_policy_strategy`` is a NO-OP there.
    ``DAPO`` means filtering groups whose configured reward metric is identical across all trajectories,
             for async reward-computation path.
    ``failure`` is the group status described above.
    The matrix applies to the training partition, in which ``k`` is the number of prompts to evict.
    Validation treats all terminal groups as sampleable

    |   trainer mode   |   drop   |               DAPO               |            failure            |
    | ---------------- | -------- | -------------------------------- | ----------------------------- |
    |       sync       |   NO-OP  |    Evict ``k``; refill ``2k``    |  NO-OP or opt-in refill ``k`` |
    |      async       |              All the same: Evict ``k``; refill ``k``.                       |

    In sync mode, DAPO is opt-in and trades generation time for training stability. Each ``k`` evictions add
    ``2k`` logical refill credits, but prompts are fetched only as bounded pending/running slots become available.
    Terminal groups are filtered while other requests remain in flight. Once enough groups are sampleable, inflight
    requests are drained and discarded. By default, failed groups remain sampleable and missing trajectories are
    padded downstream. Setting ``sync_refill_failed_groups=True`` allows refilling failed samples.
    In async mode, ``num_warmup_batches`` absorbs retry cost, so all three paths refill exactly ``k`` prompts.

    Args:
        trainer_mode (str): Trainer mode.
        trainer_config (DictConfig): Trainer configuration.
        max_off_policy_threshold (int): Maximum number of model versions that trajectory can span.
        max_off_policy_strategy (str): How to handle trajectory that exceeds the maximum number of model versions.
        sampler_kwargs (dict): Additional kwargs for the custom sampler.
        poll_interval (float, optional): Poll interval in seconds. Defaults to 2.0.
        refill_fn (callable, optional): Trainer-injected function that submits an exact number of fresh prompts.
        filter_groups_metric (str, optional): DAPO group-filtering metric read from each trajectory's
            ``extra_fields.reward_extra_info``. ``None`` disables DAPO filtering.
        train_batch_size (int, optional): Prompt count represented by one Sync DAPO in-flight batch.
        gen_batch_size (int, optional): Dataloader fetch granularity for refill dispatches.
        max_inflight_gen_batches (int): Maximum Sync DAPO prompt batches concurrently pending or running.
        max_num_gen_batches (int): Maximum complete candidate batches generated for one Sync DAPO
            optimizer batch. Non-positive values leave the replacement budget unbounded.
        sync_refill_failed_groups (bool): Whether sync sampling replaces failed groups with no trajectories.
        refill_all_failed_groups (bool): Whether any failed group is replaced, including groups with some
            materializable sibling trajectories.
    """

    def __init__(
        self,
        trainer_mode: str,
        trainer_config: DictConfig,
        max_off_policy_threshold: int,
        max_off_policy_strategy: str,
        sampler_kwargs: DictConfig,
        poll_interval: float = 2.0,
        refill_fn=None,
        filter_groups_metric: str | None = None,
        train_batch_size: int | None = None,
        gen_batch_size: int | None = None,
        max_inflight_gen_batches: int = 1,
        max_num_gen_batches: int = 0,
        sync_refill_failed_groups: bool = False,
        refill_all_failed_groups: bool = False,
    ):
        self.trainer_mode = trainer_mode
        self.trainer_config = trainer_config
        self.max_off_policy_threshold = max_off_policy_threshold
        self.max_off_policy_strategy = max_off_policy_strategy
        self.sampler_kwargs = sampler_kwargs
        self.poll_interval = poll_interval
        self.refill_fn = refill_fn
        self.filter_groups_metric = filter_groups_metric
        self.train_batch_size = train_batch_size
        self.gen_batch_size = gen_batch_size
        self.max_inflight_gen_batches = max_inflight_gen_batches
        self.max_num_gen_batches = max_num_gen_batches
        self.sync_refill_failed_groups = sync_refill_failed_groups
        self.refill_all_failed_groups = refill_all_failed_groups

        assert isinstance(self.max_off_policy_threshold, int) and self.max_off_policy_threshold > 0, (
            f"Invalid max off policy threshold: {self.max_off_policy_threshold}, must be an integer greater than 0"
        )
        assert self.max_off_policy_strategy in ["drop", "wait"], (
            f"Invalid max off policy strategy: {self.max_off_policy_strategy}, must be one of ['drop', 'wait']"
        )
        if self.filter_groups_metric is not None and self.refill_fn is None:
            raise ValueError("Group filtering (filter_groups_metric) requires refill_fn to replace evicted groups")
        if (self.sync_refill_failed_groups or self.refill_all_failed_groups) and self.refill_fn is None:
            raise ValueError("failed-group refill requires refill_fn to replace failed groups")
        self._validate_mode_config()
        # partition_id => {key: tag}
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)
        self.pending_keys: dict[str, set] = defaultdict(set)
        self.running_keys: dict[str, set] = defaultdict(set)
        self.finished_keys: dict[str, set] = defaultdict(set)
        self.failure_keys: dict[str, set] = defaultdict(set)
        # partition_id => {prompt_key: global_steps}, used to prioritize older samples.
        self.prompt_global_steps: dict[str, dict[str, int]] = defaultdict(dict)
        # Finished groups are immutable, so their DAPO classification can be reused across polling iterations.
        self._dapo_classification_cache: dict[str, dict[str, float | None]] = defaultdict(dict)

    def _validate_mode_config(self) -> None:
        if self.filter_groups_metric is not None:
            if not isinstance(self.max_inflight_gen_batches, int) or self.max_inflight_gen_batches <= 0:
                raise ValueError("max_inflight_gen_batches must be a positive integer")
            if not isinstance(self.max_num_gen_batches, int):
                raise ValueError("max_num_gen_batches must be an integer")
        if self.sync_refill_failed_groups and self.gen_batch_size != 1:
            raise ValueError("sync_refill_failed_groups requires gen_batch_size=1")

    def _sync_metadata_from_transfer_queue(self):
        """Sync the metadata from TransferQueue."""
        self.partitions.clear()
        self.pending_keys.clear()
        self.running_keys.clear()
        self.finished_keys.clear()
        self.failure_keys.clear()
        self.prompt_global_steps.clear()

        data = tq.kv_list()
        if data is None:
            return

        for partition_id, items in data.items():
            partition = self.partitions[partition_id]
            for key, tag in items.items():
                if tag.get("is_prompt", False):
                    # see: [GRPO group sampling control]
                    self.prompt_global_steps[partition_id][key] = tag["global_steps"]
                    match tag["status"]:
                        case "pending":
                            self.pending_keys[partition_id].add(key)
                        case "running":
                            self.running_keys[partition_id].add(key)
                        case "finished":
                            self.finished_keys[partition_id].add(key)
                        case "failure":
                            self.failure_keys[partition_id].add(key)
                        case _:
                            raise ValueError(f"Unknown status: {tag['status']}")
                else:
                    # see: [Trajectories storage format]
                    if key not in partition:
                        partition[key] = {}
                    partition[key].update(tag)

    @staticmethod
    def _metrics_prefix(partition_id: str) -> str:
        return "training" if partition_id == "train" else "validation"

    def _clear_groups(self, partition_id: str, uids: set[str]) -> None:
        """Remove prompt groups from TransferQueue and the active metadata snapshot."""
        if not uids:
            return

        trajectory_keys = {key for key in self.partitions[partition_id] if key.split("_")[0] in uids}
        tq.kv_clear(
            partition_id=partition_id,
            keys=[*uids, *trajectory_keys],
        )

        # Keep same-poll decisions consistent with tq.
        for key in trajectory_keys:
            del self.partitions[partition_id][key]
        for status_keys in (self.pending_keys, self.running_keys, self.finished_keys, self.failure_keys):
            status_keys[partition_id].difference_update(uids)
        for uid in uids:
            self.prompt_global_steps[partition_id].pop(uid, None)
            self._dapo_classification_cache[partition_id].pop(uid, None)

    def _dapo_filtered_keys(self, partition_id: str) -> tuple[set[str], Counter]:
        """Finished groups whose configured DAPO metric is identical across all trajectories.

        Returns the filtered uids and a ``{shared_metric_value: group_count}`` breakdown built in the
        same scope, so the diagnostic (which reward level the no-signal groups collapse to) travels
        with the uids through the return value instead of via hidden instance state.
        """
        if partition_id == "val" or self.filter_groups_metric is None:
            return set(), Counter()

        finished_uids = self.finished_keys[partition_id]
        classification_cache = self._dapo_classification_cache[partition_id]
        for uid in classification_cache.keys() - finished_uids:
            del classification_cache[uid]

        new_finished_uids = finished_uids - classification_cache.keys()
        trajectory_keys = [key for key in self.partitions[partition_id] if key.split("_")[0] in new_finished_uids]
        metrics_by_uid: dict[str, list[float]] = defaultdict(list)
        missing_metric_uids = new_finished_uids - {key.split("_")[0] for key in trajectory_keys}

        if trajectory_keys:
            data = tq.kv_batch_get(
                keys=trajectory_keys,
                partition_id=partition_id,
                select_fields=["extra_fields"],
            )
            extra_fields_list = list(data["extra_fields"])
        else:
            extra_fields_list = []

        for key, extra_fields in zip(trajectory_keys, extra_fields_list, strict=True):
            uid = key.split("_")[0]
            extra_fields = getattr(extra_fields, "data", extra_fields)
            reward_extra_info = extra_fields.get("reward_extra_info", {}) if isinstance(extra_fields, dict) else {}
            if self.filter_groups_metric not in reward_extra_info:
                missing_metric_uids.add(uid)
            else:
                metrics_by_uid[uid].append(float(reward_extra_info[self.filter_groups_metric]))

        if missing_metric_uids:
            raise RuntimeError(
                f"Finished groups are missing DAPO metric {self.filter_groups_metric!r}: "
                f"{sorted(missing_metric_uids)[:5]}"
            )

        for uid in new_finished_uids:
            values = metrics_by_uid[uid]
            classification_cache[uid] = float(values[0]) if len(values) > 1 and float(np.std(values)) == 0.0 else None

        filtered_rewards = {uid: reward for uid, reward in classification_cache.items() if reward is not None}
        return set(filtered_rewards), Counter(filtered_rewards.values())

    def _terminal_eviction_reasons(
        self, global_steps: int, partition_id: str
    ) -> tuple[set[str], set[str], set[str], Counter]:
        """Return stale, DAPO-filtered, and failed groups (plus the DAPO value->count breakdown).

        The three sets may overlap. Callers clear and refill their union, so one prompt is never handled
        twice. ``dapo_counts`` is the {shared_metric_value: group_count} diagnostic for ``dapo_uids``; it
        rides along in the return value so no hidden state is needed between production and consumption.
        """
        if partition_id == "val":
            return set(), set(), set(), Counter()

        dapo_uids, dapo_counts = self._dapo_filtered_keys(partition_id)
        failed_uids = set()
        if self.refill_all_failed_groups:
            failed_uids = set(self.failure_keys[partition_id])
        elif self.sync_refill_failed_groups:
            materializable_uids = {key.split("_")[0] for key in self.partitions[partition_id]}
            failed_uids = self.failure_keys[partition_id] - materializable_uids
        return set(), dapo_uids, failed_uids, dapo_counts

    def _sampleable_terminal_keys(
        self,
        partition_id: str,
        eviction_reasons: tuple[set[str], set[str], set[str], Counter],
    ) -> set[str]:
        terminal_uids = self.finished_keys[partition_id] | self.failure_keys[partition_id]
        stale_uids, dapo_uids, failed_uids, _dapo_counts = eviction_reasons
        return terminal_uids - (stale_uids | dapo_uids | failed_uids)

    def _evict_terminal_groups(
        self,
        global_steps: int,
        partition_id: str,
        eviction_reasons: tuple[set[str], set[str], set[str], Counter],
    ) -> tuple[set[str], int, int, dict]:
        """Evict terminal groups selected by any active policy exactly once."""
        stale_uids, dapo_uids, failed_uids, dapo_counts = eviction_reasons
        evicted_uids = stale_uids | dapo_uids | failed_uids
        if not evicted_uids:
            return set(), 0, 0, {}

        prefix = self._metrics_prefix(partition_id)
        metrics: dict = {}
        if stale_uids:
            prompt_global_steps = self.prompt_global_steps[partition_id]
            spans = np.array(
                [global_steps - prompt_global_steps.get(uid, global_steps) + 1 for uid in stale_uids],
                dtype=float,
            )
            metrics.update(
                {
                    f"{prefix}/off_policy/evicted_samples": len(stale_uids),
                    f"{prefix}/off_policy/evicted_samples_staleness/mean": spans.mean(),
                    f"{prefix}/off_policy/evicted_samples_staleness/max": spans.max(),
                    f"{prefix}/off_policy/evicted_samples_staleness/min": spans.min(),
                }
            )
        if dapo_uids:
            metrics[f"{prefix}/filter_groups/evicted_samples"] = len(dapo_uids)
            # Non-scalar diagnostic: how many filtered (no-signal) groups collapsed to each metric value.
            metrics[DAPO_FILTERED_REWARD_COUNTS_KEY] = dict(dapo_counts)
        if failed_uids:
            metrics[f"{prefix}/rollout_failure/evicted_samples"] = len(failed_uids)

        self._clear_groups(partition_id, evicted_uids)
        return evicted_uids, len(stale_uids), len(dapo_uids), metrics

    def _select_prompt_uids(
        self, partition_id: str, sampleable_keys: set[str], batch_size: int
    ) -> tuple[list[str], dict[str, dict], dict[str, int]]:
        prompt_global_steps_snapshot = dict(self.prompt_global_steps[partition_id])
        partition_snapshot = dict(self.partitions[partition_id])
        ordered_keys = sorted(
            sampleable_keys,
            key=lambda key: prompt_global_steps_snapshot.get(key, 0),
        )
        return ordered_keys[:batch_size], partition_snapshot, prompt_global_steps_snapshot

    def _materialize_batch(
        self, partition_id: str, selected_prompt_uids: list[str], partition_snapshot: dict[str, dict]
    ) -> KVBatchMeta:
        tq.kv_clear(partition_id=partition_id, keys=selected_prompt_uids)

        keys, tags = [], []
        selected = set(selected_prompt_uids)
        for key, tag in partition_snapshot.items():
            uid = key.split("_")[0]
            if uid in selected:
                keys.append(key)
                tags.append(tag)
        return KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)

    def _wait_for_next_poll(self, partition_id: str, last_debug_time: float) -> float:
        time.sleep(self.poll_interval)
        now = time.time()
        if now - last_debug_time > VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS:
            logger.info(
                f"pending: {len(self.pending_keys[partition_id])}, "
                f"running: {len(self.running_keys[partition_id])}, "
                f"finished: {len(self.finished_keys[partition_id])}, "
                f"failure: {len(self.failure_keys[partition_id])}"
            )
            return now
        return last_debug_time

    @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> tuple[KVBatchMeta, dict]:
        """Sample a batch using synchronous rollout semantics.

        NOTE: user can customize sampling strategy by setting:
        ```bash
        trainer.v1.sampler.custom_sampler.path = "path/to/your/sampler.py"
        trainer.v1.sampler.custom_sampler.name = "UserCustomReplayBuffer"
        ```

        Args:
            global_steps (int): Global steps of the current training.
            partition_id (str): Partition of TransferQueue, e.g. "train" or "val".
            batch_size (int, optional): Batch size.

        Returns:
            KVBatchMeta: A batch of data.
            dict: Auxiliary metrics.
        """
        last_debug_time = time.time()
        eviction_metrics: dict = {}
        dapo_enabled = partition_id != "val" and self.filter_groups_metric is not None
        refill_credit = 0
        draining = False
        max_inflight_prompts = 0
        candidate_prompts_generated = None
        max_candidate_prompts = 0
        if dapo_enabled:
            max_inflight_prompts = self.max_inflight_gen_batches * self.train_batch_size
            if self.max_num_gen_batches > 0:
                max_candidate_prompts = self.max_num_gen_batches * self.train_batch_size

        while True:
            # Eviction, gating, and selection below must all use this snapshot.
            self._sync_metadata_from_transfer_queue()
            if dapo_enabled and candidate_prompts_generated is None:
                candidate_prompts_generated = len(self.prompt_global_steps[partition_id])

            eviction_reasons = self._terminal_eviction_reasons(global_steps, partition_id)
            failed_count = len(eviction_reasons[2])
            evicted_uids, stale_count, dapo_count, metrics = self._evict_terminal_groups(
                global_steps, partition_id, eviction_reasons
            )
            if evicted_uids:
                _accumulate_eviction_metrics(eviction_metrics, metrics, stale_count)

            sampleable_keys = self._sampleable_terminal_keys(partition_id, eviction_reasons)
            has_enough_samples = len(sampleable_keys) >= batch_size
            inflight_count = len(self.pending_keys[partition_id]) + len(self.running_keys[partition_id])

            if not dapo_enabled and failed_count > 0 and not has_enough_samples:
                self.refill_fn(failed_count)
                continue

            if dapo_enabled:
                if has_enough_samples:
                    # Stop speculative dispatch, then drain requests already running under this policy version.
                    draining = True
                    refill_credit = 0
                elif not draining:
                    refill_credit += 2 * dapo_count + failed_count

                if not draining and refill_credit > 0:
                    available_slots = max(0, max_inflight_prompts - inflight_count)
                    dispatch_count = min(refill_credit, available_slots)
                    if max_candidate_prompts > 0:
                        assert candidate_prompts_generated is not None
                        dispatch_count = min(
                            dispatch_count,
                            max(0, max_candidate_prompts - candidate_prompts_generated),
                        )
                    assert self.gen_batch_size is not None
                    dispatch_count -= dispatch_count % self.gen_batch_size
                    if dispatch_count > 0:
                        assert self.refill_fn is not None
                        self.refill_fn(dispatch_count)
                        assert candidate_prompts_generated is not None
                        candidate_prompts_generated += dispatch_count
                        refill_credit -= dispatch_count
                        continue

                if (
                    not has_enough_samples
                    and inflight_count == 0
                    and max_candidate_prompts > 0
                    and candidate_prompts_generated >= max_candidate_prompts
                ):
                    raise ValueError(
                        f"algorithm.filter_groups exhausted max_num_gen_batches={self.max_num_gen_batches} "
                        f"after generating {candidate_prompts_generated} candidate prompts; "
                        f"only {len(sampleable_keys)} of {batch_size} required prompt groups had reward variance"
                    )

            can_select = has_enough_samples and (not dapo_enabled or inflight_count == 0)
            if can_select:
                selected_prompt_uids, partition_snapshot, _prompt_global_steps_snapshot = self._select_prompt_uids(
                    partition_id, sampleable_keys, batch_size
                )

                # Sync remains bufferless: all speculative requests are drained, then surplus is discarded.
                if dapo_enabled:
                    surplus_uids = sampleable_keys - set(selected_prompt_uids)
                    if surplus_uids:
                        self._clear_groups(partition_id, surplus_uids)
                        key = f"{self._metrics_prefix(partition_id)}/filter_groups/discarded_surplus_samples"
                        eviction_metrics[key] = eviction_metrics.get(key, 0) + len(surplus_uids)
                break

            last_debug_time = self._wait_for_next_poll(partition_id, last_debug_time)

        selected_uids = set(selected_prompt_uids)
        if partition_id != "val" and not any(key.split("_")[0] in selected_uids for key in partition_snapshot):
            message = "Sync replay buffer selected terminal groups with no materializable trajectories."
            if not self.sync_refill_failed_groups:
                message += " Enable trainer.v1.sampler.sync_refill_failed_groups to replace failed groups."
            raise RuntimeError(message)
        return self._materialize_batch(partition_id, selected_prompt_uids, partition_snapshot), eviction_metrics


class ReplayBufferAsync(ReplayBuffer):
    """Async sampling policy over the shared TransferQueue and dynamic-filter implementation."""

    def _validate_mode_config(self) -> None:
        pass

    def _stale_terminal_keys(self, global_steps: int, partition_id: str) -> set[str]:
        if partition_id == "val" or self.max_off_policy_strategy != "drop":
            return set()
        prompt_global_steps = self.prompt_global_steps[partition_id]
        terminal_keys = self.finished_keys[partition_id]
        return {
            uid
            for uid in terminal_keys
            if global_steps - prompt_global_steps.get(uid, global_steps) + 1 > self.max_off_policy_threshold
        }

    def _terminal_eviction_reasons(
        self, global_steps: int, partition_id: str
    ) -> tuple[set[str], set[str], set[str], Counter]:
        if partition_id == "val":
            return set(), set(), set(), Counter()

        stale_uids = self._stale_terminal_keys(global_steps, partition_id)
        dapo_uids, dapo_counts = self._dapo_filtered_keys(partition_id)
        return stale_uids, dapo_uids, set(self.failure_keys[partition_id]), dapo_counts

    def _has_enough_samples(
        self,
        global_steps: int,
        partition_id: str,
        batch_size: int,
        sampleable_keys: set[str],
    ) -> bool:
        # Dropless off-policy control: block sampling while any in-flight prompt has reached the staleness
        # threshold, so it can finish and be trained on instead of dropped.
        if self.max_off_policy_strategy == "wait":
            for key in self.pending_keys[partition_id] | self.running_keys[partition_id]:
                prompt_global_steps = self.prompt_global_steps[partition_id][key]
                if (global_steps - prompt_global_steps + 1) >= self.max_off_policy_threshold:
                    return False

        return len(sampleable_keys) >= batch_size

    @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> tuple[KVBatchMeta, dict]:
        """Sample a batch while evicting and replacing stale, DAPO-filtered, or failed groups."""
        last_debug_time = time.time()
        eviction_metrics: dict = {}

        while True:
            # Eviction and selection share one snapshot so newly terminal stale groups wait for the next eviction pass.
            self._sync_metadata_from_transfer_queue()

            eviction_reasons = self._terminal_eviction_reasons(global_steps, partition_id)
            evicted_uids, stale_count, _dapo_count, metrics = self._evict_terminal_groups(
                global_steps, partition_id, eviction_reasons
            )
            if evicted_uids:
                _accumulate_eviction_metrics(eviction_metrics, metrics, stale_count)
                if self.refill_fn is not None:
                    self.refill_fn(len(evicted_uids))
                continue

            sampleable_keys = self._sampleable_terminal_keys(partition_id, eviction_reasons)
            if self._has_enough_samples(global_steps, partition_id, batch_size, sampleable_keys):
                selected_prompt_uids, partition_snapshot, prompt_global_steps_snapshot = self._select_prompt_uids(
                    partition_id, sampleable_keys, batch_size
                )
                break

            last_debug_time = self._wait_for_next_poll(partition_id, last_debug_time)

        if partition_id != "val" and self.max_off_policy_strategy == "drop":
            selected_spans = [
                global_steps - prompt_global_steps_snapshot.get(uid, global_steps) + 1 for uid in selected_prompt_uids
            ]
            assert all(span <= self.max_off_policy_threshold for span in selected_spans), (
                f"drop strategy selected stale prompts: spans={selected_spans}, "
                f"threshold={self.max_off_policy_threshold}"
            )

        return self._materialize_batch(partition_id, selected_prompt_uids, partition_snapshot), eviction_metrics


ACTIVE_SAMPLING_METRIC_PREFIX = "active_sampling/"


class ActiveSamplingRounds:
    """Round arithmetic of TRL's GRPO active sampling (CarbonTeq TRL 1.12.0.post11).

    Counts are prompt groups. One optimizer batch needs ``target`` groups whose reward spread
    exceeds the epsilon; the candidate pool holds ``max_rounds * target`` groups. The first round
    dispatches ``target + oversample``; each later round dispatches the missing groups plus
    ``oversample_refill``, never more than the first round and never more than the pool left.
    A round is never cut below the groups it is missing: that exhausts the pool instead. The
    batch keeps the first ``target`` retained groups in dispatch order.
    """

    def __init__(self, *, target: int, max_rounds: int, oversample: int = 0, oversample_refill: int = 0):
        if target < 1 or max_rounds < 1:
            raise ValueError("active sampling requires a positive target and round limit")
        if oversample < 0 or oversample_refill < 0:
            raise ValueError("active sampling oversample and oversample_refill must be non-negative")
        self.target = target
        self.max_rounds = max_rounds
        self.oversample = oversample
        self.oversample_refill = oversample_refill
        self.pool = max_rounds * target
        self.cursor = 0
        self.rounds = 0
        self.retained = 0
        self.oversampled = 0
        self.round_log: list[tuple[int, int, int]] = []  # (requested, generated, retained) per round

    @property
    def oversampling(self) -> bool:
        return bool(self.oversample or self.oversample_refill)

    @property
    def missing(self) -> int:
        return max(self.target - self.retained, 0)

    def next_round(self) -> tuple[int, int] | None:
        """``(requested, round_size)`` of the next round, or ``None`` when the batch is full or rounds ran out."""
        if self.missing == 0 or self.rounds >= self.max_rounds:
            return None
        missing = self.missing
        requested = missing
        round_size = missing
        if self.oversampling:
            extra = self.oversample_refill if self.rounds else self.oversample
            requested = min(missing + extra, self.target + self.oversample)
            round_size = max(min(requested, self.pool - self.cursor), missing)
        if self.cursor + round_size > self.pool:
            raise RuntimeError(
                f"active sampling exhausted its bounded candidate pool: {missing} groups are missing but only "
                f"{self.pool - self.cursor} of {self.pool} candidates remain"
            )
        return requested, round_size

    def record(self, requested: int, round_size: int, retained: int) -> None:
        missing = self.missing
        self.cursor += round_size
        self.oversampled += round_size - missing
        self.rounds += 1
        self.retained += retained
        self.round_log.append((requested, round_size, retained))

    def require_full(self) -> None:
        if self.retained < self.target:
            raise RuntimeError(
                f"active sampling exhausted {self.max_rounds} generation rounds before filling its generation "
                f"batch; retained {self.retained} of {self.target} prompt groups"
            )

    def metrics(self, group_size: int) -> dict[str, float]:
        """TRL post11's metric names and values. Its ``candidate_groups_*`` counters count rows."""
        rows = group_size
        generated = self.cursor
        metrics: dict[str, float] = {
            "generation_rounds": self.rounds,
            "retained_fraction": self.retained / generated if generated else 0.0,
            "generated_rows": generated * rows,
            "candidate_groups_reserved": self.pool * rows,
            "candidate_groups_generated": generated * rows,
            "candidate_groups_retained": self.retained * rows,
            "candidate_groups_unused": (self.pool - self.cursor) * rows,
        }
        if self.oversampling:
            for index, (requested, generated_groups, retained) in enumerate(self.round_log, start=1):
                metrics[f"round_{index}_requested_groups"] = requested
                metrics[f"round_{index}_generated_groups"] = generated_groups
                metrics[f"round_{index}_retained_groups"] = retained
            metrics["oversampled_groups"] = self.oversampled
            metrics["discarded_groups"] = max(self.retained - self.target, 0)
        return {f"{ACTIVE_SAMPLING_METRIC_PREFIX}{name}": value for name, value in metrics.items()}


def active_sampling_round_capacity_error(
    *,
    train_batch_size: int,
    rollout_n: int,
    oversample: int,
    engine_max_num_seqs: int | None,
    max_concurrent_episodes: int | None,
    worker_slots: int | None,
) -> str | None:
    """Explain why the oversampled first round would not run as one concurrent wave.

    The first round, ``(train_batch_size + oversample) * rollout_n`` episodes, is the largest round
    (refill rounds are capped at it). Without oversampling nothing is checked.
    """
    if oversample <= 0:
        return None
    episodes = (train_batch_size + oversample) * rollout_n
    limits = []
    if engine_max_num_seqs is not None and engine_max_num_seqs < episodes:
        limits.append(f"rollout engines serve at most {engine_max_num_seqs} sequences")
    if max_concurrent_episodes is not None and max_concurrent_episodes < episodes:
        limits.append(f"agent.max_concurrent_episodes is {max_concurrent_episodes}")
    if worker_slots is not None and worker_slots < episodes:
        limits.append(f"agent workers x max_concurrent_episodes_per_worker is {worker_slots}")
    if not limits:
        return None
    return (
        f"algorithm.active_sampling.oversample={oversample} needs {episodes} concurrent episodes for the first "
        f"round (({train_batch_size} prompts + {oversample}) x {rollout_n}), but " + "; ".join(limits)
    )


class ActiveSamplingReplayBuffer(ReplayBuffer):
    """Synchronous replay buffer that fills each optimizer batch with TRL's round-based active sampling.

    Every round dispatches an exact number of fresh prompts, waits until all of them are terminal,
    then keeps the groups whose ``metric`` spread exceeds ``reward_std_epsilon`` (sample standard
    deviation) and evicts the rest, including failed groups. Rounds follow
    :class:`ActiveSamplingRounds`. Validation sampling is unchanged.
    """

    def __init__(
        self,
        *args,
        dispatch_fn=None,
        active_max_rounds: int = 10,
        active_oversample: int = 0,
        active_oversample_refill: int = 0,
        active_reward_std_epsilon: float = 0.0,
        active_metric: str = "seq_reward",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.trainer_mode != "sync":
            raise ValueError("algorithm.active_sampling requires trainer.v1.trainer_mode=sync")
        if dispatch_fn is None:
            raise ValueError("active sampling requires a dispatch_fn that returns the dispatched prompt uids")
        if self.filter_groups_metric is not None:
            raise ValueError("algorithm.active_sampling and algorithm.filter_groups are separate refill strategies")
        if not np.isfinite(active_reward_std_epsilon) or active_reward_std_epsilon < 0:
            raise ValueError("algorithm.active_sampling.reward_std_epsilon must be finite and non-negative")
        ActiveSamplingRounds(
            target=1,
            max_rounds=active_max_rounds,
            oversample=active_oversample,
            oversample_refill=active_oversample_refill,
        )
        self.dispatch_fn = dispatch_fn
        self.active_max_rounds = active_max_rounds
        self.active_oversample = active_oversample
        self.active_oversample_refill = active_oversample_refill
        self.active_reward_std_epsilon = active_reward_std_epsilon
        self.active_metric = active_metric
        self.group_size: int | None = None

    def _keeps_group(self, values: list[float]) -> bool:
        if len(values) < 2:
            return False
        return float(np.std(values, ddof=1)) > self.active_reward_std_epsilon

    def _classify_round(self, partition_id: str, round_uids: list[str]) -> list[str]:
        """Return the kept uids of a finished round in dispatch order and evict the others."""
        finished = [uid for uid in round_uids if uid in self.finished_keys[partition_id]]
        trajectory_keys = [key for key in self.partitions[partition_id] if key.split("_")[0] in set(finished)]
        values: dict[str, list[float]] = defaultdict(list)
        if trajectory_keys:
            data = tq.kv_batch_get(keys=trajectory_keys, partition_id=partition_id, select_fields=["extra_fields"])
            for key, extra_fields in zip(trajectory_keys, list(data["extra_fields"]), strict=True):
                extra_fields = getattr(extra_fields, "data", extra_fields)
                info = extra_fields.get("reward_extra_info", {}) if isinstance(extra_fields, dict) else {}
                if self.active_metric not in info:
                    raise RuntimeError(
                        f"Finished group {key.split('_')[0]} is missing active-sampling metric {self.active_metric!r}"
                    )
                values[key.split("_")[0]].append(float(info[self.active_metric]))
        kept = [uid for uid in finished if self._keeps_group(values[uid])]
        sizes = {len(values[uid]) for uid in kept}
        if len(sizes) > 1 or (self.group_size is not None and sizes and sizes != {self.group_size}):
            raise RuntimeError(f"active sampling received incomplete prompt groups: sizes {sorted(sizes)}")
        if sizes:
            self.group_size = sizes.pop()
        self._clear_groups(partition_id, set(round_uids) - set(kept))
        return kept

    @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> tuple[KVBatchMeta, dict]:
        if partition_id == "val":
            return super().sample(global_steps, partition_id, batch_size)
        rounds = ActiveSamplingRounds(
            target=batch_size,
            max_rounds=self.active_max_rounds,
            oversample=self.active_oversample,
            oversample_refill=self.active_oversample_refill,
        )
        retained: list[str] = []
        last_debug_time = time.time()
        while (plan := rounds.next_round()) is not None:
            requested, round_size = plan
            round_uids = list(self.dispatch_fn(round_size))
            if len(round_uids) != round_size:
                raise RuntimeError(f"active sampling dispatched {len(round_uids)} prompts, expected {round_size}")
            while True:
                self._sync_metadata_from_transfer_queue()
                terminal = self.finished_keys[partition_id] | self.failure_keys[partition_id]
                if all(uid in terminal for uid in round_uids):
                    break
                last_debug_time = self._wait_for_next_poll(partition_id, last_debug_time)
            kept = self._classify_round(partition_id, round_uids)
            retained.extend(kept)
            rounds.record(requested, round_size, len(kept))
        rounds.require_full()
        self._sync_metadata_from_transfer_queue()
        selected = retained[:batch_size]
        surplus = set(retained[batch_size:])
        self._clear_groups(partition_id, surplus)
        metrics = rounds.metrics(self.group_size or 1)
        partition_snapshot = dict(self.partitions[partition_id])
        if not any(key.split("_")[0] in set(selected) for key in partition_snapshot):
            raise RuntimeError("active sampling selected groups with no materializable trajectories")
        return self._materialize_batch(partition_id, selected, partition_snapshot), metrics
