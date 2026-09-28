# CarbonTeq verl fork ledger

## Status

**Unpublished candidate: round-based active sampling (branch
`codex/vortex-active-sampling`, from the post5 receipt `9c10bd1a`).** Adds the
`algorithm.active_sampling` block and `ActiveSamplingReplayBuffer`, and the
`data.prompt_selector` extension point, described under "Maintained delta".
Phases 2, 3 and 5 of Posttrain's `docs/plan/verl-vortex-port.md` (plus the
`sequence_clip` loss and SAMPO hierarchy metrics for SAMPO parity with TRL); no
release or tag yet (the next release candidate, post6, collects phases 2-5).

**Release candidate `0.9.0.post5`.** Post5 is the post4 release commit
(`54124edf`) plus the token-clip policy loss and the unclipped k3 KL estimator
described under "Maintained delta" (commit
`a4d84ad30b94c11c4de41b3d915eca6399ad2b6a`). It changes no dependency and no
existing loss or KL mode. It fixes a live consumer bug: Posttrain already
selects `policy_loss.loss_mode=token_clip` and `kl_loss_type=k3_unclipped` for
GDPO and CAPO, and post4 registers neither, so those runs failed at the first
actor update. It also provides the OLMo 3 objective for Posttrain's VORTEX
port (`docs/plan/verl-vortex-port.md`). Branch: `codex/vortex`, tagged
`carbonteq-v0.9.0.post5` at this release commit.

Its immutable release commit is
`9fd6e7a31396ba33a29233cc869ab05b0a9e5a80`, tagged `carbonteq-v0.9.0.post5`
(annotated tag object `3945e01a4e3fabad02b123713201e757109bb18c`). Built from
two clean clones with `SOURCE_DATE_EPOCH` set to the commit time: the wheel is
byte-identical across both builds and the two sdists have identical members.
Retained wheel `verl-0.9.0.post5-py3-none-any.whl` SHA-256:
`c16a2ad14d1bd60947229bde41fae99955a4fcb7ab3f892020ef0c24e35dd158`;
retained sdist `verl-0.9.0.post5.tar.gz` SHA-256:
`3c9e17c2d04a798eae1836e8dd82f64480bcbe4d8ee94441028625b2da71df57`.
`twine check` passes for both. The GitHub release and the Posttrain
retained-asset publisher (`publish-verl-internal.yml`) are pending.

**Release candidate `0.9.0.post4`.** Its immutable release commit is
`54124edfb8d0b73694696400cf07a76a14d9be65`, tagged `carbonteq-v0.9.0.post4`;
retained wheel SHA-256
`1e5e5a50c14ec486019421ca06de03fbbc24010f850f5e66f2d731469a8f8eb0`, retained
sdist SHA-256
`8620646c250e85a0dee984d360a4c102e97a4776d5711accea8eb66a743445e5` (recorded
by receipt commit `39622a5e` on `codex/precision-fp16`). Post4 is post3 (`18338a0e`, through its
publication receipt `78266f97`) plus fp16 training evidence from the FSDP
engine's existing loss scaling (`actor/loss_scale` and
`actor/optimizer_step_skipped` per optimizer step) and the per-token
rollout-versus-actor log-probability gap
(`training/rollout_logp_diff_{mean,p99,max}`,
`training/rollout_seq_logp_diff_abs_mean`). It changes no training behavior and
no dependency, including the `vllm` extra. Branch: `codex/precision-fp16`.

**Release candidate `0.9.0.post3`.** Post3 retains the post2 runtime and updates
the `vllm` extra to the exact CarbonTeq Uno source-overlay commit
`37706e7d920abc97c705ffecee0919d64ef31485`, released as
`carbonteq-v0.26.1.dev1`. This dependency update does not claim native K2
training support in veRL. Branch: `codex/verl-rollout-execution`.

Its immutable release commit is
`18338a0efbd6f103378d2861f4a078ad243db455`, tagged
`carbonteq-v0.9.0.post3`. Wheel SHA-256:
`55db59d956084c27941a787952c1d66fcdbd4f7c1d71c9ddd1f20c283af7186f`;
sdist SHA-256:
`75caad0671f3f3f70241b049ff6b3fb937cee34a4bdcadd74230b1c46bc4bb31`.
Posttrain retained-asset publisher `35292934219` passed exact-byte development
readback and clean installation.

Post2 extends the published post1 runtime with bounded concurrent agent-loop
episodes, explicit Ray CPU-resource reservations, complete-group preservation
in the V1 TransferQueue path, and partial-rollout policy-version provenance.
Its immutable release commit is `98742d3e9507318ba0b5d4944034deb7db1ec84b`,
tagged `carbonteq-v0.9.0.post2`. Wheel SHA-256:
`adeef5700a7f56a10beaef5304f6a5626b015c45334d516d2330243bcad174f5`;
sdist SHA-256:
`0c1c1ac543a93d2ff5ba50c4394c4b9838ad90615c981422ce22df4693224a0a`.
Posttrain retained-asset publisher `34335257738` passed exact-byte
development-channel readback and clean installation.

Post1 integrated the maintained consumer pin
`808923d487aa2c524fda02cf5289110541b4221f` onto stable upstream v0.9.0 at
`cec7e74c361bb973b641db8dfbb75a5544c33139`. It was released as
`carbonteq-v0.9.0.post1` and published to `carbonteq/dev` by Posttrain Actions
run `34006221953`. Wheel SHA-256:
`3d66ac6b78848ef591dd6e4be4d324fcac12c27247a05ad1c14dcf9fb370256e`;
sdist SHA-256:
`6da77c10e5d37399c655b15e3ed78d520be2ca648ed763d43ccb7404dfe53d72`.
Historical `0.9.0.dev2` candidate advanced the
published runtime, SAMPO, and dense-distillation lineage at
`c3f49b9117b882fa888e25e4a771461e13167848` without changing its supported
algorithm surface. It exists so consumers can select one immutable fork commit
and a uniquely versioned private distribution rather than an ambiguous
`0.9.0.dev` build.

The candidate must pass the focused CPU suite, build/install verification, a
Posttrain runtime-image rebuild, and the bounded GPU gates before it is
promoted on `origin/main` and published to the internal package index. The
existing SAMPO and distillation evidence remains historical qualification; it
does not make a newly built distribution release-qualified.

Upstream repository: `https://github.com/verl-project/verl.git`

Upstream base for the maintained line:
`483b8a009ba3a97563edee3a19887e4862b8094a` (`v0.9.0`)

Reconstruction base for the published runtime delta:
`553280b88afe4e7fbc4aefeff27bbf0a22e7c048`

Expected remotes:

- `origin`: `git@github.com:carbonteq-ai/verl.git`
- `upstream`: `https://github.com/verl-project/verl.git`

Published SAMPO implementation commit:
`8a718e5be7a107587f63967336ece333a5c160e1`.

Published SAMPO evidence, Python 3.13, and vLLM-selection commit:
`b42495dfe138dcc114c39b486a2c58c0e1ff6f29`.

Published dense distillation teacher-logprob alignment implementation:
`83a0fa8edb5c65014604d32546f2362c1151677a`, with nested micro-batch
handling in `8c22e71ef9eafeecafa3942a014b7f9ea343bee0` and fully masked padding
handling in `8cb6d338dbefb5b387647ab7dadbe5b23218c51b`.

Release-candidate parent: `c3f49b9117b882fa888e25e4a771461e13167848`.
The published release commit and index artifact hashes are recorded here only
after the candidate is committed, pushed, built once, and read back.

## Maintained delta

### TRL-equivalent settings: sampler correction bounds, GRPO scaling, row exclusion, admission, linear LR (candidate)

Posttrain selects TRL's GRPO semantics for every online-RL setting; these deltas
let veRL reproduce the ones upstream cannot express.

- Sampler correction (`verl/trainer/ppo/rollout_corr_helper.py`,
  `RolloutCorrectionConfig`): `rollout_is_clip_min` clamps truncated weights from
  below (TRL `vllm_importance_sampling_clip_min`), and
  `rollout_is_log_ratio_bound` (default 20, `null` disables) removes the
  log-ratio safety clamp so weights equal TRL's `exp(log ratio)`. Mask modes use
  the existing "lower_upper" IcePop threshold.
- GRPO scaling (`compute_grpo_outcome_advantage`, `AlgoConfig`):
  `grpo_std_epsilon` (TRL adds 1e-4), `grpo_std_scope` `group` or `batch`
  (TRL `scale_rewards="batch"`), and rows listed in
  `non_tensor_batch["exclude_from_group_stats"]` leave group statistics exactly
  as TRL's NaN rewards do (nan-mean, unbiased nan-std, undefined advantages 0).
- Row exclusion (`algorithm.exclude_flagged_rows`, V1 `_compute_advantage`):
  rows whose agent-loop `extra_fields` carry `exclude_from_loss=True` (TRL's
  masked truncated completions) are dropped from GRPO statistics, and after
  advantages their response mask, loss mask, advantages and correction weights
  are zeroed. SAMPO keeps them in its centring, as TRL's SAMPO path does. The
  DAPO filter and active sampling ignore non-finite metric values (NaN marks an
  excluded trajectory) and need two finite values with spread.
- Group admission (`trainer.v1.sampler.failed_group_attempts`, sync
  `ReplayBuffer`): a failed group is re-dispatched with its prompt until it has
  been attempted that many times, then dropped; plain GRPO trains the remaining
  groups with the seq-mean normalized over real rows (TRL's
  `admission_loss_scale`), DAPO refills a dropped group like a filtered one.
- Linear LR (`get_linear_schedule_with_warmup`, FSDP optimizer): Hugging Face's
  `lr_scheduler_type="linear"`.

CPU regression coverage: `tests/trainer/ppo/v1/test_trainer_base_on_cpu.py`,
`tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py`,
`tests/trainer/ppo/v1/test_active_sampling_on_cpu.py`,
`tests/utils/test_linear_lr_schedule_on_cpu.py`.

### Sequence-ratio clip loss and SAMPO hierarchy evidence (candidate)

TRL's GRPO trainer with `importance_sampling_level="sequence"` (the objective
Posttrain's TRL SAMPO path runs) forms one ratio per row,
`s_i = exp(mean_t(log_prob - old_log_prob))`, and takes the gradient through
that mean. veRL's `gspo` uses GSPO's stop-gradient token form, which gives each
token its own advantage; with SAMPO's per-turn advantages the two gradients
differ. `verl/trainer/ppo/core_algos.py` registers `sequence_clip` with TRL's
form (per-token clip of the shared ratio, rollout-correction weights, normal
aggregation). With `seq-mean-token-mean` it matches TRL to 1e-8 relative
(veRL divides by tokens + 1e-8 where TRL clamps at 1).

The SAMPO estimator also reports `sampo/episode_advantage_abs_mean`,
`sampo/turn_advantage_abs_mean`, `sampo/turn_advantage_informative_fraction`,
`sampo/singleton_anchor_fraction` and `sampo/turn_credit_share`, the hierarchy
evidence the consumer records for its TRL path.

CPU regression coverage: `tests/trainer/ppo/test_token_clip_policy_loss_on_cpu.py`
and `tests/trainer/ppo/test_sampo_advantage_on_cpu.py`.

### Prompt selector extension point (candidate)

A caller-owned curriculum must choose which tasks every dispatch uses, from
evidence of earlier rounds, and checkpoint its state with the model. Upstream
veRL draws prompts only from the dataloader's sampler and has no feedback path.

- `verl/trainer/ppo/v1/prompt_selector.py`: the `PromptSelector` protocol
  (`select(num_prompts, global_steps, stage, round_index)` returning dataset
  indices; `observe([(dataset_index, metric values)], global_steps)`;
  `save_checkpoint(dir)`; `load_checkpoint(dir)`) and `load_prompt_selector`.
- `data.prompt_selector` (`class_path`, `kwargs`, `metric`) in
  `legacy_data.yaml` and the regenerated reference configs.
- `verl/trainer/ppo/v1/trainer_base.py`: dispatches build their batch from the
  selected dataset rows (stage `initial_batch`, or `active_sampling_refill`
  with rounds numbered from 1); finished groups are observed after each
  active-sampling round (kept and rejected, failed groups excluded) or after
  the step's batch is sampled; the selector saves into and loads from each
  `global_step_*` folder. Sync mode only; not with DAPO `filter_groups`.
- `ActiveSamplingReplayBuffer` passes `round_index` to the dispatcher and calls
  `observe_fn` with each round's finished groups in dispatch order.

CPU regression coverage: `tests/trainer/ppo/v1/test_prompt_selector_on_cpu.py`
and `tests/trainer/ppo/v1/test_active_sampling_on_cpu.py`.

### Round-based active sampling in the synchronous V1 trainer (candidate)

veRL's DAPO `filter_groups` path streams: every evicted group adds two refill
credits and replacement prompts start while earlier ones still run. TRL's GRPO
active sampling (CarbonTeq TRL 1.12.0.post11, used by the OLMo 3 / VORTEX and
SAMPO recipes) works in rounds, and the consumer's curriculum contract needs a
decision boundary per round, so the port needs TRL's round semantics.

- `verl/trainer/config/algorithm.py` `ActiveSamplingConfig` and the
  `algorithm.active_sampling` block in `ppo_trainer.yaml` (regenerated
  `_generated_ppo_*trainer.yaml`): `enable`, `max_candidate_batches`,
  `oversample`, `oversample_refill`, `reward_std_epsilon`, `metric`.
- `verl/trainer/ppo/v1/replay_buffer.py`: `ActiveSamplingRounds` (TRL's round
  arithmetic in prompt groups: round one `target + oversample`, later rounds
  `missing + oversample_refill` capped at round one, extra groups cut to the
  remaining pool of `max_candidate_batches * target`, pool exhaustion and
  round exhaustion errors, TRL's `active_sampling/...` metric names and values,
  including TRL's row-valued `candidate_groups_*` counters);
  `ActiveSamplingReplayBuffer` dispatches each round through the trainer,
  waits until every group of the round is terminal, keeps groups whose metric
  sample standard deviation exceeds the epsilon, evicts the rest (failed groups
  included), and keeps the first `train_batch_size` retained groups in dispatch
  order; `active_sampling_round_capacity_error` checks that an oversampled
  first round fits the rollout engines' `max_num_seqs`, the agent
  `max_concurrent_episodes`, and workers x `max_concurrent_episodes_per_worker`.
- `verl/trainer/ppo/v1/trainer_base.py`: selects the buffer when enabled (sync
  mode, `parameter_sync_step=1`, not with a custom sampler or `filter_groups`),
  runs the capacity check at startup, forces `data.gen_batch_size=1`, adds
  `_dispatch_prompts` (returns uids in dispatch order), and lets the buffer own
  the first dispatch of each step.

CPU regression coverage: `tests/trainer/ppo/v1/test_active_sampling_on_cpu.py`.
The side-by-side check against TRL's real `_prepare_active_sampling_inputs`
lives in the consumer (Posttrain `packages/train/tests/test_verl_active_sampling_parity.py`).

### Token-clip policy loss and unclipped k3 KL (post5)

Upstream's `vanilla` PPO loss is not the loss TRL's GRPO trainer computes for
the OLMo 3 and DAPO recipes: it applies dual clipping (a negative-advantage
token's loss is capped at `-A * clip_ratio_c`) and clamps the log ratio to
[-20, 20], and upstream's `low_var_kl`/`k3` KL clamps the estimate to
[-10, 10]. A consumer that selects the OLMo 3 recipe must get the same
objective on both backends.

- `verl/trainer/ppo/core_algos.py` registers policy loss `token_clip`:
  `max(-A * r, -A * clip(r, 1 - clip_ratio_low, 1 + clip_ratio_high))` per
  sampled token with `r = exp(log_prob - old_log_prob)`, multiplied by the
  rollout-correction weights when present and aggregated with `agg_loss` and
  the actor's global batch information. It raises on a non-finite ratio.
- `kl_penalty_forward(..., "k3_unclipped")` returns
  `expm1(ref - logp) - (ref - logp)` without clamping.

Both names match the unpublished GDPO/CAPO candidate in the
`codex/gdpo-capo-support` worktree so the two deltas converge to one
implementation when merged. Posttrain selects them for OLMo 3 (and already
selects them for GDPO/CAPO). Keep them until upstream offers a clip mode
without dual clipping and an unclamped k3 estimator.

CPU regression coverage:
`tests/trainer/ppo/test_token_clip_policy_loss_on_cpu.py` (TRL formula and
gradient, no dual clip, rollout-correction weights, global token
normalization, non-finite rejection, unclipped k3 value and gradient). The
cross-backend equivalence test lives in the consumer:
Posttrain `packages/train/tests/test_verl_olmo3_parity.py`.

Build post5 from the tagged commit exactly as post4 was built (the wheel is
byte-reproducible; the sdist archive carries build times, so the retained
release asset is the authority for its hash):

```bash
git clone --branch carbonteq-v0.9.0.post5 https://github.com/carbonteq-ai/verl.git verl-post5
cd verl-post5
SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)" uv build --out-dir dist
sha256sum dist/verl-0.9.0.post5-py3-none-any.whl dist/verl-0.9.0.post5.tar.gz
uvx twine check dist/*
```

Stable-base candidate evidence (2026-09-06): 151 focused CPU tests passed.
Coverage includes core algorithms, REINFORCE++ credit, SAMPO routing and V1
metadata materialization, model/QLoRA settings, replay/refill, 3-D position
identity, dense teacher alignment, teacher pools, Qwen decoder compatibility,
and speculative counters. The temporary test harness uses Ray 2.49.2; the
production runtime selects Ray 2.56.1 and must be tested as a built image.
No GPU/runtime-image or stable-index promotion is claimed by these CPU tests.

### Unpublished REINFORCE++ observation-credit correction (2026-09-06)

Superseded during stable-base integration: use upstream's correction once,
retaining our independent regression tests. PRIME's Python 3.13 module-runtime
repair is also upstream now. SAMPO estimator/V1 identity and sparse-mask
transport, bounded failed-group refill, dense teacher microbatch alignment,
repaired 3-D position IDs, QLoRA, and speculative metrics remain maintained.
The upstream cache release before rollout wake is retained alongside adapter
staging before wake. Upstream's injectable load-balancer class coexists with
the maintained speculative-counter snapshots. Transformers retains the fork's
<5.15 ceiling and upstream's >=5.5.3 floor; 5.6.0 remains excluded.
The estimator-registry test now restores global state so running it before
SAMPO tests cannot erase registered algorithms.

The isolated `codex/gdpo-capo-support` worktree starts at consumer pin
`808923d487aa2c524fda02cf5289110541b4221f`, preserving the unrelated dirty
`verl-upstream` checkout. It carries running returns through masked observations,
following upstream fix `8a1bf6d5b080173e29834aca55ee8d47549b2ea6`. Discounting
advances only on sampled actions; observation/padding returns are zero. This
is REINFORCE++ only: CAPO must not acquire reward-to-go semantics.

`tests/trainer/ppo/test_reinforce_observation_credit_on_cpu.py` has five cases
covering observation-insertion invariance, non-unit gamma, float32/float64,
padding, and masked reward exclusion. Together with `test_core_algos_on_cpu.py`,
28 CPU tests pass. No model training, runtime rebuild, release publication, or
consumer-pin change is implied. On a veRL v0.9.0 rebase, use its upstream fix
and retain these regressions instead of duplicating the maintained delta.
Consumer state: Posttrain `docs/tooling/verl/README.md` and
`docs/plan/gdpo-capo-dual-backend-support.md`.

### Qwen 3.5 QLoRA and runtime counters

The published branch includes the earlier candidate commits `05f83242` and
`5da56132`. They add FSDP QLoRA support for Qwen 3.5 and retain
speculative-decoding runtime totals. Their consumer-facing qualification state
is recorded in `docs/tooling/verl/README.md` in the post-training framework
repository.

### Preserve repaired 3-D position IDs through mini-batch selection

The upstream workaround for 3-D VLM position IDs marks the sequence dimension
as ragged after a TensorDict/Ray round trip. Its underlying PyTorch nested
storage can still describe the fixed position-ID channel dimension as jagged.
Calling `unbind()` through the repaired target axis then attempts to split a
sequence using channel lengths, which fails for Qwen 3.5's four-channel
position IDs during a real actor update.

The fork now detects that storage/target-axis mismatch only while selecting
rows, temporarily uses the storage axis to read complete samples, and rebuilds
the selected tensor on the intended sequence axis. Correctly constructed
non-last-ragged tensors retain their existing fast path.

CPU regression coverage:
`tests/test_protocol_v2_on_cpu.py::test_index_select_tensor_dict_handles_repaired_3d_position_ids`.

### SAMPO hierarchical multi-turn advantages

Upstream verl provides the GSPO policy-loss kernel but not SAMPO's GiGPO-style
episode and anchor-state-relative advantage estimator.

The published candidate delta:

- registers `AdvantageEstimator.SAMPO` in
  `verl/trainer/ppo/core_algos.py`;
- computes token-aligned joint advantages from trajectory reward, ordered
  assistant-turn spans, anchor-state keys, and optional step rewards;
- routes agent-loop metadata through `verl/trainer/ppo/ray_trainer.py`;
- defines typed algorithm configuration in
  `verl/trainer/config/algorithm.py` and `ppo_trainer.yaml`;
- regenerates every `_generated_ppo_*trainer.yaml` reference config;
- covers the equations, sparse/explicit rewards, masks, invalid metadata, and
  trainer routing in `tests/trainer/ppo/test_sampo_advantage_on_cpu.py`.

The semantic reference is the official ARL-Arena SAMPO implementation at
`a25a2a229c85431b421ac785fa5f375a99b2072a`. That implementation maps SAMPO
to GiGPO hierarchical advantages plus the GSPO loss. This fork keeps one
complete multi-turn trajectory in each batch row and applies the same
hierarchical equations to token-aligned turn spans.

### Bound dynamic group replacement in the core V1 trainer

Upstream veRL's V1 replay buffer already filters reward-constant prompt groups
and refills them with fresh rollouts, but it ignores
`algorithm.filter_groups.max_num_gen_batches`. The CarbonTeq delta forwards
that field into the synchronous replay buffer and treats it as a total
candidate budget for one optimizer batch. Once the configured number of full
candidate batches has been generated, the trainer fails with the observed
sampleable-group count instead of retrying forever.

This behavior adapts the bounded replacement semantics from the Apache-2.0
`verl-project/verl-recipe` DAPO implementation at
`230ee612279d552a4f34ecbfab931c213abd514d`. It is implemented directly in
`verl.trainer.ppo.v1` rather than copying the recipe entrypoint or maintaining a
second runtime checkout.

CPU regression coverage:
`tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py::test_sync_dapo_stops_at_bounded_candidate_batch_limit`
and
`tests/trainer/ppo/v1/test_trainer_base_on_cpu.py::test_builtin_filter_groups_forwards_total_generation_limit`.

### Preserve SAMPO rollout metadata in the V1 advantage path

The V1 trainer previously fetched only reward, mask, and log-probability fields
from TransferQueue before advantage computation. Agent-loop rollouts therefore
stored the three SAMPO metadata arrays successfully, but the driver omitted
them from the reconstructed `DataProto` and rejected every SAMPO optimizer
batch as incomplete.

The CarbonTeq delta requests the agent loop's `extra_fields` object for SAMPO
batches and extracts `sampo_turn_spans`, `sampo_anchor_state_keys`, and
`sampo_step_rewards` into `DataProto.non_tensor_batch` before calling the
shared advantage router. Other advantage estimators retain their existing
TransferQueue field selection.

CPU regression coverage:
`tests/trainer/ppo/v1/test_trainer_base_on_cpu.py::test_sampo_advantage_fetches_and_forwards_rollout_metadata`.

### Preserve SAMPO prompt-group identity

SAMPO requires every optimizer batch to contain exactly `rollout.n`
trajectories for each prompt. TransferQueue's replay-buffer key is the
authoritative `{prompt_uid}_{session_id}_{output_index}` identity, while the
materialized `uid` field may be populated by an adapter with a trajectory-local
identity.

For SAMPO only, the V1 advantage path now derives the prompt identity from the
authoritative batch key before computing group-relative advantages. Other
advantage estimators retain their existing materialized `uid` behavior.

CPU regression coverage is included in
`tests/trainer/ppo/v1/test_trainer_base_on_cpu.py::test_sampo_advantage_fetches_and_forwards_rollout_metadata`,
which supplies distinct materialized trajectory identifiers for two members of
one replay-buffer prompt group.

### Runtime dependency compatibility

The runtime delta adds `orjson`, permits supported Transformers releases below
5.15 while excluding known-broken 5.6.0, and selects CarbonTeq vLLM commit
`37706e7d920abc97c705ffecee0919d64ef31485` in the `vllm` extra. That vLLM
commit is based on upstream `75c71390d5b399f5397a9166920fc45902f99f14`
(`v0.26.1rc0` development line) and carries both the earlier bounded
TurboQuant cache correction and native Uno proposer documented in the vLLM
fork's `CARBONTEQ_FORK.md`.

The framework's veRL runtime remains a separately locked Python environment.
Its release lock independently selects the same immutable vLLM commit; the
fork extra prevents ad hoc veRL installations from silently resolving an
unqualified upstream wheel.

### Keep local Python 3.13 development reproducible

`setup.py` declares the runtime dependencies imported by the core trainer,
keeps the historical `prime` extra as a compatibility no-op, and replaces
PRIME's removed `pyext` runtime with standard-library module compilation in
`verl/utils/reward_score/prime_code/testing_util.py`. `pyproject.toml` records
the supported Python range and mutually exclusive vLLM/SGLang resolver
environments.

Regression coverage:
`tests/utils/reward_score/test_sandbox_on_cpu.py` and the focused SAMPO/V1
trainer suites.

### Report SAMPO hierarchical evidence

### Bounded agent-loop episode admission

The agent-loop manager already partitions a collection across Ray workers, but
each worker previously started every locally assigned trajectory immediately.
This fork adds opt-in `agent.max_concurrent_episodes_per_worker` and
`agent.max_concurrent_episodes` configuration. The worker uses one
`asyncio.Semaphore` shared by all rows in its collection. The manager validates
that a collection-wide ceiling is paired with a local gate and that
`num_workers * per_worker <= global`; it rejects advisory limits that could be
exceeded by normal partitioning. Defaults remain unbounded for compatibility.
`agent.num_cpus_per_worker` also makes each Ray actor's CPU scheduling
reservation explicit; it does not claim operating-system affinity isolation.

The V1 TransferQueue worker routes every session through the same gate. It
already waits for every sibling session before publishing a terminal prompt
status, so a session exception becomes a failed prompt group instead of a
batch-wide exception. `trainer.v1.sampler.refill_all_failed_groups` now lets a
consumer require complete groups: any group with a failed session is evicted
and replaced even if other sibling trajectories are materializable. SAMPO's
existing complete-group default remains intact; other algorithms opt in.

This changes rollout admission and terminal-group handling only. It does not
change sampling, tensor packing, reward/advantage equations, optimizer
behavior, or Ray placement. The legacy non-TransferQueue manager still lacks
typed recoverable row outcomes. CPU coverage is in
`tests/experimental/agent_loop/test_episode_capacity_on_cpu.py`,
`tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py`,
`tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py`, and
`tests/trainer/ppo/v1/test_trainer_base_on_cpu.py`.

The SAMPO estimator emits batch-level episode-advantage, turn-advantage,
anchor-group-size, and sparse-reward-projection metrics from the same
hierarchical tensors used by the optimizer. Both the legacy and V1 trainer
paths retain those metrics through `DataProto.meta_info`, allowing the
framework to normalize them without deriving optimizer evidence from traces.

Regression coverage:
`tests/trainer/ppo/test_sampo_advantage_on_cpu.py` and
`tests/trainer/ppo/v1/test_trainer_base_on_cpu.py`.

### Stage LoRA adapters before waking colocated rollout weights

`verl/workers/engine_workers.py` identifies adapter-mode actors and collects the
updated LoRA tensors before waking vLLM's base weights. Adapter collection lets
the actor engine release its model allocation before rollout weights are
restored, avoiding a temporary overlap of both full model copies. The base
model is still loaded by the rollout worker and is not resent on every step.

Regression coverage:
`tests/checkpoint_engine/test_global_steps_on_cpu.py::test_lora_adapter_is_staged_before_rollout_weights_wake`.

### Qualify exact partial-rollout continuation evidence

The native `FullyAsyncLLMServerClient` remains the owner of mid-generation
abort and resume. A deterministic regression now proves that it resubmits the
original prompt plus the retained generated prefix, reduces the remaining
response budget, concatenates sampled token IDs and behavior log probabilities
exactly once, and reports the minimum and maximum served `global_steps` across
the resumed request. This is protocol qualification only; changed-weight GPU
execution remains a consumer release gate.

Regression coverage:
`tests/workers/rollout/test_llm_server_response_length_cap_on_cpu.py::test_resume_preserves_exact_tokens_logprobs_and_policy_span`.

### Honor chunked entropy for dense FSDP inputs

`verl/workers/engine/fsdp/transformer_impl.py` applies configured entropy
chunking when `use_remove_padding=False`. It flattens dense token rows, invokes
the selected chunked entropy implementation, and restores batch/sequence shape
instead of allocating an unchunked vocabulary softmax.

GPU regression coverage:
`tests/models/test_fsdp_no_padding_on_gpu.py::test_prepare_model_outputs_chunks_entropy_without_remove_padding_on_gpu`.

### Preserve Qwen 3.5 attention dispatch under FSDP2

`verl/models/transformers/qwen3_5.py` derives the decoder attention type from
the retained child module when a dynamically generated FSDP2 wrapper no longer
exposes the plain `layer_type` attribute. Native unwrapped layers continue to
use their explicit value.

CPU regression coverage:
`tests/models/test_qwen35_decoder_layer_forward_on_cpu.py`.

### Preserve response-token accounting

`verl/trainer/ppo/metric_utils.py` reports `response_length/total` alongside the
existing distribution statistics. The raw per-step denominator permits
lossless external aggregation.

CPU regression coverage:
`tests/trainer/ppo/test_metric_utils_on_cpu.py`.

### Report fp16 loss scaling and the rollout log-probability gap (post4 candidate)

Upstream v0.9 already trains FSDP in float16: `fsdp_config.mixed_precision`
with `param_dtype: fp16` (reduce and buffer dtypes fp32) makes
`FSDPEngine._build_fsdp_module` create a `ShardedGradScaler(growth_interval=400)`
that scales the loss, unscales before gradient clipping, and skips a step whose
gradients overflow, over float32 master weights (the recipe of Qi et al. 2025,
"Defeating the Training-Inference Mismatch via FP16", arXiv 2510.26788). It
reported neither the scale nor the skips, and the rollout-versus-actor metrics
compared probabilities, which hides the log-probability gap on unlikely tokens
that precision changes most.

- `FSDPEngine.optimizer_step` records `loss_scale` (the scale used for the step)
  and `optimizer_step_skipped` (1 when `GradScaler.update` lowered the scale,
  exactly when the step found inf/NaN gradients); `BaseEngine.train_batch` adds
  them to the step's metrics on the output rank and
  `EngineWorker._postprocess_output` keeps them per step like `grad_norm`, so
  the trainer logs `actor/loss_scale` and `actor/optimizer_step_skipped` (the
  skipped share of the update's optimizer steps). bf16/fp32 report nothing new.
- `calculate_debug_metrics` adds `training/rollout_logp_diff_mean`, `_p99`,
  `_max` (|log p_actor - log p_rollout| per response token) and
  `training/rollout_seq_logp_diff_abs_mean` (mean absolute per-sequence sum).

Regression: `tests/workers/test_fsdp_loss_scale_metrics_on_cpu.py` (scaled step,
overflow skip with the scale halved and weights unchanged, bf16 no-op,
`train_batch` merge, per-step worker metrics) and
`tests/utils/debug/test_metrics.py`. Consumer:
Posttrain branch `codex/precision-fp16-verl` maps these to `train/loss_scale`,
`train/optimizer_steps_skipped` and `train/rl/sampling_logp_delta_*`, and
already tolerates the infinite gradient norm of a skipped step on post3.

Build from the tagged commit. The wheel is byte-reproducible; setuptools
stamps the sdist archive with checkout and build times, so two sdist builds
have identical member contents but different bytes, and the retained release
asset is the authority for its hash:

```bash
git clone --branch carbonteq-v0.9.0.post4 https://github.com/carbonteq-ai/verl.git verl-post4
cd verl-post4
SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)" uv build --out-dir dist
sha256sum dist/verl-0.9.0.post4-py3-none-any.whl dist/verl-0.9.0.post4.tar.gz
```

### Report step-local vLLM MTP acceptance

`verl/workers/rollout/llm_server.py` reads aggregate vLLM Prometheus
speculative-decoding counters at the synchronous rollout boundary. It sums
replicas, subtracts the previous process-lifetime snapshot, handles engine
counter resets, and reports raw draft/accepted counts plus normalized
acceptance rate and acceptance length. `trainer_sync.py` captures the snapshot
before rollout sleep; `trainer_base.py` merges it into the current step.

CPU regression coverage:
`tests/workers/rollout/test_spec_decode_counter_metrics_on_cpu.py`.

### Colocate distillation teachers with the actor pool

The upstream distillation path always allocates a dedicated teacher resource
pool in addition to the actor/rollout pool. That makes a one-GPU student plus
teacher composition request two Ray GPUs even though the teacher loop already
supports colocated vLLM servers and coordinated rollout sleep/wake.

`distillation.enable_resource_pool` now selects the topology explicitly. Its
default remains `true` for backward compatibility. When false, both legacy and
V1 trainers map `Role.TeacherModel` to `global_pool` and align the teacher
topology with the trainer topology instead of adding another GPU pool.

### Align teacher log probabilities with response tokens

The teacher loop initially retains prompt log probabilities as a dense
`[batch, prompt + response, 1]` tensor. The actor update then converts them to
a jagged tensor, and engine micro-batching can produce a logical nested view
whose backing storage still contains rows from the larger optimizer batch.
The reverse-KL estimator previously passed that view to
`no_padding_2_padding`, which compared the logical sequence lengths with the
full backing-storage length and failed during the first forward/backward pass.

The estimator now slices both dense inputs and logical rows of nested
micro-batch views from `prompt_length - 1`, matching the vLLM prompt-logprob
convention that omits the first token and appends a trailing dummy row.
Distillation diagnostics also omit min, max, and absolute-loss samples for a
fully masked synthetic padding micro-batch. Its optimizer loss remains zero
under the existing global-batch normalization.

CPU regression coverage:
`tests/trainer/test_distillation_dense_teacher_logprobs_on_cpu.py`.

CPU regression coverage:
`tests/trainer/test_distillation_resource_pool_on_cpu.py`.

## Compatibility and operating constraints

- The post-training agent loop must provide `sampo_turn_spans`,
  `sampo_anchor_state_keys`, and `sampo_step_rewards` in
  `DataProto.non_tensor_batch`.
- Turn spans are half-open response-token offsets and may contain only sampled
  policy tokens.
- Step rewards must be present for every turn or absent for every turn. When
  absent, SAMPO assigns the trajectory reward to the final turn and zero to
  earlier turns before discounting.
- The actor must select `policy_loss.loss_mode=gspo` with
  `loss_agg_mode=seq-mean-token-mean`.
- Bounded reward-constant group replacement runs through the core V1 trainer.
  `algorithm.filter_groups.max_num_gen_batches` is the maximum number of full
  candidate batches per optimizer batch; non-positive values remain unbounded.
- The current qualified framework slice is limited to Qwen 3.5 and FSDP2.
- veRL does not monkey-patch vLLM. TurboQuant cache reshaping is owned by the
  separately maintained CarbonTeq vLLM fork and selected by immutable commit.
- The repository contains no candidate-specific virtual environment,
  `runtime/` lock directory, or `sitecustomize` module. TurboQuant remains
  unqualified until the consuming framework's DAPO and SAMPO GPU gates pass.

## Validation

Post5 focused CPU validation (plus `tests/trainer/ppo/v1/` for active sampling):

```bash
PYTHONPATH=$PWD python -m pytest -q \
  tests/trainer/ppo/v1/ \
  tests/trainer/ppo/test_token_clip_policy_loss_on_cpu.py \
  tests/trainer/ppo/test_core_algos_on_cpu.py \
  tests/trainer/ppo/test_rollout_corr_integration.py
ruff check verl/trainer/ppo/core_algos.py tests/trainer/ppo/test_token_clip_policy_loss_on_cpu.py
```

Post4 candidate focused CPU validation:

```bash
PYTHONPATH=$PWD python -m pytest -q \
  tests/workers/test_fsdp_loss_scale_metrics_on_cpu.py \
  tests/utils/debug/test_metrics.py \
  tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py
ruff check verl/utils/debug/metrics.py verl/workers/engine/base.py \
  verl/workers/engine/fsdp/transformer_impl.py verl/workers/engine_workers.py
```

Published SAMPO CPU validation:

```bash
.venv/bin/python -m pytest \
  tests/trainer/ppo/test_sampo_advantage_on_cpu.py \
  tests/trainer/ppo/test_core_algos_on_cpu.py -q

PATH="$PWD/.venv/bin:$PATH" bash scripts/generate_trainer_config.sh
```

The published focused CPU suite passes with 32 tests. The generator
intentionally reports generated files as changed until staged or committed.

Release-candidate source/CPU validation:

```bash
ruff check \
  setup.py \
  verl/models/transformers/qwen3_5.py \
  verl/trainer/ppo/metric_utils.py \
  verl/trainer/ppo/v1/trainer_base.py \
  verl/trainer/ppo/v1/replay_buffer.py \
  verl/trainer/ppo/v1/trainer_sync.py \
  verl/workers/engine/fsdp/transformer_impl.py \
  verl/workers/engine_workers.py \
  verl/workers/rollout/llm_server.py \
  tests/checkpoint_engine/test_global_steps_on_cpu.py \
  tests/models/test_fsdp_no_padding_on_gpu.py \
  tests/models/test_qwen35_decoder_layer_forward_on_cpu.py \
  tests/trainer/ppo/test_metric_utils_on_cpu.py \
  tests/trainer/ppo/test_sampo_advantage_on_cpu.py \
  tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py \
  tests/trainer/ppo/v1/test_trainer_base_on_cpu.py \
  tests/trainer/test_distillation_dense_teacher_logprobs_on_cpu.py \
  tests/trainer/test_distillation_resource_pool_on_cpu.py \
  tests/utils/reward_score/test_sandbox_on_cpu.py \
  tests/workers/rollout/test_llm_server_response_length_cap_on_cpu.py \
  tests/workers/rollout/test_spec_decode_counter_metrics_on_cpu.py
pytest -q \
  tests/checkpoint_engine/test_global_steps_on_cpu.py \
  tests/models/test_qwen35_decoder_layer_forward_on_cpu.py \
  tests/trainer/ppo/test_metric_utils_on_cpu.py \
  tests/trainer/ppo/test_sampo_advantage_on_cpu.py \
  tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py \
  tests/trainer/ppo/v1/test_trainer_base_on_cpu.py \
  tests/trainer/test_distillation_dense_teacher_logprobs_on_cpu.py \
  tests/utils/reward_score/test_sandbox_on_cpu.py \
  tests/workers/rollout/test_llm_server_response_length_cap_on_cpu.py \
  tests/workers/rollout/test_spec_decode_counter_metrics_on_cpu.py
git diff --check
```

Release promotion also requires the separately locked veRL environment to
resolve from this candidate and GPU qualification for:

- dense-FSDP chunked entropy;
- at least one complete LoRA RL update including adapter staging, rollout
  wake/sync, backward, optimizer update, and subsequent rollout;
- Qwen 3.5 FSDP2 old-logprob and actor-update paths;
- MTP rollout metrics across at least two steps, proving step deltas rather
  than process-lifetime totals.

SAMPO release qualification still requires a short multi-turn GPU run proving
non-zero hierarchical advantages, GSPO clipping, bounded replacement sampling,
an optimizer update, checkpoint recovery state, and exported weights or an
adapter.

## Rebase and retirement

1. Fetch `upstream` and create a new branch from the selected immutable base.
2. Reapply the maintained commits in ledger order.
3. Resolve SAMPO estimator/configuration files and runtime-delta files
   independently; their contracts are conflict-sensitive.
4. Regenerate all `_generated_ppo_*trainer.yaml` files when SAMPO configuration
   changes.
5. Run the focused CPU suites and the relevant GPU qualification gates.
6. Update this ledger's bases and published commits, then update the
   framework's immutable selection and tooling page.

Retire the SAMPO delta only when upstream provides equivalent hierarchical
turn-aware advantages, validated metadata handling, and a stable public
configuration contract. Upstream GSPO support alone is insufficient. Retire a
runtime patch only after its regression passes against upstream behavior.

## Deferred behavior

- Similarity-based fuzzy anchor grouping is intentionally unsupported; stable
  exact anchor keys are the reproducible contract.
- TurboQuant and quantization-aware cache layouts require the pinned CarbonTeq
  vLLM fork and remain unsupported until their framework GPU gates pass.
- QLoRA and quantization-aware actor loading are not part of the new runtime
  delta.
- A GPU quality or convergence claim is deferred until the applicable release
  gate runs.
- Distillation qualification belongs to the consuming framework.
