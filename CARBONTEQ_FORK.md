# CarbonTeq verl fork ledger

## Status

**Release candidate `0.9.0.post1`, retained GitHub prerelease.**
Release commit: `cec7e74c361bb973b641db8dfbb75a5544c33139`.
Tag: `carbonteq-v0.9.0.post1`.
Wheel SHA-256: `3d66ac6b78848ef591dd6e4be4d324fcac12c27247a05ad1c14dcf9fb370256e`.
Source SHA-256: `6da77c10e5d37399c655b15e3ed78d520be2ca648ed763d43ccb7404dfe53d72`.
Both hashes match GitHub asset digests. Sixty-three retained-feature tests pass
against the installed wheel, independent of source-checkout imports.
Development publisher: Posttrain Actions run `34006221953`; stable promotion
and runtime-image qualification remain open. This candidate integrates the
maintained consumer pin `808923d487aa2c524fda02cf5289110541b4221f` onto stable
upstream v0.9.0. Branch: `codex/verl-0.9-retained-features`.
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
`7817d845727af570352622dc8d58f2d43c76d89d` in the `vllm` extra. That vLLM
commit is based on upstream 0.25.1 and carries the bounded TurboQuant
cache-reshape correction documented in the vLLM fork's `CARBONTEQ_FORK.md`.

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
