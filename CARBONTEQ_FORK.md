# CarbonTeq verl fork ledger

## Status

**Published SAMPO candidate with an active release-delta candidate.** The
maintained CarbonTeq fork and earlier SAMPO branch are published. The current
SAMPO telemetry, local Python 3.13, and immutable vLLM-selection changes are
maintained on `codex/sampo-v1-metadata`; they remain candidates until this
ledger and implementation are committed, pushed, selected by the framework's
dependency-only lock, and exercised by the bounded GPU gates.

Upstream repository: `https://github.com/verl-project/verl.git`

Upstream base for the published SAMPO line:
`a35908ca3c9632859c58d6a2855d858918ae21dc`

Reconstruction base for the published runtime delta:
`553280b88afe4e7fbc4aefeff27bbf0a22e7c048`

Expected remotes:

- `origin`: `git@github.com:carbonteq-ai/verl.git`
- `upstream`: `https://github.com/verl-project/verl.git`

Published SAMPO implementation commit:
`8a718e5be7a107587f63967336ece333a5c160e1`.

## Maintained delta

### Qwen 3.5 QLoRA and runtime counters

The published branch includes the earlier candidate commits `05f83242` and
`5da56132`. They add FSDP QLoRA support for Qwen 3.5 and retain
speculative-decoding runtime totals. Their consumer-facing qualification state
is recorded in `docs/tooling/verl/README.md` in the post-training framework
repository.

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

Unpublished runtime-delta source/CPU validation:

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
  tests/utils/reward_score/test_sandbox_on_cpu.py \
  tests/workers/rollout/test_spec_decode_counter_metrics_on_cpu.py
pytest -q \
  tests/checkpoint_engine/test_global_steps_on_cpu.py \
  tests/models/test_qwen35_decoder_layer_forward_on_cpu.py \
  tests/trainer/ppo/test_metric_utils_on_cpu.py \
  tests/trainer/ppo/test_sampo_advantage_on_cpu.py \
  tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py \
  tests/trainer/ppo/v1/test_trainer_base_on_cpu.py \
  tests/utils/reward_score/test_sandbox_on_cpu.py \
  tests/workers/rollout/test_spec_decode_counter_metrics_on_cpu.py
git diff --check
```

Runtime-delta release also requires the separately locked veRL environment to
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
