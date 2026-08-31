# CarbonTeq verl fork ledger

## Status

Gemma replicated-buffer export backport (2026-10-02): adopt merged upstream
PR #7610, commit ddb96db19850bf820fe0aa13e4cb71b21f285869. The merger's
merge_non_dtensor_shards retains one plain replicated tensor and rejects rank
disagreement instead of concatenating it. This fixes the observed ordinary
Gemma E2B single-rank scalar-buffer export crash and multi-rank buffer shapes.
Files: verl/model_merger/fsdp_model_merger.py and the upstream regression
tests/model_merger/test_fsdp_merge_non_dtensor_shards_on_cpu.py. Candidate base
d3ca05a4; native export from the retained Gemma checkpoint remains a gate.
On rebase, remove the backport once upstream includes #7610. No new upstream
PR is needed. Open #8016 separately fixes mesh discovery; it is not adopted here.

Declared native microbatch candidate (2026-10-02): engine/utils.py accepts
optional ordered micro_batch_sizes on static batches. Positive sizes must
cover all rows, preserve force_group_size, satisfy requested count limits and
agree in count across data-parallel ranks. The default uniform/static and
dynamic paths are unchanged. This preserves externally resolved execution
boundaries and a partial final pack inside the existing native backward and
single optimizer lifecycle. Regression:
tests/workers/test_declared_microbatches_on_cpu.py. CPU gradient/coverage tests
pass (14 cases, including real two-process Gloo rank validation); native GPU
packing and distributed model execution qualification remain open.
Candidate base 83c35675fcfe3dc5d0be47a66f624b0e1024ff6c; published source
7cf687284ba054e836b8ce34475a3e695b3dd22b on origin/codex/resolved-engine-worker.
Consumer pins and immutable release assets remain unchanged.
Rebase by retaining the opt-in metadata branch before default scheduling;
remove it only when upstream offers equivalent ordered partial-tail support.

Strict full-determinism candidate (2026-10-02): enable_full_determinism uses
torch.use_deterministic_algorithms(True, warn_only=False). PyTorch SDPA Flash
Attention backward remains nondeterministic in warn-only mode, even with the
existing environment flags. An instrumented ordinary LFM BF16 resume verifies
247 restored state tensors exactly, then demonstrates unequal gradients on
three identical backwards; strict mode makes all 24 gradient tensors identical
across three repeats. No optimizer update is performed by those diagnostics.
Ownership: verl/workers/engine/utils.py; regression:
tests/workers/test_full_determinism_on_cpu.py (one CPU case passes), Ruff/diff
checks pass. Base: 77fe49a9de909f036aa72d8568cc957d226b1e7c on
codex/resolved-engine-worker. Full determinism is opt-in; unsupported native
operations now raise rather than silently retaining nondeterminism. Fresh full
BF16/FP16 job continuation, distributed qualification and consumer pin/runtime
adoption remain gates. Published source:
8f0de2365f1041954b67f74df5a14c7ba0532755 on
origin/codex/resolved-engine-worker. Rebase must retain strict kernel selection.

V1 empty-batch cleanup candidate (2026-10-02): recipe extensions can apply
another optimizer update from retained evidence without allocating TransferQueue
keys. V1 fit skips kv_clear for an empty batch, preserving save/log/counter
handling. The CPU regression executes the actual fit loop over one allocated
and one empty batch; all 15 trainer-base CPU tests pass. Base:
ef5aac6ff92d5a69f72cfe222f0a409af4220314 on codex/resolved-engine-worker.
Published source: 076072b92baf336c2e18e9f38bf7434e4a5f3cb7 on
origin/codex/resolved-engine-worker. Pins and runtime adoption remain separate.

Plain V1 recipe runner extension candidate (2026-10-02): expose TaskRunnerV1Base
before Ray decoration and retain TaskRunnerV1 as its default Ray actor wrapper.
Recipes can subclass the native manager lifecycle without Ray-private metadata
or replacing the global trainer registry. Ownership: verl/trainer/main_ppo.py;
regression: tests/trainer/test_task_runner_extension_on_cpu.py. Native base
construction/subclassing and existing Ray wrapper availability pass. Consumer
lifecycle tests additionally cover queue initialization, manager initialization,
fit and failure cleanup. Base: 8ae3500d80c1ec48179d786766afe6b4931bacc0 on
codex/resolved-engine-worker. Published source:
70baba82c0b2b0a8981089ca62c5ab474efe1816 on origin/codex/resolved-engine-worker;
this is not a versioned release, GPU qualification or consumer pin adoption.
Rebase must preserve the default wrapper and plain subclassable base together.

Engine factory extension candidate (2026-10-02): TrainingWorker.create_engine
constructs the registry-selected engine before model initialization and dispatch
registration. Recipe subclasses can specialize model-output handling without
mutating an initialized engine or replacing native lifecycle ownership. Default
construction forwards the same native configuration values. Ownership:
verl/workers/engine_workers.py; regression:
tests/workers/test_engine_factory_on_cpu.py (two CPU tests, default and recipe
override; both pass against the native worker constructor with explicit platform
fakes). Ruff and diff checks pass. Branch codex/resolved-engine-worker starts
from acad5211619aef5ed80b25e9657262f08a436b03. Published source:
cad959e97471600376203d1a1d860259d32dc64d on origin/codex/resolved-engine-worker;
this is not a versioned release, native GPU/driver qualification or consumer pin
adoption. Rebase must preserve pre-dispatch construction and default forwarding.

Selected-logprob cancellation source candidate (2026-10-01): normalize with
batch-row log_softmax before gathering in logprobs_from_logits_v2 for every
dtype. Absolute selected-logit minus logsumexp corrupts values and derivatives
at large common offsets even in FP32/FP64. Equal FP32 logits1e8 returned0 and
gradient[0,-1] rather than-log2 and[.5,-.5]. Preserve shape/dtype/half behavior.
Ownership: verl/utils/torch_functional.py and
tests/utils/test_selected_logprob_precision_on_cpu.py. Five CPU cases fail
before repair; after repair, selected-score/temperature/actual FSDP route
tests pass68 cases, with two intentional nonfinite-FP16-offset skips.
Ruff/diff checks pass. Rebase must retain stable normalization and scalar
derivative regressions. Normalized FP32/FP64 vocabulary buffers add a memory
gate for larger contexts; half score-only routes already use the independently
qualified token-row helper. Native optimizer/full-controller qualification,
throughput, assets and production adoption remain separate. Extreme offsets
are demonstrated corner cases, not measured causes of current poor task quality.
Published source:661bbf395e90a060acde0ec0bbed84ef67b5cee3 on
origin/codex/posttrain-math-parity. Focused command in the compatible runtime:
python -m pytest tests/utils/test_selected_logprob_precision_on_cpu.py
tests/utils/test_temperature_precision_on_cpu.py
tests/workers/test_temperature_scoring_routes_on_cpu.py -q.
Actual-model qualification subsequently exercises both repaired backend source
bodies over four Qwen/LFM BF16/FP16 full-response fixtures. Sixteen detached-logit
score backwards and80 independent full-vocabulary scalar checks pass; maximum
score error3.39e-7 and checked coordinate derivative5.97e-10, peaks3.152GB Qwen
and4.545GB LFM with model resident. Observed unshifted scaled logits[-28.44,48.75]
do not show synthetic extreme offsets. Detached-logit memory is qualified;
complete model/optimizer backward and representative throughput remain open.

Row-wise temperature-scoring source candidate (2026-10-01): the published
full-FP32 division fixes score rounding but Qwen's8GB SAMPO backward gate fails
with it, including expandable allocator and saved-score offload controls. Native
activation offload also raises a tuple-restoration assertion after an allocation
warning; its independence from memory exhaustion is unproven.

Candidate temperature_scaled_logprobs unbinds half model logits into token rows
before FP32 scaling/log_softmax. Autograd returns half row gradients without a
full FP32 logit-gradient buffer. Both non-fused FSDP routes select this path for
half score-only requests. Entropy, sum-pi-squared and distillation retain the
published full-FP32 path and its separate memory gate. Model/head arithmetic,
temperature floor and policy/KL definitions remain unchanged. This bypasses
optional flash CE for these requests; throughput and fused-kernel integration
are adoption gates, alongside larger/distributed workloads.

Forty-two focused CPU/CUDA scalar/empty-gradient/route/distillation cases pass.
The combined utilities and route/distillation suite passes95 tests, with four
distributed cases deliberately deselected; Ruff and diff checks pass.
All four Qwen/LFM BF16/FP16 arms apply12/12 SAMPO updates with ordinary allocator
and no extra offload. Twenty-four independent losses/score derivatives,288 matrix
checks and1,152 scalar dots pass: max loss5.39e-8, derivative9.96e-11, worker
aggregate4.62e-8 and Adam4.12e-9. Thirty-two independent full-vocabulary scalar
checks agree within1.54e-7. Peak tensor allocations3.218GB Qwen and2.014GB LFM.
Initial weights match exactly; Qwen BF16 scores match the full-FP32 reference
within2.27e-6. BF16 later clipping changes substantially, FP16 counts stay equal;
first-update ratios remain1. This qualifies a source candidate, not task quality,
full controller admission/refill, runtime assets or production adoption.
Ownership: verl/utils/torch_functional.py, the two FSDP output routes, and
tests/utils/test_temperature_precision_on_cpu.py plus the existing route tests.
Rebase must preserve promotion before division and first-order derivatives
without restoring the full FP32 gradient allocation in score-only half paths.

Published row-wise source commit: 7cf23e101ba70813a0b23392465b4a6eaf073731
on origin/codex/posttrain-math-parity. Runtime assets and production pin
adoption remain separate from the completed source/memory qualification.

Temperature-scaling source candidate (2026-10-01): promote represented
BF16/FP16 logits before temperature division in both non-fused FSDP
prepare_model_outputs routes. Helper ownership is
verl/utils/torch_functional.py:scale_logits_by_temperature; integration is
verl/workers/engine/fsdp/transformer_impl.py. Preserve FP32/FP64 arithmetic,
model/head computation and the existing temperature floor. Half-input gradients
retain their input dtype. This changes policy/reference scores intentionally;
old/current scores must both be recomputed consistently on a resumed population.
It adds FP32 logit buffers and needs a memory gate at intended batch/context sizes.

Regression ownership: tests/utils/test_temperature_precision_on_cpu.py.
Packed/unpacked integration ownership:
tests/workers/test_temperature_scoring_routes_on_cpu.py; execute actual
prepare_model_outputs with the real CPU fallback normalizer and independent
scalar scores/derivatives. Four route cases plus six existing top-K distillation
compatibility cases pass. The initial CPU normalizer adapter omitted the packed
route's inplace_backward argument; that harness failure is retained externally.
Combined utility regressions pass62 CPU cases(10 CUDA/distributed deselected).
The original scaling body fails8/15 scalar score/derivative/overflow/floor cases;
the candidate passes15/15. Real Qwen/LFM BF16/FP16 logits reproduce pre-candidate
native scores exactly and show sampled-score differences up to0.080662/0.011010
and0.153319/0.021459 respectively when promoting before division. CUDA autocast
already returns FP32 fallback scores; it cannot undo prior half-division rounding.
Native LFM GRPO/GSPO/DAPO BF16/FP16 model/optimizer/private-queue replay applies
18/18 updates, passes36 independent losses/score gradients,432 matrix checks
and1,728 scalar dots. Max loss error4.59e-8, derivative7.42e-11, worker aggregate
5.08e-8 and independent Adam2.14e-9; peak3.714GB. Focused CPU/CUDA temperature
regressions pass17/17. First-gradient changes are3.07–3.13% BF16 and0.397–0.402%
FP16; later GRPO/DAPO clipping counts shift modestly while GSPO counts stay
unchanged. First-update ratios remain1. Source qualification does not establish
task-quality improvement, complete controller admission/refill, a runtime asset
release or production adoption. No production dependency pin selects it yet.
Fused scoring and other engine types
are outside this candidate. Rebase must preserve promotion before division in
both packed and unpacked routes and independent probability/derivative checks.

Duplicate-work check: upstream open PR8004 adds opt-in FP32 lm_head projection,
preserves default behavior and excludes LoRA. This candidate scales existing
represented logits without changing the head. No upstream PR is proposed.
Focused command: PYTHONPATH=. python -m pytest -c /dev/null -p no:cacheprovider
tests/utils/test_temperature_precision_on_cpu.py -q. Run in the compatible native
runtime with auxiliary dependency paths appended after its Torch/Transformers.

Published temperature-scaling source commit:
f5333c4f647494e497896eaed14160e2cd7186c4 on origin/codex/posttrain-math-parity.
This identifies the tested correction and regressions; runtime assets/pins
and the remaining controller/fused/quality gates are separate.

Entropy precision repair (2026-10-01): `verl/utils/torch_functional.py`
computes entropy from normalized log probabilities instead of subtracting
two common-offset-sized values. Half inputs use float32 arithmetic; float32
and float64 retain their arithmetic dtype. Zero-probability terms have finite
value and gradient even when finite extreme logit differences overflow.
Chunked entropy delegates to the same arithmetic. This intentionally changes
unchunked half entropy outputs to float32; inputs retain their gradient dtype.
It does not change policy/KL formulas or enable chunking by default.

Independent represented-logit controls reproduce BF16[10,11] entropy0.5625
versus0.582203, and uniform high-offset cancellation to0, including chunked
float32 at1e8. Before repair,24 of36 new CPU regressions fail; after repair,
all36 pass. Six CUDA BF16/FP16/FP32 value/gradient controls also pass.
The related non-distributed utility slice passes53 tests (four distributed
cases deliberately deselected): `PYTHONPATH=. python -m pytest -c /dev/null
-p no:cacheprovider tests/utils/test_entropy_precision_on_cpu.py
tests/utils/test_torch_functional.py -k 'not distributed' -q`.
The17-case external scalar probe's maximum repaired value error is1.87e-8.
Regression ownership: `tests/utils/test_entropy_precision_on_cpu.py`.
Rebase/retirement requires equivalent stable values, derivatives, half-input
promotion and zero-probability behavior. Full-model/runtime assets and distributed
entropy execution remain qualification gates; this is a source candidate.

Published entropy source commit: `c1e477d7d83badb4be6742c9efde483956698f01`.
Actual Qwen BF16 native controller old/reference method bodies, private real
TransferQueue and GPU worker pass with64-token entropy chunks and Posttrain's
seq-mean-token-mean aggregation. Independent causal score slices and masks
are exact; reference/base identity and actor restoration pass. After three
updates the actor differs from the base by1.03025 at the largest response-score
coordinate, so the reference check is nontrivial. Entropy aggregation agrees
with scalar row means within5.41e-9. Parameters/scores at steps0–3 and all three
preclip gradients remain bitwise identical to the pre-repair control. The
actual entropy metric changes only1.19e-7 on this trace; no task-quality or
production-cause conclusion follows. Unchunked controller entropy exceeded
the8GB memory budget before repair; chunking remains an explicit probe setting.
Full controller construction, Ray GPU dispatch/admission/refill and other
families/precisions remain separate integration gates.

Single-rank FSDP accumulation repair (2026-10-01): keep gradient synchronization
enabled when the data-parallel group has one rank. There is no cross-rank
communication to defer. Torch2.13 FSDP2's deferred-sync post-backward path can
access an uninitialized `_unsharded_param` in an independently sharded branch
that never ran, including unused vision modules in text-only Qwen training.
Native minimal frozen/trainable and CPU-offload/device controls all fail with
deferred sync and all pass with ordinary sync. This is a runtime callback
failure, not a SAMPO formula error. Multi-rank deferral remains unchanged;
multi-rank conditional-unused-branch behavior is not qualified by this repair.

Source: `verl/workers/engine/fsdp/transformer_impl.py`. Six new regressions in
`tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py` fail before the
repair and pass afterward. The complete12-test suite passes, including four
actual CUDA BF16/FP16 offload/device accumulation cases with an independent
closed-form gradient and the existing two-rank Gloo FSDP1/FSDP2 equivalence test:
`PYTHONPATH=. python -m pytest -c /dev/null -p no:cacheprovider tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py -q`.

A complete collected Qwen3.5-0.8B task population now executes through FSDP2
CPU offload:1,119 prompt tokens,289/420 response tokens,133/165 sampled actions,
rank4/alpha8 q/v LoRA, two accumulated microbatches, scale1024 and two applied
FP16 SAMPO updates. Independent loss/score/mask and Adam checks pass. Rewards
are both1; the direct probe deliberately bypasses production reward-constant
admission. This does not qualify fresh learning, runtime pins or distributed
conditional-branch execution. Retire the single-rank guard only after equivalent
unused-branch and accumulation regressions pass under adopted upstream behavior.

The matching Qwen BF16 and FP16 FP32-delta-region diagnostic arms also apply
two updates each; all12 Qwen loss/score/mask checks pass, with maximum Adam
error2.24e-9 and3.67GiB peak Torch allocation. A native LFM1.2B FP16 SAMPO
regression applies two updates with48 aggregated LoRA matrix-gradient checks
and192 independent scalar dots; all manual native gradients match exactly,
Adam error is at most2.14e-9 and peak allocation2.06GiB. These remain bounded
correctness probes, not convergence evidence or production runtime adoption.

FSDP FP16 checkpoint-state candidate (2026-10-01): the engine now binds its
optional scaler to `FSDPCheckpointManager`. Per-rank extra state persists the
loss scale, growth/backoff configuration and growth tracker, alongside RNG
and scheduler state. Full restore with an enabled scaler validates presence
before model/optimizer loading; older checkpoints without scaler state fail
clearly. Explicit model-only loading, BF16 and disabled-scaler legacy extra
state remain compatible. Source delta: `verl/utils/checkpoint/fsdp_checkpoint_manager.py`
and `verl/workers/engine/fsdp/transformer_impl.py`; five new regressions are
in `tests/checkpoint_engine/test_fsdp_lora_only_checkpoint_on_cpu.py`.

The native LFM1.2B FP16 FSDP2 CPU-offload probe saves LoRA-only model state
with full optimizer/extra state at scale512 and growth tracker1. Same-process
restore and a fresh engine in another process both reproduce the next update
exactly: adapter parameters, optimizer moments, scaler, scheduler and RNG.
The control omitting scaler binding restores every other field but keeps
scale1024/tracker0 and fails. Independent loss/score and AdamW checks pass;
peak allocated Torch memory is2.06GiB. Native distributed/full-weight,
other architectures and production wheels/pins remain unqualified.
Checkpoint, cleanup and scaler slices pass34 tests together:
`PYTHONPATH=. python -m pytest -c /dev/null -p no:cacheprovider tests/checkpoint_engine/test_fsdp_lora_only_checkpoint_on_cpu.py tests/utils/ckpt/test_checkpoint_cleanup_on_cpu.py tests/utils/test_sharded_grad_scaler.py -q`.

Rebase must preserve scaler binding, per-rank extra-state persistence and
early missing-state validation. Remove this delta only when adopted upstream
checkpoint handling provides equivalent exact-replay evidence. No released
runtime or dependency pin is changed. The native probe uses two logical
updates plus replay attempts, not three different updates.

CPU-offload FP16 unscale candidate (2026-10-01): Torch2.13's sharded scaler
uses nonblocking scalar replication. A CPU foreach kernel can read a
CUDA-to-CPU inverse-scale copy before completion. A delayed scalar control
on ordinary CPU gradients fails three of four iterations at scale1024;
affected gradients are multiplied by1024 rather than divided by1024.
Synchronous scalar reads hide the failure. This is not DTensor-specific.

`verl/utils/sharded_grad_scaler.py` adds `CPUOffloadShardedGradScaler`;
the FSDP engine in `verl/workers/engine/fsdp/transformer_impl.py` selects it.
Only gradients with CPU storage trigger synchronous host staging of inverse
scale and overflow scalars. Device-only scaling and the parent's distributed
overflow reduction remain native. `tests/utils/test_sharded_grad_scaler.py`
checks finite/overflow CPU gradients, deliberately delayed CUDA-to-CPU copies
and the unchanged CUDA-only path: five tests pass on the local CUDA runtime.
Native LFM1.2B BF16/FP16 offload runs each apply two updates, with all48
linear parameter gradients matching an independent `D.T @ X` construction
exactly and192 selected scalar dot-product checks per arm. These checks do
not independently differentiate nonlinear model blocks or qualify multi-rank
overflow/recovery. Source and release adoption remain separate gates.

Base for this delta:8778c5d6e2ddd847d5098a24f4dc882f11ac57b4. Focused command:
`PYTHONPATH=. python -m pytest -c /dev/null -p no:cacheprovider tests/utils/test_sharded_grad_scaler.py -q`.
Run with CUDA before release; CPU-only execution skips the three CUDA cases.
Rebase must preserve host readiness before CPU kernels. Retire this wrapper
only after an adopted Torch scaler provides equivalent ordering and the
delayed-copy regressions pass. No runtime wheel or Posttrain pin is changed.

Additional lifecycle qualification (2026-10-01): the scaler regression file
now has eight passing tests. A finite/overflow/finite/finite sequence verifies
that restoring parameter, AdamW moments and scaler state after the skipped
step matches uninterrupted execution exactly, on CPU scaling/Gloo and CUDA
scaling/NCCL with CPU gradients. CUDA uses the delayed-copy control. A separate
two-rank Gloo test injects overflow on rank0 only; both ranks skip, back off
to scale512 and apply the next finite update. These are tensor/scaler tests,
not a native-model checkpoint round trip or multi-GPU engine qualification.

Native padded-engine repair candidate (2026-10-01): eager/SDPA execution
must not require the optional FlashAttention package just for indexing.
When that package is absent, attention_utils reuses the existing Torch
padding implementation; broken transitive imports still propagate.
Padded Qwen3.5 without sequence parallelism or fused kernels retains the
native Transformers forward instead of the legacy packed replacement.
The latter expects attributes absent from newer Transformers models.
Four padding-fallback checks, eight retained padding checks and two Qwen
forward-selection regressions pass. A real single-rank FSDP1 Qwen0.8B
BF16 probe loads FP32 masters, computes in BF16, applies two native AdamW
updates and passes four independent loss checks. FP16 scale1 also applies
updates in this token fixture, while scales128/1024 can overflow. Broader
native masks/gradients/optimizer-oracle checks are recorded in Posttrain.
No released runtime adoption, distributed qualification or packed-path
compatibility with newer Transformers is implied.

Finer KL transition correction (2026-10-01): a shared represented-input
Decimal80 sweep finds 268/2332 failures around the candidate's 0.05 boundary
at relative tolerance 1e-6. TRL and veRL agree, exposing why parity alone is
insufficient. A tenth-order polynomial through absolute delta 0.25 removes
the remaining cancellation; truncation is below 2e-12 relative in exact
arithmetic. The expanded 3880-case sweep passes, including immediate typed
neighbors of 0.01, 0.05 and 0.25. The dedicated math/hierarchy slice passes
87 tests after adding twenty half-input regressions. This remains a source
candidate without release assets, production mapping or pin adoption.

**Unpublished Posttrain math parity candidate.** Isolated branch
`codex/posttrain-math-parity` starts from runtime-pinned
`ef1c37715fa75de5973ae5b3c398383cd7e0093d`. No release version, asset,
dependency pin or production recipe is changed by this candidate.

The delta is in `verl/trainer/ppo/core_algos.py` and
`verl/workers/utils/losses.py`, with independent regressions in
`tests/trainer/ppo/test_posttrain_math_on_cpu.py`:

- `k3_unclipped` uses a sixth-order small-delta series to preserve its value
  and derivative near zero. Half scores are promoted before subtraction.
  The unused polynomial branch is bounded to avoid overflowing backward.
  The estimator, coefficient and legacy derivative convention are unchanged.
- `ppo_loss` neutralizes excluded policy, behavior, reference, advantage,
  correction-weight and entropy entries before nonlinear loss arithmetic.
  This prevents invalid tool/padding scores from poisoning valid gradients.
- Opt-in `sampo_token_credit` uses geometric sequence ratios with the
  GSPO-token local derivative and declared asymmetric clipping, without
  native GSPO's additional log-ratio cap. Native `gspo` remains unchanged.

Upstream duplicate check found PR
<https://github.com/verl-project/verl/pull/6349>, which masks invalid values
at sequence aggregation. This candidate instead repairs the earlier
ratio/exponential boundary, including its backward path. No duplicate
upstream PR is proposed. Rebase conflicts should preserve early score masking,
small-KL derivatives and the distinction between native GSPO and SAMPO.

CPU validation uses the existing Ray/Hydra/Torch CPU parity environment with
this checkout on `PYTHONPATH`:

```bash
python -m pytest -c /dev/null -p no:cacheprovider \
  tests/trainer/ppo/test_posttrain_math_on_cpu.py \
  tests/trainer/ppo/test_sampo_advantage_on_cpu.py -q
```

Result: 130 passed, including 14 Decimal KL checks, 12 BF16/FP16 small-KL
checks, 10 independent signed-credit cases, 18 actual PPO-wrapper masking
checks, 12 retained hierarchy tests and 63 existing core/loss regressions.
The half-score boundary tests exposed value cancellation when half rounding
moved delta0.01 outside the original series region; the series now covers
absolute delta through0.05. Its omitted terms are below FP32 rounding there.
The original wrapper, substituted with candidate kernels held fixed, fails
14/18 mask tests; the corrected wrapper passes all18. Posttrain's48 existing
settings/algorithm/sampling/curriculum parity tests also pass on this checkout.
The initial KL negative control failed 12/14 near-zero cases. The new policy
loss was absent from the baseline registry. Model/distributed BF16/FP16
qualification, Posttrain capability/mapping adoption, publication and immutable
runtime pins remain open. CPU wrapper tests do not qualify unavailable optional
model engines or production training.

**Release candidate `0.9.0.post8`.** Post8 is the post7 asset receipt
(`07ecac23`) plus one fix, "Agent-loop defaults declared in the trainer
config" under "Maintained delta": since post2 the agent-loop manager read
`agent.num_cpus_per_worker`, `agent.max_concurrent_episodes` and
`agent.max_concurrent_episodes_per_worker` from the struct trainer config,
but `rollout.yaml` never declared them, so any agent-loop run that did not
override all three failed at start with `ConfigAttributeError`. No dependency
change. Branch: `codex/agent-loop-config-defaults`. The wheel and sdist
SHA-256 values of the retained assets are recorded in the receipt commit that
follows the tag.

**Release candidate `0.9.0.post7`.** Post7 is the post6 publication receipt
(`100a0a88`) plus one fix, "LoRA tensor sync keeps constituent module names"
under "Maintained delta": veRL's in-memory LoRA sync to vLLM renamed stacked
constituents (`q_proj`/`k_proj`/`v_proj`, LFM2's `w1`/`w3`) onto one module, so
any LoRA on a packed vLLM layer failed at the first weight sync. It blocks
LFM2.5 on veRL (Posttrain `docs/plan/verl-vortex-port.md`, Phase 4). No
dependency change. Branch: `codex/vortex-lora-sync`.

Its immutable release commit is
`6069abe14e2b3d27c89815a6502b849f15124e12`, tagged `carbonteq-v0.9.0.post7`
(annotated tag object `b60890dc0f453a975111e83318e35281e10d6d77`). Built from two clean clones with
`SOURCE_DATE_EPOCH` set to the commit time (`1790629033`): the wheel is
byte-identical across both builds and the two sdists have identical members.
Retained wheel `verl-0.9.0.post7-py3-none-any.whl` SHA-256:
`9786ec44fbdba367791d8e9a4c58a895955639c79c4b9dd0e3a403a05b5e29c5`; retained
sdist `verl-0.9.0.post7.tar.gz` SHA-256:
`3b0259523476d67aff9282b2b3c775c081bd866c04d369ad7a5eda8af6607088`. `twine
check` passes for both. The branch push, tag push, GitHub release and the
Posttrain retained-asset publisher (`publish-verl-internal.yml`) are pending.

**Release candidate `0.9.0.post6`.** Post6 is the post5 asset receipt
(`9c10bd1a`) plus the VORTEX and SAMPO deltas of Posttrain's
`docs/plan/verl-vortex-port.md` (phases 2 to 5), each marked "(post6)" under
"Maintained delta": round-based active sampling
(`algorithm.active_sampling`), the `data.prompt_selector` extension point with
checkpointed selector state, the `sequence_clip` policy loss and SAMPO
hierarchy evidence metrics, the TRL-equivalent GRPO settings (sampler
correction lower bound and log-ratio bound, TRL-mode GRPO statistics with
group or batch scope, row exclusion, failed-group admission retries, the
`linear` learning-rate schedule) and candidate-batch DAPO
(`algorithm.filter_groups.candidate_batches`), plus an LFM2 LoRA export
regression test. It changes no dependency. Every new behaviour is opt-in: with
the new keys at their defaults the trainer behaves as post5. Branch:
`codex/vortex-active-sampling`.

Its immutable release commit is
`1badbebd22aee7af5b85185760275f697af3a073`, tagged `carbonteq-v0.9.0.post6`
(annotated tag object `3a7eb765060cf2d691114cb8cfefd2c5a84da180`). Built from
two clean clones with `SOURCE_DATE_EPOCH` set to the commit time
(`1790624464`): the wheel is byte-identical across both builds and the two
sdists have identical members. Retained wheel
`verl-0.9.0.post6-py3-none-any.whl` SHA-256:
`63efd613e15f0011e840fadbb67e7d053353aa7b31437c84555e0261331faee5`;
retained sdist `verl-0.9.0.post6.tar.gz` SHA-256:
`c7a00ceb1858011799ec22f8d82d66f07632e1eb4ecb26e95bb207c4e1abeb1a`.
`twine check` passes for both. Published: GitHub release
<https://github.com/carbonteq-ai/verl/releases/tag/carbonteq-v0.9.0.post6>,
Posttrain retained-asset publisher run
<https://github.com/carbonteq-ai/posttrain/actions/runs/36474625071>;
`carbonteq/dev` serves both hashes. Posttrain branch
`codex/verl-vortex-active-sampling` pins it for the `online-rl-verl-py313` kind
(release 0.4.13).

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

### Agent-loop defaults declared in the trainer config (post8)

"Bounded agent-loop episode admission" (post2) added `num_cpus_per_worker`,
`max_concurrent_episodes_per_worker` and `max_concurrent_episodes` to
`AgentLoopConfig`, and `AgentLoopManager.__init__` and `AgentLoopWorker` read
them as attributes of the Hydra struct config. `rollout/rollout.yaml` did not
declare them, so the composed config had no such keys and a run that did not
pass all three as overrides failed before its first rollout
(`omegaconf.errors.ConfigAttributeError: Key 'num_cpus_per_worker' is not in
struct`). `rollout.yaml` now declares them with the dataclass defaults (1.0,
null, null: one reserved CPU per worker and unbounded fan-out, the pre-post2
behaviour), and the four `_generated_*` reference configs are regenerated.
Found by Posttrain qualification `q0412h-verl-qwen08b-bf16-r1` (a veRL job
with no `rollout_execution`). Regression test
`test_default_trainer_config_declares_every_agent_loop_field` in
`tests/experimental/agent_loop/test_episode_capacity_on_cpu.py` composes the
default `ppo_trainer` config, requires every `AgentLoopConfig` field the
manager reads, and fails on post7.

### LoRA tensor sync keeps constituent module names (post7)

`VLLMHijack`'s `_load_adapter` (`verl/utils/vllm/utils.py`), which loads the
actor's LoRA tensors into vLLM (`TensorLoRARequest`), named LoRA modules with
the model's full `hf_to_vllm_mapper`. Its stacked maps (`q_proj`/`k_proj`/
`v_proj` -> `qkv_proj`, LFM2's `w1`/`w3` -> `w13`) are for base weights: on
LoRA names they collapse the constituents onto one name, the last one wins,
and vLLM's merged column layer fails in `set_lora` (`IndexError: tuple index
out of range`). vLLM's own loader uses `hf_to_vllm_mapper.get_rename_mapper()`;
the new `lora_weights_mapper` does the same (falling back to the full mapper
on vLLM releases without it) and passes the model's `lora_skip_prefixes`.
Found by Posttrain run `verl-vortex-lfm12-check-20260929-r3` (LFM2.5-1.2B,
all-linear LoRA, post6); Qwen3.5 runs never hit it because their LoRA targets
only `o_proj`/`down_proj`. Regression test
`tests/utils/test_vllm_lora_rename_mapper_on_cpu.py` (needs vLLM; run it in the
kind image) keeps `w1`, `w3`, `q_proj`, `k_proj` and `short_conv.in_proj`
apart and fails with the full mapper. GPU check (RTX 3070 Ti, post6 kind image
with this source mounted): an LFM2.5-1.2B all-linear adapter synced as tensors
scores a 29-token text within 0.055 nats/token of PEFT (vLLM's base-model gap
is 0.040), against an adapter effect of 1.03 nats/token.

### LFM2 LoRA export regression test (post6)

No source change: LFM2.5 (hybrid short-convolution and attention blocks, tied
input and output embeddings) trains through the generic FSDP2, PEFT and vLLM
LoRA paths. `tests/model_merger/test_lfm2_lora_export_on_cpu.py` pins the
export: a PEFT `all-linear` adapter on a tiny `Lfm2ForCausalLM` exports the
eight projection targets (`q_proj`, `k_proj`, `v_proj`, `out_proj`,
`in_proj`, `w1`, `w2`, `w3`), drops the tied `lm_head`, and reloads with
identical logits.

### TRL-equivalent settings: sampler correction bounds, GRPO scaling, row exclusion, admission, linear LR (post6)

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
- Candidate-batch DAPO (`algorithm.filter_groups.candidate_batches`,
  `CandidateBatchReplayBuffer`): TRL's `_prepare_dynamic_sampling_inputs`. Each
  step reserves `max_num_gen_batches * train_batch_size` candidates (the next
  dataloader prompts, or one prompt-selector `initial_batch` decision), then
  dispatches whole candidate batches of `train_batch_size` in order, completes
  each (admission retries in place), observes it, keeps groups with spread and
  stops once the batch is full, keeping the first groups in candidate order.
  The reward std of each candidate batch's admitted trajectories (TRL's
  `nanstd`) is recorded per kept group and used for `grpo_std_scope=batch`.
  Metrics: TRL's `dynamic_sampling/candidate_batches` and
  `dynamic_sampling/retained_fraction`. The streaming DAPO refill remains the
  default when the flag is off.

CPU regression coverage: `tests/trainer/ppo/v1/test_trainer_base_on_cpu.py`,
`tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py`,
`tests/trainer/ppo/v1/test_active_sampling_on_cpu.py`,
`tests/utils/test_linear_lr_schedule_on_cpu.py`.

### Sequence-ratio clip loss and SAMPO hierarchy evidence (post6)

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

### Prompt selector extension point (post6)

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

### Round-based active sampling in the synchronous V1 trainer (post6)

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

Post6 focused CPU validation (2026-09-29: 348 passed, 2 skipped, on Python
3.13 with torch 2.13 CPU, transformers 5.14, peft 0.19, ray 2.56.1,
transferqueue 0.1.8):

```bash
PYTHONPATH=$PWD python -m pytest -q \
  tests/trainer \
  tests/model_merger \
  tests/utils/test_linear_lr_schedule_on_cpu.py
ruff check $(git diff --name-only 9c10bd1a HEAD -- '*.py')
bash scripts/generate_trainer_config.sh && git diff --exit-code
```

Posttrain's parity tests run TRL post11's real code against this delta
(`packages/train/tests/test_verl_*_parity.py`, 49 passed).

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
