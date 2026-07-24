# CarbonTeq verl fork ledger

## Status

**Published candidate.** The maintained CarbonTeq fork exists and the SAMPO
implementation branch is published. GPU qualification remains a release gate,
so this is a reproducible candidate dependency rather than a production-
qualified backend.

Upstream repository: `https://github.com/verl-project/verl.git`

Upstream base: `a35908ca3c9632859c58d6a2855d858918ae21dc`

Expected remotes after publication:

- `origin`: `git@github.com:carbonteq-ai/verl.git`
- `upstream`: `https://github.com/verl-project/verl.git`

Current local branch: `codex/sampo-agentic-advantages`

Published SAMPO implementation commit:
`8a718e5be7a107587f63967336ece333a5c160e1`.

## Maintained delta

### Qwen 3.5 QLoRA and runtime counters

The local branch includes the earlier candidate commits
`05f83242` and `5da56132`. They add FSDP QLoRA support for Qwen 3.5 and retain
speculative-decoding runtime totals. Their consumer-facing qualification state
is recorded in `docs/tooling/verl/README.md` in the post-training framework
repository.

### SAMPO hierarchical multi-turn advantages

Upstream verl provides the GSPO policy-loss kernel but not SAMPO's
GiGPO-style episode and anchor-state-relative advantage estimator.

The candidate delta:

- registers `AdvantageEstimator.SAMPO` in
  `verl/trainer/ppo/core_algos.py`;
- computes token-aligned joint advantages from trajectory reward, ordered
  assistant-turn spans, anchor-state keys, and optional step rewards;
- routes agent-loop metadata through
  `verl/trainer/ppo/ray_trainer.py`;
- defines typed algorithm configuration in
  `verl/trainer/config/algorithm.py` and `ppo_trainer.yaml`;
- regenerates every `_generated_ppo_*trainer.yaml` reference config;
- covers the equations, sparse/explicit rewards, masks, invalid metadata, and
  trainer routing in
  `tests/trainer/ppo/test_sampo_advantage_on_cpu.py`.

The semantic reference is the official ARL-Arena SAMPO implementation at
`a25a2a229c85431b421ac785fa5f375a99b2072a`. That implementation maps SAMPO
to GiGPO hierarchical advantages plus the GSPO loss. This fork keeps one
complete multi-turn trajectory in each batch row and applies the same
hierarchical equations to token-aligned turn spans.

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
- Bounded reward-constant group replacement is supplied by the pinned
  `verl-recipe` DAPO trainer; it is not implemented by the base trainer.
- The current qualified framework slice is limited to Qwen 3.5 and FSDP2.

## Validation

From this repository:

    .venv/bin/python -m pytest \
      tests/trainer/ppo/test_sampo_advantage_on_cpu.py \
      tests/trainer/ppo/test_core_algos_on_cpu.py -q

    PATH="$PWD/.venv/bin:$PATH" bash scripts/generate_trainer_config.sh

The focused CPU suite passes with 32 tests. The generator intentionally reports
the generated files as changed until they are staged or committed.

Release qualification still requires a short multi-turn GPU run that proves
non-zero hierarchical advantages, GSPO clipping, bounded replacement sampling,
an optimizer update, checkpoint recovery state, and exported weights or an
adapter.

## Rebase and retirement

1. Fetch `upstream` and create a new branch from the selected immutable base.
2. Reapply the maintained commits in ledger order.
3. Resolve `core_algos.py`, `ray_trainer.py`, `algorithm.py`, and
   `ppo_trainer.yaml` carefully; their estimator registry and config structure
   are conflict-sensitive.
4. Regenerate all `_generated_ppo_*trainer.yaml` files.
5. Run the focused CPU suite and the GPU qualification gate.
6. Update this ledger's base and published commit, then update the framework's
   immutable selection and tooling page.

Retire the SAMPO delta only when upstream provides equivalent hierarchical
turn-aware advantages, validated metadata handling, and a stable public
configuration contract. Upstream GSPO support by itself is insufficient.

## Deferred behavior

- Similarity-based fuzzy anchor grouping is intentionally unsupported; stable
  exact anchor keys are the reproducible contract.
- A GPU quality or convergence claim is deferred until the release gate runs.
- Production qualification is deferred until the GPU gate passes.
