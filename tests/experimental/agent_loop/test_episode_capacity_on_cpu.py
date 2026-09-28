"""CPU-only tests for agent-loop episode concurrency validation."""

import asyncio

import pytest

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, validate_agent_loop_episode_capacity


def test_episode_capacity_accepts_enforceable_contract() -> None:
    validate_agent_loop_episode_capacity(
        num_workers=4,
        max_concurrent_episodes=32,
        max_concurrent_episodes_per_worker=8,
    )


@pytest.mark.parametrize(
    ("num_workers", "num_cpus_per_worker", "max_concurrent_episodes", "max_concurrent_episodes_per_worker"),
    [
        (0, 1.0, None, None),
        (4, 0.0, 32, 8),
        (4, 1.0, 0, 8),
        (4, 1.0, 32, 0),
        (4, 1.0, 32, None),
        (4, 1.0, 31, 8),
    ],
)
def test_episode_capacity_rejects_unenforceable_contract(
    num_workers: int,
    num_cpus_per_worker: float,
    max_concurrent_episodes: int | None,
    max_concurrent_episodes_per_worker: int | None,
) -> None:
    with pytest.raises(ValueError):
        validate_agent_loop_episode_capacity(
            num_workers=num_workers,
            num_cpus_per_worker=num_cpus_per_worker,
            max_concurrent_episodes=max_concurrent_episodes,
            max_concurrent_episodes_per_worker=max_concurrent_episodes_per_worker,
        )


def test_worker_episode_gate_limits_concurrent_agent_loops() -> None:
    async def run() -> None:
        worker = object.__new__(AgentLoopWorker)
        worker._episode_semaphore = asyncio.Semaphore(2)
        active = 0
        peak_active = 0

        async def run_agent_loop(*args, **kwargs):
            nonlocal active, peak_active
            del args, kwargs
            active += 1
            peak_active = max(peak_active, active)
            await asyncio.sleep(0)
            active -= 1
            return object()

        worker._run_agent_loop = run_agent_loop
        await asyncio.gather(
            *[
                worker._run_agent_loop_with_capacity({}, {}, agent_name="test", trace=False)
                for _ in range(5)
            ]
        )
        assert peak_active == 2

    asyncio.run(run())


def test_default_trainer_config_declares_every_agent_loop_field() -> None:
    """The agent-loop manager reads these keys from the struct config, so a
    trainer launched without overrides must find each one declared."""
    import dataclasses
    import os

    from hydra import compose, initialize_config_dir

    from verl.workers.config import AgentLoopConfig

    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "verl", "trainer", "config")
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")
    agent = config.actor_rollout_ref.rollout.agent
    declared = set(agent.keys())
    required = {f.name for f in dataclasses.fields(AgentLoopConfig)} - {"agent_loop_manager_class"}
    assert required <= declared, sorted(required - declared)
    validate_agent_loop_episode_capacity(
        num_workers=agent.num_workers,
        num_cpus_per_worker=agent.num_cpus_per_worker,
        max_concurrent_episodes=agent.max_concurrent_episodes,
        max_concurrent_episodes_per_worker=agent.max_concurrent_episodes_per_worker,
    )
