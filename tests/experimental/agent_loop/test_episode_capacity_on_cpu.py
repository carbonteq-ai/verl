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
    ("num_workers", "max_concurrent_episodes", "max_concurrent_episodes_per_worker"),
    [
        (0, None, None),
        (4, 0, 8),
        (4, 32, 0),
        (4, 32, None),
        (4, 31, 8),
    ],
)
def test_episode_capacity_rejects_unenforceable_contract(
    num_workers: int,
    max_concurrent_episodes: int | None,
    max_concurrent_episodes_per_worker: int | None,
) -> None:
    with pytest.raises(ValueError):
        validate_agent_loop_episode_capacity(
            num_workers=num_workers,
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
