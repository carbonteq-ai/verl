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

import asyncio

from verl.trainer.ppo.v1.agent_loop_tq import _run_session_with_capacity, _settle_session_tasks


def test_settle_session_tasks_waits_for_siblings_after_failure():
    async def run():
        settled = asyncio.Event()

        async def fail():
            raise RuntimeError("session failed")

        async def finish_later():
            await asyncio.sleep(0.01)
            settled.set()

        tasks = [asyncio.create_task(fail()), asyncio.create_task(finish_later())]
        errors = await _settle_session_tasks(tasks)

        assert settled.is_set()
        assert all(task.done() for task in tasks)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    asyncio.run(run())


def test_transfer_queue_session_uses_worker_episode_gate():
    class FakeWorker:
        def __init__(self):
            self.calls = []

        async def _run_agent_loop_with_capacity(self, sampling_params, **kwargs):
            self.calls.append((sampling_params, kwargs))
            return "completed"

    async def run():
        worker = FakeWorker()
        result = await _run_session_with_capacity(
            worker,
            {"temperature": 0.7},
            {"step": 3},
            trace=False,
            session_id=2,
            prompt={"uid": "group-a", "example_id": "example-a"},
        )
        assert result == "completed"
        assert worker.calls == [
            (
                {"temperature": 0.7},
                {
                    "trajectory": {"step": 3},
                    "trace": False,
                    "session_id": 2,
                    "uid": "group-a",
                    "example_id": "example-a",
                },
            )
        ]

    asyncio.run(run())
