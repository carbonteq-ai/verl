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
"""fp16 loss-scaler evidence from the FSDP engine's optimizer step and train_batch."""

from types import SimpleNamespace

import torch
from tensordict import TensorDict

from verl.workers.engine.base import BaseEngine
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def _engine(scaler):
    engine = object.__new__(FSDPEngine)
    engine.module = torch.nn.Linear(2, 1)
    engine.optimizer = torch.optim.SGD(engine.module.parameters(), lr=0.1)
    engine.optimizer_config = SimpleNamespace(clip_grad=1.0)
    engine.scaler = scaler
    engine._qat_enabled = False
    engine.last_loss_scale_metrics = {}
    return engine


def _backward(engine, loss):
    engine.optimizer.zero_grad()
    engine.scaler.scale(loss).backward() if engine.scaler is not None else loss.backward()


def test_scaled_step_reports_scale_and_no_skip():
    engine = _engine(torch.amp.GradScaler("cpu", init_scale=1024.0, growth_interval=400))
    before = engine.module.weight.detach().clone()
    _backward(engine, engine.module(torch.ones(1, 2)).sum())
    grad_norm = engine.optimizer_step()
    assert grad_norm == grad_norm  # finite
    assert engine.last_loss_scale_metrics == {"loss_scale": 1024.0, "optimizer_step_skipped": 0.0}
    assert not torch.equal(before, engine.module.weight)


def test_overflowing_step_is_skipped_and_reported():
    engine = _engine(torch.amp.GradScaler("cpu", init_scale=1024.0, growth_interval=400))
    before = engine.module.weight.detach().clone()
    _backward(engine, engine.module(torch.ones(1, 2)).sum() * float("inf"))
    grad_norm = engine.optimizer_step()
    assert not torch.isfinite(torch.tensor(grad_norm))
    assert engine.last_loss_scale_metrics == {"loss_scale": 1024.0, "optimizer_step_skipped": 1.0}
    assert engine.scaler.get_scale() == 512.0
    assert torch.equal(before, engine.module.weight)


def test_bf16_step_reports_no_loss_scale():
    engine = _engine(None)
    _backward(engine, engine.module(torch.ones(1, 2)).sum())
    engine.optimizer_step()
    assert engine.last_loss_scale_metrics == {}


def test_train_batch_adds_loss_scale_metrics_on_the_output_rank():
    calls = []
    engine = SimpleNamespace(
        optimizer_zero_grad=lambda: calls.append("zero"),
        forward_backward_batch=lambda data, loss_function, forward_only: {"metrics": {"loss": [0.5]}},
        optimizer_step=lambda: 2.0,
        is_mp_src_rank_with_outputs=lambda: True,
        last_loss_scale_metrics={"loss_scale": 65536.0, "optimizer_step_skipped": 0.0},
    )
    outputs = BaseEngine.train_batch(engine, TensorDict({}, batch_size=[]), loss_function=None)
    assert outputs["metrics"] == {
        "loss": [0.5],
        "grad_norm": 2.0,
        "loss_scale": 65536.0,
        "optimizer_step_skipped": 0.0,
    }


def test_worker_keeps_loss_scale_metrics_per_step(monkeypatch):
    from verl.workers import engine_workers

    class _Device:
        @staticmethod
        def max_memory_allocated():
            return 0

        @staticmethod
        def max_memory_reserved():
            return 0

    monkeypatch.setattr(engine_workers, "get_torch_device", lambda: _Device)
    worker = object.__new__(engine_workers.TrainingWorker)
    worker.device_name = "cpu"
    worker.engine = SimpleNamespace(get_data_parallel_group=lambda: None)
    worker.flops_counter = None
    output = {
        "loss": [0.25, 0.25],
        "metrics": {"pg_loss": [0.1, 0.2], "grad_norm": 3.0, "loss_scale": 32768.0, "optimizer_step_skipped": 1.0},
    }
    final = worker._postprocess_output(
        output, global_token_num=None, delta_time=1.0, forward_only=False, images_seqlens=None
    )
    metrics = final["metrics"]
    assert metrics["loss_scale"] == 32768.0
    assert metrics["optimizer_step_skipped"] == 1.0
    assert metrics["grad_norm"] == 3.0
    assert metrics["pg_loss"] == [0.1, 0.2]
