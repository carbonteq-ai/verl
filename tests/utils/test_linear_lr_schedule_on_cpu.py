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
"""The FSDP engine's linear schedule is Hugging Face's ``lr_scheduler_type="linear"``."""

import pytest
import torch

from verl.utils.torch_functional import get_linear_schedule_with_warmup
from verl.workers.config.optimizer import FSDPOptimizerConfig


def _lrs(factory, steps):
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=5e-5)
    scheduler = factory(optimizer)
    values = []
    for _ in range(steps):
        values.append(optimizer.param_groups[0]["lr"])  # the LR the optimizer step uses
        optimizer.step()
        scheduler.step()
    return values


@pytest.mark.parametrize(("warmup", "total"), [(0, 10), (3, 10), (1, 1), (5, 20)])
def test_linear_schedule_matches_transformers(warmup, total):
    transformers = pytest.importorskip("transformers")
    ours = _lrs(lambda opt: get_linear_schedule_with_warmup(opt, warmup, total), total + 2)
    reference = _lrs(lambda opt: transformers.get_linear_schedule_with_warmup(opt, warmup, total), total + 2)
    assert ours == pytest.approx(reference, rel=0, abs=0)


def test_fsdp_optimizer_config_accepts_linear():
    assert FSDPOptimizerConfig(lr=1e-5, lr_scheduler_type="linear").lr_scheduler_type == "linear"
