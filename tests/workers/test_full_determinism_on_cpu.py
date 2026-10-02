"""Full determinism must enforce the native PyTorch kernel contract."""

import random

import numpy as np
import torch

from verl.workers.engine import utils


def test_full_determinism_enforces_strict_torch_kernels(monkeypatch):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    monkeypatch.setattr(utils.os, "environ", dict(utils.os.environ))
    monkeypatch.setattr(utils, "device_manual_seed", lambda seed: None)
    monkeypatch.setattr(utils, "device_manual_seed_all", lambda seed: None)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        utils.enable_full_determinism(71)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert utils.os.environ["FLASH_ATTENTION_DETERMINISTIC"] == "1"
        assert utils.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled = cudnn
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
