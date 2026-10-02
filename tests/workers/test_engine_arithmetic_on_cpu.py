"""Native arithmetic scope admission, backward coverage and state restoration."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch

from verl.utils.engine_arithmetic import engine_arithmetic
from verl.workers.config.engine import FSDPEngineConfig


def state():
    result = {
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "math_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
    }
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        result["fp16_accumulation"] = torch.backends.cuda.matmul.allow_fp16_accumulation
    return result


@pytest.mark.parametrize("full,math", [(False, False), (True, False), (False, True), (True, True)])
def test_independent_controls_cover_backward_and_restore(full, math):
    before = state()
    observed = []
    with engine_arithmetic(full_precision_matmul=full, math_sdpa=math):
        inside = state()
        if full:
            assert not any(inside[name] for name in ("tf32", "cudnn_tf32", "fp16_reduction", "bf16_reduction"))
            assert not inside.get("fp16_accumulation", False)
        else:
            assert all(inside[name] == before[name] for name in ("tf32", "fp16_reduction", "bf16_reduction"))
        if math:
            assert inside["math"]
            assert not any(inside[name] for name in ("flash", "efficient", "cudnn", "math_reduction"))
        else:
            assert all(inside[name] == before[name] for name in ("flash", "efficient", "cudnn", "math"))
        value = torch.tensor(2.0, requires_grad=True)
        value.register_hook(lambda gradient: observed.append(state()))
        (value * value).backward()
        assert observed == [inside]
    assert state() == before


def test_nested_scopes_restore_outer_profile_on_exception():
    before = state()
    with engine_arithmetic(full_precision_matmul=True):
        outer = state()
        with pytest.raises(ValueError, match="training failure"):
            with engine_arithmetic(math_sdpa=True):
                raise ValueError("training failure")
        assert state() == outer
    assert state() == before


def test_missing_math_control_rejects_before_matmul_mutation(monkeypatch):
    before = state()
    monkeypatch.setattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp", None)
    with pytest.raises(RuntimeError, match="requires PyTorch"):
        with engine_arithmetic(full_precision_matmul=True, math_sdpa=True):
            pytest.fail("unsupported runtime entered")
    assert state() == before


def test_cooperating_default_scope_waits_for_opted_in_scope():
    attempted, entered = Event(), Event()
    before = state()

    def default_scope():
        attempted.set()
        with engine_arithmetic():
            entered.set()
            return state()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with engine_arithmetic(full_precision_matmul=True, math_sdpa=True):
            future = pool.submit(default_scope)
            assert attempted.wait(2)
            assert not entered.wait(0.05)
        assert future.result(timeout=2) == before
    assert state() == before


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("failure", ["body", "enter", "exit"])
def test_native_mode_context_restores_settings_for_failures(monkeypatch, mode, failure):
    from verl.workers.engine.base import BaseEngineCtx
    from verl.workers.engine.fsdp import transformer_impl as native

    engine = object.__new__(native.FSDPEngine)
    engine.engine_config = FSDPEngineConfig(full_precision_matmul=True, math_sdpa=True, fsdp_size=1)
    engine.module = torch.nn.Linear(1, 1)
    engine.ulysses_parallel_group = None
    engine.optimizer_zero_grad = lambda: None
    monkeypatch.setattr(native, "get_ulysses_sequence_parallel_group", lambda: None)
    monkeypatch.setattr(native, "set_ulysses_sequence_parallel_group", lambda value: None)

    def enter(self):
        if failure == "enter":
            raise RuntimeError("injected failure")

    def exit(self, *args):
        if failure == "exit":
            raise RuntimeError("injected failure")

    monkeypatch.setattr(BaseEngineCtx, "__enter__", enter)
    monkeypatch.setattr(BaseEngineCtx, "__exit__", exit)
    context_type = native.EngineTrainModeCtx if mode == "train" else native.EngineEvalModeCtx
    before = state()
    with pytest.raises(RuntimeError, match="injected failure"):
        with context_type(engine):
            assert state()["math"] and not state()["flash"]
            if failure == "body":
                raise RuntimeError("injected failure")
    assert state() == before


def test_engine_config_defaults_preserve_runtime_selection():
    config = FSDPEngineConfig()
    assert not config.full_precision_matmul
    assert not config.math_sdpa
