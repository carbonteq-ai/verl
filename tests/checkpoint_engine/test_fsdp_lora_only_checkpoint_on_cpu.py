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

"""Unit tests for save_lora_only checkpoint support in FSDPCheckpointManager."""

from collections import namedtuple

import pytest

from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager

# Return type mimicking torch.nn.Module.load_state_dict with strict=False
_LoadStateDictResult = namedtuple("_LoadStateDictResult", ["missing_keys", "unexpected_keys"])


class TestBaseCheckpointManagerLoraOnly:
    """Tests for LoRA-only checkpoint properties on BaseCheckpointManager."""

    @pytest.fixture(autouse=True)
    def _patch_dist(self, monkeypatch):
        import torch.distributed

        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
        monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    def test_should_save_lora_only_default(self):
        mgr = self._make_manager(checkpoint_config=None)
        assert mgr.should_save_lora_only is False

    def test_should_save_lora_only_false(self):
        mgr = self._make_manager(checkpoint_config={"save_lora_only": False})
        assert mgr.should_save_lora_only is False

    def test_should_save_lora_only_true(self):
        mgr = self._make_manager(checkpoint_config={"save_lora_only": True})
        assert mgr.should_save_lora_only is True

    def test_is_lora_only_state_dict_empty(self):
        assert BaseCheckpointManager.is_lora_only_state_dict({}) is False

    def test_is_lora_only_state_dict_none(self):
        assert BaseCheckpointManager.is_lora_only_state_dict(None) is False

    def test_is_lora_only_state_dict_full(self):
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": None,
            "model.layers.0.self_attn.k_proj.weight": None,
            "lm_head.weight": None,
        }
        assert BaseCheckpointManager.is_lora_only_state_dict(state_dict) is False

    def test_is_lora_only_state_dict_lora(self):
        state_dict = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": None,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": None,
        }
        assert BaseCheckpointManager.is_lora_only_state_dict(state_dict) is True

    def test_is_lora_only_state_dict_adapter_key(self):
        state_dict = {
            "model.layers.0.attn.adapter_A.weight": None,
            "model.layers.0.attn.adapter_B.weight": None,
        }
        assert BaseCheckpointManager.is_lora_only_state_dict(state_dict) is True

    def test_is_lora_only_state_dict_mixed(self):
        state_dict = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": None,
            "model.layers.0.self_attn.q_proj.weight": None,
        }
        assert BaseCheckpointManager.is_lora_only_state_dict(state_dict) is False

    @staticmethod
    def _make_manager(checkpoint_config):
        class MockModel:
            pass

        class MockOptimizer:
            pass

        return BaseCheckpointManager(
            model=MockModel(),
            optimizer=MockOptimizer(),
            lr_scheduler=None,
            processing_class=None,
            checkpoint_config=checkpoint_config,
        )


class TestFSDPCheckpointManagerLoraOnly:
    """Tests for LoRA-only checkpoint logic on FSDPCheckpointManager."""

    @pytest.fixture(autouse=True)
    def _patch_dist(self, monkeypatch):
        import torch.distributed

        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
        monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    def _make_fsdp_manager(self, checkpoint_config, model=None, grad_scaler=None):
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

        if model is None:
            model = _FakeFSDPModel(has_lora=True)

        return FSDPCheckpointManager(
            model=model,
            grad_scaler=grad_scaler,
            optimizer=None,
            lr_scheduler=None,
            processing_class=None,
            checkpoint_config=checkpoint_config,
        )

    def test_has_lora_true(self):
        mgr = self._make_fsdp_manager(checkpoint_config={"save_lora_only": True})
        assert mgr._has_lora() is True

    def test_has_lora_false(self):
        model = _FakeFSDPModel(has_lora=False)
        mgr = self._make_fsdp_manager(checkpoint_config={"save_lora_only": True}, model=model)
        assert mgr._has_lora() is False

    def test_save_lora_only_filters_state_dict(self, tmp_path):
        model = _FakeFSDPModel(has_lora=True)
        mgr = self._make_fsdp_manager(
            checkpoint_config={"save_lora_only": True, "save_contents": ["model"]},
            model=model,
        )

        save_dir = tmp_path / "ckpt"
        mgr.save_checkpoint(local_path=str(save_dir), global_step=1)

        saved_path = save_dir / "model_world_size_1_rank_0.pt"
        import torch

        state_dict = torch.load(saved_path, weights_only=False)
        for key in state_dict:
            assert "lora_" in key or ".adapter_" in key, "Unexpected base key: {}".format(key)
        assert len(state_dict) == 4

    def test_save_lora_only_no_lora_saves_full(self, tmp_path):
        model = _FakeFSDPModel(has_lora=False)
        mgr = self._make_fsdp_manager(
            checkpoint_config={"save_lora_only": True, "save_contents": ["model"]},
            model=model,
        )

        save_dir = tmp_path / "ckpt"
        mgr.save_checkpoint(local_path=str(save_dir), global_step=1)

        saved_path = save_dir / "model_world_size_1_rank_0.pt"
        import torch

        state_dict = torch.load(saved_path, weights_only=False)
        assert "base.weight" in state_dict
        assert "lora_A.weight" in state_dict

    def test_save_lora_only_disabled_saves_full(self, tmp_path):
        model = _FakeFSDPModel(has_lora=True)
        mgr = self._make_fsdp_manager(
            checkpoint_config={"save_lora_only": False, "save_contents": ["model"]},
            model=model,
        )

        save_dir = tmp_path / "ckpt"
        mgr.save_checkpoint(local_path=str(save_dir), global_step=1)

        saved_path = save_dir / "model_world_size_1_rank_0.pt"
        import torch

        state_dict = torch.load(saved_path, weights_only=False)
        assert "base.weight" in state_dict
        assert "lora_A.weight" in state_dict

    def test_load_lora_only_checkpoint_merges(self, tmp_path):
        model_before = _FakeFSDPModel(has_lora=True)
        model_before._fsdp_wrapped_module.base_weight.data.fill_(1.0)

        mgr = self._make_fsdp_manager(
            checkpoint_config={"save_lora_only": True, "save_contents": ["model"]},
            model=model_before,
        )
        save_dir = tmp_path / "lora_only"
        mgr.save_checkpoint(local_path=str(save_dir), global_step=1)

        model_after = _FakeFSDPModel(has_lora=True)
        model_after._fsdp_wrapped_module.base_weight.data.fill_(99.0)

        mgr2 = self._make_fsdp_manager(
            checkpoint_config={"load_contents": ["model"]},
            model=model_after,
        )
        mgr2.load_checkpoint(local_path=str(save_dir))

        assert model_after._fsdp_wrapped_module.base_weight.item() == 99.0
        assert model_after._fsdp_wrapped_module.lora_A_weight.item() == 0.5

    def test_load_full_checkpoint_still_works(self, tmp_path):
        model = _FakeFSDPModel(has_lora=True)
        model._fsdp_wrapped_module.base_weight.data.fill_(42.0)
        mgr = self._make_fsdp_manager(
            checkpoint_config={"save_lora_only": False, "save_contents": ["model"]},
            model=model,
        )
        save_dir = tmp_path / "full_ckpt"
        mgr.save_checkpoint(local_path=str(save_dir), global_step=1)

        model_after = _FakeFSDPModel(has_lora=True)
        model_after._fsdp_wrapped_module.base_weight.data.fill_(0.0)
        mgr2 = self._make_fsdp_manager(
            checkpoint_config={"load_contents": ["model"]},
            model=model_after,
        )
        mgr2.load_checkpoint(local_path=str(save_dir))

        assert model_after._fsdp_wrapped_module.base_weight.item() == 42.0

    def test_extra_state_restores_loss_scale_and_growth_counter(self, tmp_path):
        import torch

        scaler = torch.amp.GradScaler("cpu", init_scale=512, growth_interval=3)
        expected = scaler.state_dict()
        expected["_growth_tracker"] = 2
        scaler.load_state_dict(expected)
        manager = self._make_fsdp_manager({"save_contents": ["extra"], "load_contents": ["extra"]}, grad_scaler=scaler)
        manager.save_checkpoint(str(tmp_path), global_step=4)
        extra = torch.load(tmp_path / "extra_state_world_size_1_rank_0.pt", weights_only=False)
        assert extra["grad_scaler"] == expected
        replacement = torch.amp.GradScaler("cpu", init_scale=65536)
        loader = self._make_fsdp_manager({"load_contents": ["extra"]}, grad_scaler=replacement)
        loader.load_checkpoint(str(tmp_path))
        assert replacement.state_dict() == expected

    def test_fp16_missing_scaler_fails_before_model_load(self, tmp_path):
        import torch

        torch.save({"lr_scheduler": None}, tmp_path / "extra_state_world_size_1_rank_0.pt")
        model = _FakeFSDPModel(has_lora=True)
        model._fsdp_wrapped_module.base_weight.data.fill_(99.0)
        loader = self._make_fsdp_manager(
            {"load_contents": ["model", "extra"]}, model=model, grad_scaler=torch.amp.GradScaler("cpu")
        )
        with pytest.raises(ValueError, match="missing GradScaler state"):
            loader.load_checkpoint(str(tmp_path))
        assert model._fsdp_wrapped_module.base_weight.item() == 99.0

    def test_bf16_extra_state_remains_compatible(self, tmp_path):
        import torch

        manager = self._make_fsdp_manager({"save_contents": ["extra"], "load_contents": ["extra"]})
        manager.save_checkpoint(str(tmp_path))
        extra = torch.load(tmp_path / "extra_state_world_size_1_rank_0.pt", weights_only=False)
        assert "grad_scaler" not in extra
        manager.load_checkpoint(str(tmp_path))

    def test_disabled_scaler_accepts_legacy_extra_state(self, tmp_path):
        import torch

        saver = self._make_fsdp_manager({"save_contents": ["extra"]})
        saver.save_checkpoint(str(tmp_path))
        loader = self._make_fsdp_manager(
            {"load_contents": ["extra"]}, grad_scaler=torch.amp.GradScaler("cpu", enabled=False)
        )
        loader.load_checkpoint(str(tmp_path))

    def test_model_only_fp16_load_does_not_require_scaler_state(self, tmp_path):
        import torch

        saver = self._make_fsdp_manager({"save_contents": ["model"], "save_lora_only": True})
        saver.save_checkpoint(str(tmp_path))
        model = _FakeFSDPModel(has_lora=True)
        model._fsdp_wrapped_module.lora_A_weight.data.fill_(99.0)
        scaler = torch.amp.GradScaler("cpu", init_scale=1024)
        expected = scaler.state_dict()
        loader = self._make_fsdp_manager({"load_contents": ["model"]}, model=model, grad_scaler=scaler)
        loader.load_checkpoint(str(tmp_path))
        assert model._fsdp_wrapped_module.lora_A_weight.item() == 0.5
        assert scaler.state_dict() == expected


class _FakeConfig:
    """Minimal config mock that supports save_pretrained."""

    def save_pretrained(self, path):
        pass


class _FakeFSDPModel:
    """A fake FSDP-wrapped model that mimics the interface used by FSDPCheckpointManager."""

    def __init__(self, has_lora=True):
        self._fsdp_wrapped_module = _FakePEFTModel(has_lora=has_lora)
        self.config = self._fsdp_wrapped_module.config
        self.can_generate = self._fsdp_wrapped_module.can_generate

    def state_dict(self):
        return self._fsdp_wrapped_module.state_dict()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = self._fsdp_wrapped_module.load_state_dict(state_dict, strict=strict, assign=assign)
        if result is None:
            return _LoadStateDictResult(missing_keys=[], unexpected_keys=[])
        return result

    def named_buffers(self):
        return {}.items()


class _FakePEFTModel:
    """Fake PEFT-style model with a peft_config."""

    def __init__(self, has_lora=True):
        import torch

        self.base_weight = torch.nn.Parameter(torch.tensor(1.0))
        self.lora_A_weight = torch.nn.Parameter(torch.tensor(0.5))
        self.lora_B_weight = torch.nn.Parameter(torch.tensor(0.3))
        self.config = _FakeConfig()
        self.can_generate = lambda: False

        if has_lora:
            self.peft_config = {"default": type("Cfg", (), {"r": 8, "lora_alpha": 16, "task_type": "CAUSAL_LM"})()}

    def state_dict(self):
        d = {
            "base.weight": self.base_weight,
            "lora_A.weight": self.lora_A_weight,
            "lora_B.weight": self.lora_B_weight,
        }
        if hasattr(self, "peft_config"):
            d["base_model.model.lora_A.lora_A.weight"] = self.lora_A_weight
            d["base_model.model.lora_B.lora_B.weight"] = self.lora_B_weight
        return d

    def load_state_dict(self, state_dict, strict=True, assign=False):
        unexpected_keys = []
        for key, tensor in state_dict.items():
            if "base.weight" in key or key == "base.weight":
                self.base_weight.data.copy_(tensor)
            elif "lora_A" in key:
                self.lora_A_weight.data.copy_(tensor)
            elif "lora_B" in key:
                self.lora_B_weight.data.copy_(tensor)
            else:
                unexpected_keys.append(key)
        return _LoadStateDictResult(missing_keys=[], unexpected_keys=unexpected_keys)
