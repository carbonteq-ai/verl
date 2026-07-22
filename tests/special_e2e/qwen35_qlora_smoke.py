"""One-GPU Qwen 3.5 2B NF4 QLoRA loader and backward smoke test.

Run explicitly; this module is intentionally not named ``test_*.py`` so the
4.3 GB model gate is not collected by the normal CPU test suite.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from verl.workers.config.model import BitsAndBytesQuantizationConfig, HFModelConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    args = parser.parse_args()

    model_config = HFModelConfig(
        path=args.model_path,
        override_config={"attn_implementation": "eager", "use_cache": False},
        use_remove_padding=False,
        lora_rank=8,
        lora_alpha=16,
        target_modules="all-linear",
        bitsandbytes=BitsAndBytesQuantizationConfig(enable=True),
    )
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.model_config = model_config
    engine.engine_config = type("EngineConfig", (), {"model_dtype": "bf16"})()
    engine.device_mesh = None
    engine.use_remove_padding = False
    engine.ulysses_sequence_parallel_size = 1

    module = engine._build_lora_module(engine._build_module())
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="cuda")
    logits = module(input_ids=input_ids, use_cache=False).logits
    loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
    )
    loss.backward()

    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    nonzero_grad_tensors = sum(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0 for parameter in trainable
    )
    if nonzero_grad_tensors == 0:
        raise RuntimeError("QLoRA backward produced no nonzero adapter gradients")
    print(
        {
            "loss": float(loss.detach()),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "nonzero_grad_tensors": nonzero_grad_tensors,
            "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        }
    )


if __name__ == "__main__":
    main()
