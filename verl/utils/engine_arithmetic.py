"""Scoped opt-in arithmetic controls for serialized native engine operations.

PyTorch backend switches are process-global. Cooperating engine scopes share a
reentrant lock; unrelated external threads must not execute PyTorch work while
an opted-in scope is active. Defaults preserve the caller's runtime settings.
"""

from contextlib import ExitStack, contextmanager
from threading import RLock

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_ENGINE_ARITHMETIC_LOCK = RLock()


@contextmanager
def engine_arithmetic(*, full_precision_matmul=False, math_sdpa=False):
    """Cover forward, backward and recomputation, then restore runtime state.

Full precision disables TF32 and reduced FP16/BF16 GEMM accumulation; it does
not promote model activations. Math SDPA selects the math attention kernel and
disables its optional reduced-precision intermediate reduction. Neither option
promises invariant results across arbitrary tensor shapes or architectures.
"""
    with _ENGINE_ARITHMETIC_LOCK, ExitStack() as stack:
        settings = []
        if full_precision_matmul:
            settings.extend(
                (owner, name, False)
                for owner, name in (
                    (torch.backends.cuda.matmul, "allow_tf32"),
                    (torch.backends.cudnn, "allow_tf32"),
                    (torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"),
                    (torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"),
                )
            )
            # Older PyTorch has no opt-in FP16 accumulation switch.
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                settings.append((torch.backends.cuda.matmul, "allow_fp16_accumulation", False))
        # Read every requested setting before making any mutation.
        previous = [(owner, name, getattr(owner, name)) for owner, name, _ in settings]
        if math_sdpa:
            getter = getattr(torch.backends.cuda, "fp16_bf16_reduction_math_sdp_allowed", None)
            setter = getattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp", None)
            if getter is None or setter is None:
                raise RuntimeError("math_sdpa requires PyTorch math SDP reduction controls")
            old_reduction = getter()
        for owner, name, value in previous:
            stack.callback(setattr, owner, name, value)
        for owner, name, value in settings:
            setattr(owner, name, value)
        if math_sdpa:
            stack.callback(setter, old_reduction)
            setter(False)
            stack.enter_context(sdpa_kernel(SDPBackend.MATH))
        yield
