"""Env fixups for Qwen3-VL GRPO training on this host (transformers 5.x / vllm 0.11
/ CUDA 13.1). Installed as ``sitecustomize.py`` so it auto-runs for EVERY
interpreter in the env — including the Ray vLLM rollout actors where these bite.

Two fixups, both required to get the vLLM rollout to start:

1. transformers>=5.0 removed ``PreTrainedTokenizerBase.all_special_tokens_extended``,
   but ``vllm<=0.11.0``'s ``get_cached_tokenizer`` still reads it ->
   ``AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended``.
   We restore it (returns ``all_special_tokens``). The repo pins ``vllm<=0.11.0``
   and needs transformers 5.x for Qwen3-VL, so this shim — not a version change —
   is the bridge.

2. The host has a CUDA **13.1** ptxas on PATH, which Triton 3.4.0 (bundled with
   torch cu128 / vllm 0.11) cannot target -> ``RuntimeError: Triton only support
   CUDA 10.0 or higher, but got CUDA version: 13.1`` while compiling the Qwen3-VL
   vision rotary kernel in vLLM's profile_run. We point ``TRITON_PTXAS_PATH`` at
   Triton's own bundled CUDA-12.8 ptxas (sibling dir of this file), which Triton
   supports. Verified: triton kernels then compile with max_err 0.0.

INSTALL (run for every interpreter, incl. Ray actors):

    cp verltool/patches/sitecustomize_transformers5_vllm.py \
       $(python -c "import site;print(site.getsitepackages()[0])")/sitecustomize.py

Reversible: delete that sitecustomize.py. Idempotent and never raises at startup.
"""
import os


def _fix_tokenizer() -> None:
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
            PreTrainedTokenizerBase.all_special_tokens_extended = property(
                lambda self: self.all_special_tokens
            )
    except Exception:
        pass


def _fix_triton_ptxas() -> None:
    # Point Triton at its bundled (CUDA 12.8) ptxas instead of the host's 13.1 one.
    # NOTE: the env explicitly exports TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
    # (the broken 13.1 one), so we OVERRIDE it whenever the bundled ptxas exists —
    # the bundled binary is the one guaranteed compatible with the bundled Triton.
    # This file lives in site-packages, and triton/ is a sibling package dir.
    try:
        bundled = os.path.join(
            os.path.dirname(__file__),
            "triton", "backends", "nvidia", "bin", "ptxas",
        )
        if os.path.isfile(bundled) and os.access(bundled, os.X_OK):
            if os.environ.get("TRITON_PTXAS_PATH") != bundled:
                os.environ["TRITON_PTXAS_PATH"] = bundled
    except Exception:
        pass


def apply() -> None:
    _fix_tokenizer()
    _fix_triton_ptxas()


apply()
