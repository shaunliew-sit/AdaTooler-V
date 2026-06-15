"""Compatibility shim — transformers>=5.0 vs vllm<=0.11.0 tokenizer caching.

transformers 5.x removed ``PreTrainedTokenizerBase.all_special_tokens_extended``,
but ``vllm<=0.11.0``'s ``get_cached_tokenizer`` (vllm/transformers_utils/tokenizer.py)
still reads it, raising at rollout startup:

    AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended

This restores the attribute (returning ``all_special_tokens``) so the vLLM rollout
tokenizer cache works with Qwen3-VL on transformers 5.x. The repo pins
``vllm<=0.11.0`` (verltool/pyproject.toml) and needs transformers 5.x for Qwen3-VL,
so this shim — not a version change — is the supported bridge.

INSTALL (so it runs for every interpreter in the env, incl. Ray actors):

    cp verltool/patches/sitecustomize_transformers5_vllm.py \
       $(python -c "import site;print(site.getsitepackages()[0])")/sitecustomize.py

Python auto-imports ``sitecustomize`` at startup, so ``apply()`` runs everywhere.
Reversible: delete that sitecustomize.py to remove the shim.
"""


def apply() -> None:
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
            PreTrainedTokenizerBase.all_special_tokens_extended = property(
                lambda self: self.all_special_tokens
            )
    except Exception:
        # Never let the shim break interpreter startup.
        pass


apply()
