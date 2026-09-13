from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable


TokenCounter = Callable[[str], int]


def estimated_token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def build_token_counter(cfg: dict) -> tuple[TokenCounter, str]:
    """Return a cached model tokenizer counter, or an explicit fallback.

    Formal Scoped-FedTextGrad experiments set ``require_exact_tokenizer`` so a missing
    tokenizer is a hard error instead of silently weakening the budget claim.
    """
    name = cfg.get("tokenizer_name") or cfg.get("model_name") or cfg.get("model")
    require_exact = bool(cfg.get("require_exact_tokenizer", False))
    if name:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                name,
                trust_remote_code=bool(cfg.get("tokenizer_trust_remote_code", False)),
                local_files_only=bool(cfg.get("tokenizer_local_files_only", True)),
                cache_dir=cfg.get("tokenizer_cache_dir"),
            )

            @lru_cache(maxsize=100_000)
            def count(text: str) -> int:
                return len(tokenizer.encode(text, add_special_tokens=False))

            return count, f"transformers:{name}"
        except Exception as exc:
            if require_exact:
                raise RuntimeError(f"Exact tokenizer {name!r} is required: {exc}") from exc
    return estimated_token_count, "estimated_chars_div_4"
