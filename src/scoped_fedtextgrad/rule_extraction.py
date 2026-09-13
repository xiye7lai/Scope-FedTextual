from __future__ import annotations

import re
from typing import Callable

from src.llm.backend import LLMBackend
from src.textgrad.gradient_card import GradientCard


def extract_compact_rules(
    cards: list[GradientCard],
    backend: LLMBackend,
    token_counter: Callable[[str], int],
    max_rule_tokens: int = 64,
) -> list[GradientCard]:
    """Turn each local textual update into one portable imperative rule."""
    for card in cards:
        prompt = (
            "Convert the local textual-gradient feedback below into exactly one "
            "portable reasoning rule. Return only one imperative sentence, with no "
            "analysis, labels, bullets, examples, task names, or answer. The rule "
            f"must be at most {max_rule_tokens} tokens and preserve the concrete fix.\n\n"
            f"Failure mode:\n{card.failure_mode}\n\n"
            f"Local update candidate:\n{card.proposed_rule}"
        )
        raw = backend.generate(
            prompt,
            purpose="scoped_rule_extraction",
            metadata={
                "source_client": card.source_client, "source_task": card.task,
                "failure_mode": card.failure_mode, "candidate_rule": card.proposed_rule,
                "max_rule_tokens": max_rule_tokens,
            },
        )
        rule = _clean_rule(raw)
        if not rule:
            rule = _clean_rule(card.proposed_rule)
        rule = _hard_cap(rule, max_rule_tokens, token_counter)
        card.proposed_rule = rule
        card.token_length = token_counter(rule)
    return cards


def _clean_rule(value: str) -> str:
    value = value.strip()
    value = re.sub(r"(?is)</?think>", "", value)
    value = re.sub(r"(?i)^\s*(?:rule|answer)\s*:\s*", "", value)
    lines = [line.strip(" -*\t") for line in value.splitlines() if line.strip()]
    return " ".join(lines).strip()


def _hard_cap(value: str, budget: int, counter: Callable[[str], int]) -> str:
    if counter(value) <= budget:
        return value
    low, high = 1, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if counter(value[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    capped = value[:low].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return capped + "." if capped and capped[-1] not in ".!?" else capped
