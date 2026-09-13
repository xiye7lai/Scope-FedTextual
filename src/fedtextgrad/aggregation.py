from __future__ import annotations

from src.llm.backend import LLMBackend
from src.textgrad.gradient_card import GradientCard
from src.textgrad.official_compat import FORMATTING_INSTRUCTION, summarization_prompt


def aggregate_prompts(
    prompts: list[str],
    method: str,
    backend: LLMBackend,
    formatting_instruction: str = FORMATTING_INSTRUCTION,
) -> str:
    """Aggregate full locally optimized prompts, as in official FedTextGrad."""
    normalized = {
        "summary": "summarization",
        "uid_summary": "sum_uid",
    }.get(method, method)
    if normalized == "concat":
        return "\n\n".join(prompts)
    if normalized not in {"summarization", "structured_summary", "sum_uid"}:
        raise ValueError(f"Unknown aggregation method: {method}")
    uid = normalized == "sum_uid"
    return backend.generate(
        summarization_prompt(prompts, uid=uid, formatting_instruction=formatting_instruction),
        purpose="aggregate_uid" if uid else ("aggregate_structured" if normalized == "structured_summary" else "aggregate_summary"),
        metadata={"prompts": prompts, "rules": prompts},
    )


def aggregate_cards(
    cards: list[GradientCard],
    method: str,
    backend: LLMBackend,
    formatting_instruction: str = FORMATTING_INSTRUCTION,
) -> str:
    card_prompts = [
        f"Rule: {card.proposed_rule}\nApplicable condition: {card.applicable_condition}"
        for card in cards
    ]
    return aggregate_prompts(card_prompts, method, backend, formatting_instruction)


def method_label(method: str) -> str:
    return {
        "concat": "FedTextGrad-Concat",
        "summary": "FedTextGrad-Summary",
        "summarization": "FedTextGrad-Summary",
        "structured_summary": "FedTextGrad-StructuredSummary",
        "uid_summary": "FedTextGrad-UID",
        "sum_uid": "FedTextGrad-UID",
    }[method]
