from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .backend import LLMBackend


RULES = {
    "object_counting": {
        "failure_mode": "missed or double-counted object groups",
        "proposed_rule": "For object counting, list every quantity exactly once and sum all groups before answering.",
        "applicable_condition": "Questions that ask for the total number of objects across several groups.",
    },
    "multistep_arithmetic": {
        "failure_mode": "incorrect operation order",
        "proposed_rule": "For multistep arithmetic, evaluate parentheses first, then multiplication, then addition or subtraction, and verify the integer.",
        "applicable_condition": "Symbolic arithmetic expressions with two or more operations.",
    },
    "gsm8k": {
        "failure_mode": "translated the story into the wrong operations",
        "proposed_rule": "For word problems, name the known quantities, write the arithmetic expression, solve it, and return only the final number.",
        "applicable_condition": "Short grade-school math word problems.",
    },
}


class MockBackend(LLMBackend):
    def __init__(self, model: str, cache_dir: str | Path, max_llm_calls: int | None = None):
        super().__init__(model, cache_dir, max_llm_calls)

    def _call(self, prompt: str, purpose: str, metadata: dict[str, Any]) -> dict[str, Any]:
        if purpose == "solve":
            answer = str(metadata.get("answer", "0"))
            task = metadata.get("task", "")
            relevant = RULES.get(task, {}).get("proposed_rule", "").lower()
            normalized = prompt.lower()
            rule_present = relevant and relevant[:32] in normalized
            task_rule_count = sum(rule["proposed_rule"].lower()[:32] in normalized for rule in RULES.values())
            digest = int(hashlib.sha256((metadata.get("id", "") + normalized).encode()).hexdigest()[:8], 16)
            succeeds = bool(rule_present and task_rule_count <= 2) or digest % 3 != 0
            final = answer if succeeds else _wrong_answer(answer)
            text = f"I worked through the quantities carefully.\nAnswer: {final}"
        elif purpose in {"gradient", "textgrad_feedback"}:
            task = metadata.get("task", "gsm8k")
            if purpose == "textgrad_feedback":
                rule = RULES.get(task, RULES["gsm8k"])
                text = f"The prompt can fail because it has {rule['failure_mode']}. {rule['proposed_rule']}"
            else:
                card = dict(RULES.get(task, RULES["gsm8k"]))
                card["evidence_examples_ids"] = metadata.get("failure_ids", [])
                text = json.dumps(card)
        elif purpose == "textgrad_update":
            task = metadata.get("task", "gsm8k")
            current = str(metadata.get("current_prompt", "")).strip()
            rule = RULES.get(task, RULES["gsm8k"])["proposed_rule"]
            updated = current if rule in current else current + "\n" + rule
            text = f"<IMPROVED_VARIABLE>{updated.strip()}</IMPROVED_VARIABLE>"
        elif purpose == "scoped_rule_extraction":
            candidate = str(metadata.get("candidate_rule", "")).strip()
            candidate = re.sub(
                r"(?i)^apply this textual-gradient feedback when relevant:\s*", "", candidate,
            )
            text = candidate.split(".")[0].strip() + "."
        elif purpose in {"aggregate_summary", "aggregate_structured", "aggregate_uid", "aggregate_summary_budgeted", "aggregate_uid_budgeted"}:
            prompts = metadata.get("prompts") or metadata.get("rules", [])
            lines = _dedupe(
                line.strip("- ") for prompt_value in prompts
                for line in str(prompt_value).splitlines()
                if line.strip() and "last line of your response" not in line.lower()
            )
            title = "[UID-Balanced Shared Prompt]" if purpose == "aggregate_uid" else "[Shared Prompt]"
            text = title + "\n" + "\n".join(lines)
            text += "\nThe last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
        else:
            text = "OK"
        return {"text": text, "usage": {"input_tokens": max(1, len(prompt) // 4), "output_tokens": max(1, len(text) // 4)}}


def _wrong_answer(answer: str) -> str:
    match = re.search(r"-?\d+(?:\.\d+)?", answer)
    if not match:
        return "incorrect"
    number = float(match.group()) + 1
    return str(int(number)) if number.is_integer() else str(number)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))
