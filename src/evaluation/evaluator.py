from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from src.data.loaders import Example
from src.llm.backend import LLMBackend


@dataclass
class EvaluationResult:
    accuracy: float
    records: list[dict]


def extract_answer(
    value: str,
    task: str = "",
    require_answer_line: bool = False,
) -> tuple[str, bool]:
    """Extract a deterministic final answer; return (answer, extraction_failed)."""
    value = value.strip()
    answer_matches = re.findall(r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$", value)
    # Model predictions must explicitly honor the protocol.  Gold values are
    # compact references and are allowed without an ``Answer:`` prefix.
    if not answer_matches and require_answer_line:
        return "", True
    candidate = answer_matches[-1].strip() if answer_matches else value
    if task in {"gsm8k", "object_counting", "multistep_arithmetic"}:
        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", candidate)
        if not numbers and candidate != value:
            numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", value)
        return (normalize_answer(numbers[-1]), False) if numbers else ("", True)
    choice_matches = re.findall(
        r"(?im)^\s*(?:final\s+)?answer\s*[:=]\s*[\(\[]?([A-H])[\)\]]?\s*[.!]?\s*$",
        value,
    )
    if not choice_matches:
        # BBH models often emit both the choice and its text, e.g.
        # ``Answer: (A) midsize triangular ...``.  The official gold is ``(A)``.
        choice_matches = re.findall(r"(?i)^\s*\(([A-H])\)(?:\s|$)", candidate)
    if not choice_matches:
        choice_matches = re.findall(r"(?i)^\s*\$\s*\(?([A-H])\)?\s*\$\s*$", candidate)
    if not choice_matches:
        choice_matches = re.findall(r"(?i)^\s*([A-H])[.)](?:\s|$)", candidate)
    if not choice_matches:
        choice_matches = re.findall(r"(?i)^\s*[\(\[]?([A-H])[\)\]]?\s*[.!]?\s*$", candidate)
    if choice_matches:
        return choice_matches[-1].lower(), False
    normalized = re.sub(r"\s+", " ", candidate.lower()).strip(" .\t\r\n")
    return normalized, not bool(normalized)


def normalize_answer(value: str) -> str:
    value = value.strip().replace(",", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
    if numbers:
        number = numbers[-1]
        try:
            parsed = float(number)
            return str(int(parsed)) if parsed.is_integer() else str(parsed)
        except ValueError:
            return number
    return re.sub(r"\s+", " ", value.lower()).strip(" .")


class Evaluator:
    def __init__(self, backend: LLMBackend, concurrency: int = 1):
        self.backend = backend
        self.concurrency = max(1, int(concurrency))

    def evaluate(
        self,
        system_prompt: str,
        examples: list[Example],
        concurrency: int | None = None,
    ) -> EvaluationResult:
        def evaluate_one(example: Example) -> dict:
            request = (
                f"{system_prompt.strip()}\n\n"
                f"{example.question}\n"
            )
            prediction = self.backend.generate(
                request,
                purpose="solve",
                metadata={"id": example.id, "task": example.task, "answer": example.answer},
            )
            extracted, extraction_failed = extract_answer(
                prediction, example.task, require_answer_line=True,
            )
            gold, gold_failed = extract_answer(example.answer, example.task)
            if gold_failed:
                gold = normalize_answer(example.answer)
            correct = not extraction_failed and extracted == gold
            return {
                "example_id": example.id,
                "task": example.task,
                "question": example.question,
                "prediction": prediction.strip(),
                "raw_model_output": prediction.strip(),
                "extracted_answer": extracted,
                "answer": example.answer,
                "gold_answer": gold,
                "correct": bool(correct),
                "extraction_failure": bool(extraction_failed),
            }
        workers = self.concurrency if concurrency is None else max(1, int(concurrency))
        if workers <= 1:
            records = [evaluate_one(example) for example in examples]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                records = list(executor.map(evaluate_one, examples))
        accuracy = sum(row["correct"] for row in records) / len(records) if records else 0.0
        return EvaluationResult(accuracy, records)
