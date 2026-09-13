from __future__ import annotations

from dataclasses import dataclass, field

from src.data.client_split import ClientData
from src.evaluation.evaluator import Evaluator
from src.llm.backend import LLMBackend

from .gradient_card import GradientCard
from .official_compat import build_backward_prompt, build_tgd_prompt, extract_improved_variable


@dataclass
class LocalOptimizationResult:
    client_id: str
    initial_prompt: str
    optimized_prompt: str
    cards: list[GradientCard]
    score_before: float
    score_after: float
    step_history: list[dict] = field(default_factory=list)


class LocalTextGradOptimizer:
    """Protocol-compatible version of the official TextGrad local update loop."""

    def __init__(
        self,
        backend: LLMBackend,
        rounds: int = 1,
        max_failures: int = 3,
        batch_size: int = 3,
        max_steps: int | None = None,
        proximal_update: bool = True,
        evaluation_concurrency: int = 1,
    ):
        self.backend = backend
        self.evaluator = Evaluator(backend, evaluation_concurrency)
        self.local_epochs = max(1, int(rounds))
        self.max_failures = max_failures
        self.batch_size = max(1, int(batch_size))
        self.max_steps = max_steps if max_steps is not None else self.local_epochs
        self.proximal_update = proximal_update

    def optimize(self, client: ClientData, initial_prompt: str) -> LocalOptimizationResult:
        prompt = initial_prompt
        cards: list[GradientCard] = []
        history: list[dict] = []
        initial_eval = self.evaluator.evaluate(prompt, client.train)
        initial_score = initial_eval.accuracy
        step_index = 0

        for epoch in range(self.local_epochs):
            for batch_start in range(0, len(client.train), self.batch_size):
                if step_index >= self.max_steps:
                    break
                batch = client.train[batch_start:batch_start + self.batch_size]
                prompt_before = prompt
                before_eval = self.evaluator.evaluate(prompt, batch)
                evidence = before_eval.records[:self.max_failures]
                feedback = self.backend.generate(
                    build_backward_prompt(prompt, evidence),
                    purpose="textgrad_feedback",
                    metadata={
                        "task": client.task,
                        "failure_ids": [row["example_id"] for row in evidence if not row["correct"]],
                        "score": before_eval.accuracy,
                    },
                )
                raw_update = self.backend.generate(
                    build_tgd_prompt(prompt, feedback),
                    purpose="textgrad_update",
                    metadata={"task": client.task, "current_prompt": prompt, "feedback": feedback},
                )
                candidate_prompt = extract_improved_variable(raw_update, prompt)
                after_eval = self.evaluator.evaluate(candidate_prompt, batch)

                if self.proximal_update:
                    accepted = after_eval.accuracy > before_eval.accuracy or after_eval.accuracy == 1.0
                else:
                    accepted = after_eval.accuracy >= before_eval.accuracy
                if accepted:
                    prompt = candidate_prompt

                proposed_rule = _candidate_as_rule(prompt_before, candidate_prompt, feedback)

                card = GradientCard(
                    card_id=f"{client.client_id}_g{step_index}",
                    source_client=client.client_id,
                    failure_mode=feedback.strip(),
                    proposed_rule=proposed_rule,
                    applicable_condition=f"Reasoning examples from {client.task}",
                    evidence_examples_ids=[row["example_id"] for row in evidence],
                    local_score_before=before_eval.accuracy,
                    local_score_after=after_eval.accuracy if accepted else before_eval.accuracy,
                    task=client.task,
                    local_delta=(after_eval.accuracy - before_eval.accuracy) if accepted else 0.0,
                    token_length=max(1, (len(proposed_rule) + 3) // 4),
                )
                cards.append(card)
                history.append({
                    "epoch": epoch,
                    "step": step_index,
                    "batch_ids": [example.id for example in batch],
                    "prompt_before": prompt_before,
                    "textual_feedback": feedback,
                    "candidate_prompt": candidate_prompt,
                    "score_before": before_eval.accuracy,
                    "score_after": after_eval.accuracy,
                    "accepted": accepted,
                    "prompt_after": prompt,
                })
                step_index += 1
            if step_index >= self.max_steps:
                break

        final_eval = self.evaluator.evaluate(prompt, client.train)
        return LocalOptimizationResult(
            client.client_id, initial_prompt, prompt, cards,
            initial_score, final_eval.accuracy, history,
        )


def _feedback_as_rule(feedback: str) -> str:
    cleaned = " ".join(feedback.strip().split())
    if not cleaned:
        return "Retain the current reasoning strategy; no change was recommended."
    return "Apply this textual-gradient feedback when relevant: " + cleaned


def _candidate_as_rule(before: str, candidate: str, feedback: str) -> str:
    """Extract the concise rule added by a local TGD update when possible."""
    before_lines = {" ".join(line.split()).lower() for line in before.splitlines() if line.strip()}
    additions = []
    for line in candidate.splitlines():
        cleaned = " ".join(line.strip(" -*\t").split())
        if not cleaned or cleaned.lower() in before_lines:
            continue
        if "last line of your response" in cleaned.lower():
            continue
        additions.append(cleaned)
    if additions:
        return " ".join(dict.fromkeys(additions))
    return _feedback_as_rule(feedback)
