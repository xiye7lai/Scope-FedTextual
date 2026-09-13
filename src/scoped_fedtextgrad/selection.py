from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from src.evaluation.metrics import prompt_token_estimate
from src.textgrad.gradient_card import GradientCard


@dataclass
class RuleDecision:
    client_id: str
    method: str
    budget: int | None
    rule_id: str
    source_client: str
    source_task: str
    utility: float | None
    token_length: int
    local_delta: float
    selected: bool
    reason: str
    rank_score: float | None

    def to_dict(self) -> dict:
        value = asdict(self)
        value["budget"] = "unlimited" if self.budget is None else self.budget
        return value


def budget_label(budget: int | None) -> str:
    return "unlimited" if budget is None else str(int(budget))


def compose_prompt(
    base_prompt: str,
    client_id: str,
    cards: Iterable[GradientCard],
) -> str:
    cards = list(cards)
    shared = [card.proposed_rule for card in cards if card.source_client != client_id]
    local = [card.proposed_rule for card in cards if card.source_client == client_id]
    sections = [base_prompt.strip()]
    if shared:
        sections.append("[Selected Shared Rules]\n" + "\n".join(f"- {rule}" for rule in shared))
    if local:
        sections.append("[Local Rules]\n" + "\n".join(f"- {rule}" for rule in local))
    return "\n\n".join(sections)


def fit_shared_prompt(
    base_prompt: str,
    shared_prompt: str,
    budget: int | None,
    token_counter: Callable[[str], int],
    section_name: str = "Budgeted Shared Prompt",
) -> str:
    """Keep the base instruction intact and fit a shared baseline exactly."""
    if budget is None:
        return shared_prompt
    if token_counter(base_prompt) > budget:
        raise ValueError(f"budget {budget} is smaller than the base prompt")
    content = shared_prompt.strip()
    prefix = base_prompt.strip() + f"\n\n[{section_name}]\n"
    if token_counter(prefix) > budget:
        return base_prompt.strip()
    if token_counter(prefix + content) <= budget:
        return prefix + content
    low, high = 0, len(content)
    while low < high:
        middle = (low + high + 1) // 2
        if token_counter(prefix + content[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    candidate = (prefix + content[:low]).rstrip()
    while candidate and token_counter(candidate) > budget:
        candidate = candidate[:-1].rstrip()
    return candidate if token_counter(candidate) <= budget else base_prompt.strip()


def select_rules(
    *,
    client_id: str,
    cards: list[GradientCard],
    utilities: list[float],
    base_prompt: str,
    budget: int | None,
    method: str = "utility",
    epsilon: float = 0.0,
    seed: int = 0,
    token_counter: Callable[[str], int] = prompt_token_estimate,
) -> tuple[list[GradientCard], list[RuleDecision]]:
    """Select candidate rules with an auditable, budget-safe greedy policy.

    ``method`` is one of utility, random, text_similarity, local_only, or
    global_only.  Random and text-similarity never inspect utility when ranking;
    they are genuine routing ablations rather than utility-assisted variants.
    """
    if len(cards) != len(utilities):
        raise ValueError("cards and utilities must have equal length")
    if budget is not None and token_counter(base_prompt) > budget:
        raise ValueError(f"budget {budget} is smaller than the base prompt")

    local_text = " ".join(
        f"{card.failure_mode} {card.proposed_rule}"
        for card in cards if card.source_client == client_id
    )
    ranked: list[tuple[int, float, int, GradientCard, float | None]] = []
    pre_rejected: dict[str, str] = {}
    rng = random.Random(_stable_seed(seed, client_id, budget_label(budget), method))

    random_scores = {card.card_id: rng.random() for card in cards}
    for index, (card, utility) in enumerate(zip(cards, utilities)):
        length = _card_tokens(card)
        if method == "utility":
            if utility <= epsilon:
                pre_rejected[card.card_id] = "utility_not_above_epsilon"
                continue
            score = utility / length
            # Positive local rules are considered before shared rules, as
            # required by the protocol, while remaining subject to the budget.
            priority = 0 if card.source_client == client_id else 1
            rank_utility: float | None = utility
        elif method == "random":
            score, priority, rank_utility = random_scores[card.card_id], 0, None
        elif method == "text_similarity":
            score = _cosine_tokens(local_text, f"{card.failure_mode} {card.proposed_rule}")
            priority, rank_utility = 0, None
        elif method == "local_only":
            if card.source_client != client_id:
                pre_rejected[card.card_id] = "nonlocal_rule_excluded"
                continue
            if utility <= epsilon:
                pre_rejected[card.card_id] = "utility_not_above_epsilon"
                continue
            score, priority, rank_utility = utility / length, 0, utility
        elif method == "global_only":
            if card.source_client == client_id:
                pre_rejected[card.card_id] = "local_rule_excluded"
                continue
            if utility <= epsilon:
                pre_rejected[card.card_id] = "utility_not_above_epsilon"
                continue
            score, priority, rank_utility = utility / length, 0, utility
        else:
            raise ValueError(f"unknown selection method: {method}")
        ranked.append((priority, -score, index, card, rank_utility))

    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    selected: list[GradientCard] = []
    decisions: list[RuleDecision] = []
    considered = set()
    for _, neg_score, _, card, rank_utility in ranked:
        considered.add(card.card_id)
        candidate = selected + [card]
        fits = budget is None or token_counter(
            compose_prompt(base_prompt, client_id, candidate)
        ) <= budget
        if fits:
            selected.append(card)
            reason = "selected"
        else:
            reason = "token_budget_exceeded"
        decisions.append(_decision(
            client_id, method, budget, card, rank_utility, fits, reason,
            -neg_score,
        ))
    for card, utility in zip(cards, utilities):
        if card.card_id in considered:
            continue
        decisions.append(_decision(
            client_id, method, budget, card,
            utility if method in {"utility", "local_only", "global_only"} else None,
            False, pre_rejected.get(card.card_id, "excluded"), None,
        ))
    assert budget is None or token_counter(
        compose_prompt(base_prompt, client_id, selected)
    ) <= budget
    return selected, decisions


def _decision(client_id, method, budget, card, utility, selected, reason, score):
    return RuleDecision(
        client_id=client_id, method=method, budget=budget,
        rule_id=card.card_id, source_client=card.source_client,
        source_task=card.task, utility=utility, token_length=_card_tokens(card),
        local_delta=card.local_delta, selected=selected, reason=reason,
        rank_score=score,
    )


def decision_dict(decision: RuleDecision) -> dict:
    return decision.to_dict()


def _card_tokens(card: GradientCard) -> int:
    return card.token_length or max(1, math.ceil((len(card.proposed_rule) + 2) / 4))


def _stable_seed(*values) -> int:
    digest = hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()
    return int(digest[:16], 16)


def _token_counts(text: str) -> dict[str, int]:
    output: dict[str, int] = {}
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        output[token] = output.get(token, 0) + 1
    return output


def _cosine_tokens(left: str, right: str) -> float:
    a, b = _token_counts(left), _token_counts(right)
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
