from __future__ import annotations

from src.data.client_split import ClientData
from src.evaluation.evaluator import Evaluator
from src.textgrad.gradient_card import GradientCard
from concurrent.futures import ThreadPoolExecutor


def estimate_utility(
    clients: list[ClientData],
    cards: list[GradientCard],
    base_prompt: str,
    evaluator: Evaluator,
    job_concurrency: int = 1,
    request_concurrency: int = 1,
) -> tuple[list[list[float]], dict[str, float]]:
    """Clients evaluate rules locally; only scalar utilities leave this function."""
    def score(job):
        client, prompt = job
        return evaluator.evaluate(prompt, client.val, concurrency=request_concurrency).accuracy

    baseline_jobs = [(client, base_prompt) for client in clients]
    with ThreadPoolExecutor(max_workers=max(1, job_concurrency)) as executor:
        baseline_values = list(executor.map(score, baseline_jobs))
    baselines = {client.client_id: value for client, value in zip(clients, baseline_values)}
    candidate_jobs = [
        (client, base_prompt + "\n\n[Candidate rule]\n" + card.proposed_rule)
        for client in clients for card in cards
    ]
    with ThreadPoolExecutor(max_workers=max(1, job_concurrency)) as executor:
        candidate_values = list(executor.map(score, candidate_jobs))
    matrix = []
    offset = 0
    for client in clients:
        values = candidate_values[offset:offset + len(cards)]
        matrix.append([value - baselines[client.client_id] for value in values])
        offset += len(cards)
    return matrix, baselines
