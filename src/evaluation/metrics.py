from __future__ import annotations

import math
import statistics
from typing import Callable, Iterable


def prompt_token_estimate(prompt: str) -> int:
    return max(1, math.ceil(len(prompt) / 4))


def summarize_methods(
    scores: dict[str, dict[str, float]],
    prompts: dict[str, dict[str, str]],
    local_method: str = "Local TextGrad",
    token_counter: Callable[[str], int] = prompt_token_estimate,
) -> tuple[list[dict], list[dict]]:
    clients = sorted(next(iter(scores.values()))) if scores else []
    local_scores = scores.get(local_method, {})
    summary_rows = []
    client_rows = []
    for method, by_client in scores.items():
        values = [float(by_client[c]) for c in clients]
        deltas = [by_client[c] - local_scores.get(c, by_client[c]) for c in clients]
        method_prompts = prompts.get(method, {})
        lengths = [token_counter(method_prompts.get(c, "")) for c in clients]
        ntr = sum(delta < 0 for delta in deltas) / len(deltas) if deltas else 0.0
        summary_rows.append({
            "method": method,
            "AvgAcc": statistics.mean(values) if values else 0.0,
            "WorstClientAcc": min(values) if values else 0.0,
            "StdAcc": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "NTR": ntr,
            "AvgPromptTokens": statistics.mean(lengths) if lengths else 0.0,
        })
        for client, value, delta, length in zip(clients, values, deltas, lengths):
            client_rows.append({
                "method": method,
                "client_id": client,
                "accuracy": value,
                "local_accuracy": local_scores.get(client, value),
                "delta_vs_local": delta,
                "prompt_tokens": length,
            })
    return summary_rows, client_rows
