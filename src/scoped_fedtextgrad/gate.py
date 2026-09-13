from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GateDecision:
    client_id: str
    budget: str
    delta: float
    local_validation_accuracy: float
    scoped_validation_accuracy: float
    fallback: bool
    chosen_source: str
    prompt: str

    def to_dict(self) -> dict:
        value = asdict(self)
        value.pop("prompt")
        return value


def should_fallback(
    scoped_validation_accuracy: float,
    local_validation_accuracy: float,
    delta: float = 0.0,
) -> bool:
    """Return Eq. (18): fall back if Scoped is worse than Local by > delta."""
    return scoped_validation_accuracy < local_validation_accuracy - delta


def apply_validation_gate(
    *,
    client_id: str,
    local_prompt: str,
    scoped_prompt: str,
    local_validation_accuracy: float,
    scoped_validation_accuracy: float,
    delta: float = 0.0,
    budget: str | int = "unknown",
) -> GateDecision:
    fallback = should_fallback(
        scoped_validation_accuracy, local_validation_accuracy, delta,
    )
    return GateDecision(
        client_id=client_id,
        budget=str(budget),
        delta=float(delta),
        local_validation_accuracy=float(local_validation_accuracy),
        scoped_validation_accuracy=float(scoped_validation_accuracy),
        fallback=fallback,
        chosen_source="Local TextGrad" if fallback else "Scoped-FedTextGrad",
        prompt=local_prompt if fallback else scoped_prompt,
    )
