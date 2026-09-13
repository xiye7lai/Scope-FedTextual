from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math


@dataclass
class GradientCard:
    card_id: str
    source_client: str
    failure_mode: str
    proposed_rule: str
    applicable_condition: str
    evidence_examples_ids: list[str] = field(default_factory=list)
    local_score_before: float = 0.0
    local_score_after: float = 0.0
    task: str = ""
    local_delta: float = 0.0
    token_length: int = 0

    def to_dict(self) -> dict:
        value = asdict(self)
        # ``source_task`` is explicit provenance; it is not used by the main router.
        value["source_task"] = self.task
        value["local_delta"] = self.local_delta
        value["token_length"] = self.token_length or max(
            1, math.ceil(len(self.proposed_rule) / 4),
        )
        return value
