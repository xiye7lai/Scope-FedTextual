"""Scope-preserving federated textual optimization."""

from .gate import GateDecision, apply_validation_gate, should_fallback
from .rule_extraction import extract_compact_rules
from .runner import ScopedFedTextGradRunner
from .selection import RuleDecision, compose_prompt, select_rules
from .utility import estimate_utility

__all__ = [
    "GateDecision",
    "RuleDecision",
    "ScopedFedTextGradRunner",
    "apply_validation_gate",
    "compose_prompt",
    "estimate_utility",
    "extract_compact_rules",
    "select_rules",
    "should_fallback",
]
