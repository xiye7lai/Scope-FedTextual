from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.data.client_split import ClientData, build_clients
from src.data.loaders import Example
from src.scoped_fedtextgrad.runner import ScopedFedTextGradRunner
from src.scoped_fedtextgrad.gate import apply_validation_gate, should_fallback
from src.scoped_fedtextgrad.selection import compose_prompt, select_rules
from src.scoped_fedtextgrad.utility import estimate_utility
from src.llm.mock_backend import MockBackend
from src.textgrad.gradient_card import GradientCard
from src.textgrad.local_optimizer import LocalTextGradOptimizer
from src.textgrad.official_compat import OFFICIAL_INITIAL_PROMPT, UPSTREAM_COMMIT


class ScopedMethodTests(unittest.TestCase):
    @staticmethod
    def card(card_id: str, source: str, rule: str, tokens: int) -> GradientCard:
        return GradientCard(
            card_id=card_id,
            source_client=source,
            failure_mode="failure",
            proposed_rule=rule,
            applicable_condition="condition",
            task="task",
            token_length=tokens,
        )

    def test_scope_threshold_and_local_priority(self):
        cards = [
            self.card("cross-high", "client_1", "high utility cross rule", 3),
            self.card("local", "client_0", "positive local rule", 3),
            self.card("zero", "client_2", "zero utility rule", 3),
        ]
        selected, decisions = select_rules(
            client_id="client_0",
            cards=cards,
            utilities=[0.9, 0.1, 0.0],
            base_prompt="base",
            budget=100,
            epsilon=0.0,
            token_counter=lambda text: len(text.split()),
        )
        self.assertEqual([card.card_id for card in selected], ["local", "cross-high"])
        zero = next(row for row in decisions if row.rule_id == "zero")
        self.assertFalse(zero.selected)
        self.assertEqual(zero.reason, "utility_not_above_epsilon")

    def test_complete_prompt_respects_hard_budget(self):
        cards = [
            self.card("a", "client_0", "one two three", 3),
            self.card("b", "client_1", "four five six", 3),
        ]
        selected, _ = select_rules(
            client_id="client_0",
            cards=cards,
            utilities=[0.2, 0.3],
            base_prompt="base instruction",
            budget=11,
            token_counter=lambda text: len(text.split()),
        )
        prompt = compose_prompt("base instruction", "client_0", selected)
        self.assertLessEqual(len(prompt.split()), 11)

    def test_safe_gate_falls_back_only_when_margin_is_exceeded(self):
        self.assertTrue(should_fallback(0.60, 0.70, delta=0.05))
        self.assertFalse(should_fallback(0.66, 0.70, delta=0.05))
        decision = apply_validation_gate(
            client_id="client_0",
            local_prompt="local",
            scoped_prompt="scoped",
            local_validation_accuracy=0.70,
            scoped_validation_accuracy=0.60,
            delta=0.05,
            budget=256,
        )
        self.assertTrue(decision.fallback)
        self.assertEqual(decision.prompt, "local")
        self.assertEqual(decision.chosen_source, "Local TextGrad")

    def test_official_mixed_allocation_has_no_overlap(self):
        values = []
        for split, size in (("train", 50), ("val", 100), ("test", 100)):
            values.extend(
                Example(
                    f"task-{split}-{index}", f"question {index}", "a", "task",
                    difficulty=index / size, split=split,
                )
                for index in range(size)
            )
        cfg = {
            "source": "official_fedtextgrad",
            "setting": "mixed",
            "tasks": ["task"],
            "num_clients": 3,
            "train_per_client": 16,
            "val_per_client": 20,
            "test_per_client": 33,
        }
        with patch("src.data.client_split.load_task_examples", return_value=values):
            clients = build_clients(cfg, seed=13)
        for split in ("train", "val", "test"):
            ids = [example.id for client in clients for example in getattr(client, split)]
            self.assertEqual(len(ids), len(set(ids)), split)

    def test_utility_matrix_is_target_client_improvement(self):
        class Result:
            def __init__(self, accuracy):
                self.accuracy = accuracy

        class FakeEvaluator:
            def evaluate(self, prompt, examples, concurrency=1):
                target = examples[0].id
                base = {"c0": 0.5, "c1": 0.4}[target]
                gain = 0.2 if "rule-a" in prompt and target == "c0" else 0.0
                gain += -0.1 if "rule-a" in prompt and target == "c1" else 0.0
                return Result(base + gain)

        clients = [
            ClientData("client_0", "task", "task", [], [Example("c0", "q", "a", "task")], []),
            ClientData("client_1", "task", "task", [], [Example("c1", "q", "a", "task")], []),
        ]
        cards = [self.card("a", "client_0", "rule-a", 2)]
        matrix, baselines = estimate_utility(clients, cards, "base", FakeEvaluator())
        self.assertEqual(baselines, {"client_0": 0.5, "client_1": 0.4})
        self.assertAlmostEqual(matrix[0][0], 0.2)
        self.assertAlmostEqual(matrix[1][0], -0.1)

    def test_fedtextgrad_compatible_local_step(self):
        client = ClientData(
            "client_0",
            "object_counting",
            "object_counting",
            [Example("x", "I have 2 apples and 3 books. How many objects?", "5", "object_counting")],
            [],
            [],
        )
        with TemporaryDirectory() as directory:
            backend = MockBackend("mock", directory)
            result = LocalTextGradOptimizer(
                backend, rounds=1, batch_size=1, max_steps=1,
            ).optimize(client, OFFICIAL_INITIAL_PROMPT)
        self.assertEqual(len(result.cards), 1)
        self.assertEqual(len(result.step_history), 1)
        self.assertEqual(len(UPSTREAM_COMMIT), 40)

    def test_end_to_end_runner_writes_scope_and_gate_artifacts(self):
        with TemporaryDirectory() as directory:
            cfg = {
                "run": {"name": "smoke", "seed": 13, "output_dir": directory, "dry_run": True},
                "data": {
                    "source": "builtin", "setting": "task_skew",
                    "tasks": ["object_counting", "multistep_arithmetic", "gsm8k"],
                    "num_clients": 3, "train_per_client": 3,
                    "val_per_client": 2, "test_per_client": 3,
                },
                "llm": {"provider": "mock", "model": "mock", "max_llm_calls": 1000},
                "evaluation": {"concurrency": 2},
                "optimization": {
                    "local_epochs": 1, "max_steps": 1, "batch_size": 1,
                    "max_failure_examples": 1, "proximal_update": True,
                },
                "aggregation": {"methods": []},
                "scoped": {"variants": ["utility"], "budgets": [128], "epsilon": 0.0},
                "safe": {"enabled": True, "delta": 0.0},
            }
            result = ScopedFedTextGradRunner(cfg).run()
            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "rule_utility_matrix.csv").is_file())
            self.assertTrue((run_dir / "selected_rules_per_client.json").is_file())
            self.assertTrue((run_dir / "safe_gate_decisions.json").is_file())


if __name__ == "__main__":
    unittest.main()
