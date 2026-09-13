from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from src.data.client_split import build_clients
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import summarize_methods
from src.fedtextgrad.aggregation import aggregate_prompts, method_label
from src.fedtextgrad.fed_runner import INITIAL_PROMPT
from src.llm.backend import build_backend
from src.scoped_fedtextgrad.utility import estimate_utility
from src.textgrad.local_optimizer import LocalTextGradOptimizer
from src.textgrad.official_compat import (
    FORMATTING_INSTRUCTION,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
)
from src.utils.io import ensure_dir, write_csv, write_json
from src.utils.tokens import build_token_counter

from .selection import (
    budget_label,
    compose_prompt,
    decision_dict,
    fit_shared_prompt,
    select_rules,
)
from .gate import apply_validation_gate
from .rule_extraction import extract_compact_rules


LOGGER = logging.getLogger(__name__)

SCOPED_VARIANTS = {
    "Scoped-FedTextGrad": "utility",
    "Scoped w/o Utility": "random",
    "Scoped Text-Similarity only": "text_similarity",
    "Scoped Local-only": "local_only",
    "Scoped Global-only": "global_only",
}


def configured_scoped_variants(scoped_cfg: dict) -> dict[str, str]:
    """Return the requested selector subset while preserving display order."""
    requested = scoped_cfg.get("variants")
    if requested is None:
        return dict(SCOPED_VARIANTS)
    aliases = {
        display.lower(): display for display in SCOPED_VARIANTS
    } | {
        selector.lower(): display for display, selector in SCOPED_VARIANTS.items()
    }
    output = {}
    for value in requested:
        key = str(value).strip().lower()
        if key not in aliases:
            raise ValueError(
                f"Unknown scoped variant {value!r}; expected one of "
                f"{sorted(set(aliases))}"
            )
        display = aliases[key]
        output[display] = SCOPED_VARIANTS[display]
    if not output:
        raise ValueError("scoped.variants must not be empty")
    return output


class ScopedFedTextGradRunner:
    """End-to-end implementation of Scoped-FedTextGrad and Safe-Scoped."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        run_cfg = cfg["run"]
        self.run_dir = ensure_dir(Path(run_cfg.get("output_dir", "results")) / run_cfg["name"])
        llm_cfg = dict(cfg["llm"])
        llm_cfg["_dry_run"] = bool(run_cfg.get("dry_run", False))
        self.backend = build_backend(llm_cfg, self.run_dir)
        eval_cfg = cfg.get("evaluation", {})
        self.evaluator = Evaluator(self.backend, eval_cfg.get("concurrency", 1))
        opt_cfg = cfg.get("optimization", {})
        self.initial_prompt = str(opt_cfg.get("initial_prompt", INITIAL_PROMPT))
        self.formatting_instruction = str(
            opt_cfg.get("formatting_instruction", FORMATTING_INSTRUCTION)
        )
        self.token_counter, self.token_counter_name = build_token_counter(llm_cfg)

    def run(self) -> dict:
        started = time.perf_counter()
        seed = int(self.cfg["run"].get("seed", 0))
        write_json(self.run_dir / "run_status.json", {"status": "running", "seed": seed})
        write_json(self.run_dir / "resolved_config.json", self.cfg)
        clients = build_clients(self.cfg["data"], seed)
        client_ids = [client.client_id for client in clients]
        local_results = self._local_updates(clients)
        cards = [card for client in clients for card in local_results[client.client_id].cards]
        if not cards:
            raise RuntimeError("Scoped-FedTextGrad requires at least one candidate rule")
        cards = extract_compact_rules(
            cards, self.backend, self.token_counter,
            max_rule_tokens=int(self.cfg.get("scoped", {}).get("max_rule_tokens", 64)),
        )

        eval_cfg = self.cfg.get("evaluation", {})
        matrix, baselines = estimate_utility(
            clients, cards, self.initial_prompt, self.evaluator,
            job_concurrency=int(eval_cfg.get("utility_job_concurrency", 1)),
            request_concurrency=int(eval_cfg.get("utility_concurrency_per_job", 1)),
        )
        write_json(self.run_dir / "checkpoints" / "rule_utility.json", {
            "client_ids": client_ids,
            "rule_ids": [card.card_id for card in cards],
            "matrix": matrix,
            "validation_baselines": baselines,
        })

        prompts, metadata, selected_records, rejected_records = self._build_prompts(
            clients, local_results, cards, matrix, seed,
        )
        gate_records: list[dict] = []
        if bool(self.cfg.get("safe", {}).get("enabled", True)):
            gate_records = self._add_safe_prompts(prompts, metadata, clients)
        scores, predictions = self._evaluate(prompts, clients)
        summary_rows, client_rows = summarize_methods(
            scores, prompts, token_counter=self.token_counter,
        )
        runtime_seconds = time.perf_counter() - started
        for row in summary_rows:
            row.update(metadata[row["method"]])
            row.update(self._run_metadata(len(clients), seed, runtime_seconds))
        for row in client_rows:
            row.update(metadata[row["method"]])
        self._save(
            clients, local_results, cards, matrix, baselines, prompts, summary_rows,
            client_rows, predictions, selected_records, rejected_records, gate_records,
        )
        write_json(self.run_dir / "run_status.json", {
            "status": "complete", "seed": seed, "backend_stats": self.backend.stats,
            "num_rules": len(cards), "budgets": self._budgets(),
        })
        return {
            "run_dir": str(self.run_dir), "summary": summary_rows,
            "backend_stats": self.backend.stats, "gate_decisions": gate_records,
        }

    def _local_updates(self, clients):
        optimizer = self._optimizer()
        concurrency = max(1, int(
            self.cfg.get("evaluation", {}).get("local_client_concurrency", 1)
        ))

        def optimize(client):
            LOGGER.info("Scoped candidate extraction: %s (%s)", client.client_id, client.task)
            result = optimizer.optimize(client, self.initial_prompt)
            write_json(
                self.run_dir / "checkpoints" / f"local_{client.client_id}.json",
                asdict(result),
            )
            return client.client_id, result

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            return dict(executor.map(optimize, clients))

    def _optimizer(self):
        opt = self.cfg.get("optimization", {})
        return LocalTextGradOptimizer(
            self.backend,
            rounds=int(opt.get("local_epochs", opt.get("rounds", 1))),
            max_failures=int(opt.get("max_failure_examples", 3)),
            batch_size=int(opt.get("batch_size", self.cfg.get("data", {}).get("train_per_client", 3))),
            max_steps=int(opt.get("max_steps", opt.get("local_epochs", opt.get("rounds", 1)))),
            proximal_update=bool(opt.get("proximal_update", True)),
            evaluation_concurrency=int(self.cfg.get("evaluation", {}).get("concurrency", 1)),
        )

    def _budgets(self) -> list[int | None]:
        values = self.cfg.get("scoped", {}).get("budgets", [128, 256, 512, 1024, "unlimited"])
        output = []
        for value in values:
            output.append(None if str(value).lower() in {"none", "unlimited", "inf"} else int(value))
        if len(output) != len(set(output)):
            raise ValueError("scoped.budgets contains duplicates")
        return output

    def _build_prompts(self, clients, local_results, cards, matrix, seed):
        ids = [client.client_id for client in clients]
        prompts: dict[str, dict[str, str]] = {
            "Initial Prompt": {cid: self.initial_prompt for cid in ids},
            "Local TextGrad": {cid: local_results[cid].optimized_prompt for cid in ids},
        }
        metadata = {
            "Initial Prompt": {"Budget": "natural", "selection_method": "baseline", "method_family": "Initial Prompt"},
            "Local TextGrad": {"Budget": "natural", "selection_method": "baseline", "method_family": "Local TextGrad"},
        }
        for method in self.cfg.get("aggregation", {}).get(
            "methods", ["concat", "summarization", "sum_uid"],
        ):
            label = method_label(method)
            is_summary = method in {"summarization", "summary", "sum_uid", "uid_summary"}
            evaluate_natural = (
                not is_summary
                or bool(self.cfg.get("scoped", {}).get("evaluate_unbudgeted_summaries", True))
            )
            if evaluate_natural:
                shared = aggregate_prompts(
                    [local_results[cid].optimized_prompt for cid in ids], method,
                    self.backend, self.formatting_instruction,
                )
                prompts[label] = {cid: shared for cid in ids}
                metadata[label] = {"Budget": "natural", "selection_method": "baseline", "method_family": label}
            if is_summary:
                for budget in self._budgets():
                    budgeted_label = f"{label} @{budget_label(budget)}"
                    budgeted_shared = self._budgeted_summary(
                        [local_results[cid].optimized_prompt for cid in ids],
                        method, budget,
                    )
                    fitted = fit_shared_prompt(
                        self.initial_prompt, budgeted_shared, budget, self.token_counter,
                        section_name=label,
                    )
                    prompts[budgeted_label] = {cid: fitted for cid in ids}
                    metadata[budgeted_label] = {
                        "Budget": budget_label(budget),
                        "selection_method": "budgeted_baseline",
                        "method_family": label,
                    }

        scoped_cfg = self.cfg.get("scoped", {})
        selected_records, rejected_records = [], []
        for budget in self._budgets():
            for display_name, selector in configured_scoped_variants(scoped_cfg).items():
                label = f"{display_name} @{budget_label(budget)}"
                prompts[label] = {}
                metadata[label] = {
                    "Budget": budget_label(budget), "selection_method": selector,
                    "method_family": display_name,
                }
                for client, utilities in zip(clients, matrix):
                    selected, decisions = select_rules(
                        client_id=client.client_id, cards=cards, utilities=utilities,
                        base_prompt=self.initial_prompt, budget=budget, method=selector,
                        epsilon=float(scoped_cfg.get("epsilon", 0.0)),
                        seed=seed,
                        token_counter=self.token_counter,
                    )
                    prompts[label][client.client_id] = compose_prompt(
                        self.initial_prompt, client.client_id, selected,
                    )
                    records = [
                        {**decision_dict(item), "display_method": display_name}
                        for item in decisions
                    ]
                    selected_records.extend(row for row in records if row["selected"])
                    rejected_records.extend(row for row in records if not row["selected"])
        return prompts, metadata, selected_records, rejected_records

    def _add_safe_prompts(self, prompts, metadata, clients) -> list[dict]:
        """Apply Eq. (18) using validation data only, before any test evaluation."""
        safe_cfg = self.cfg.get("safe", {})
        delta = float(safe_cfg.get("delta", 0.0))
        eval_cfg = self.cfg.get("evaluation", {})
        scoped_labels = [
            label for label in prompts if label.startswith("Scoped-FedTextGrad @")
        ]
        if not scoped_labels:
            return []

        jobs = []
        for client in clients:
            jobs.append(("Local TextGrad", client, prompts["Local TextGrad"][client.client_id]))
            for label in scoped_labels:
                jobs.append((label, client, prompts[label][client.client_id]))

        def evaluate(job):
            label, client, prompt = job
            result = self.evaluator.evaluate(
                prompt,
                client.val,
                concurrency=int(eval_cfg.get(
                    "gate_concurrency_per_job", eval_cfg.get("concurrency", 1),
                )),
            )
            return label, client.client_id, result.accuracy

        with ThreadPoolExecutor(max_workers=max(1, int(
            eval_cfg.get("gate_job_concurrency", 1)
        ))) as executor:
            validation_scores = {
                (label, client_id): accuracy
                for label, client_id, accuracy in executor.map(evaluate, jobs)
            }

        records = []
        for scoped_label in scoped_labels:
            budget = scoped_label.rsplit("@", 1)[1]
            safe_label = f"Safe-Scoped @{budget}"
            prompts[safe_label] = {}
            metadata[safe_label] = {
                "Budget": budget,
                "selection_method": "validation_gate",
                "method_family": "Safe-Scoped",
                "gate_delta": delta,
            }
            for client in clients:
                local_accuracy = validation_scores[("Local TextGrad", client.client_id)]
                scoped_accuracy = validation_scores[(scoped_label, client.client_id)]
                decision = apply_validation_gate(
                    client_id=client.client_id,
                    local_prompt=prompts["Local TextGrad"][client.client_id],
                    scoped_prompt=prompts[scoped_label][client.client_id],
                    local_validation_accuracy=local_accuracy,
                    scoped_validation_accuracy=scoped_accuracy,
                    delta=delta,
                    budget=budget,
                )
                prompts[safe_label][client.client_id] = decision.prompt
                records.append(decision.to_dict())
        return records

    def _budgeted_summary(self, local_prompts, method, budget):
        if budget is None:
            return aggregate_prompts(
                local_prompts, method, self.backend, self.formatting_instruction,
            )
        uid = method in {"sum_uid", "uid_summary"}
        prompt = (
            "Merge the client prompts below into one concise system prompt. Preserve "
            "the most useful reasoning rules and the exact final Answer: $VALUE format. "
            + ("Use Uniform Information Density so no client dominates. " if uid else "")
            + f"The complete returned prompt must be at most {budget} tokens under the "
            "configured exact tokenizer. Return only the prompt.\n\n"
            + "\n\n".join(
                f"<CLIENT_PROMPT_{index + 1}>{value}</CLIENT_PROMPT_{index + 1}>"
                for index, value in enumerate(local_prompts)
            )
        )
        return self.backend.generate(
            prompt,
            purpose="aggregate_uid_budgeted" if uid else "aggregate_summary_budgeted",
            metadata={"prompts": local_prompts, "budget": budget, "uid": uid},
        )

    def _evaluate(self, prompts, clients):
        eval_cfg = self.cfg.get("evaluation", {})
        jobs = [
            (method, by_client, client)
            for method, by_client in prompts.items() for client in clients
        ]

        def evaluate(job):
            method, by_client, client = job
            result = self.evaluator.evaluate(
                by_client[client.client_id], client.test,
                concurrency=int(eval_cfg.get(
                    "test_concurrency_per_job", eval_cfg.get("concurrency", 1),
                )),
            )
            return method, client.client_id, result

        with ThreadPoolExecutor(max_workers=max(1, int(
            eval_cfg.get("method_client_concurrency", 1)
        ))) as executor:
            evaluated = list(executor.map(evaluate, jobs))
        scores = {method: {} for method in prompts}
        predictions = []
        for method, client_id, result in evaluated:
            scores[method][client_id] = result.accuracy
            predictions.extend(
                {"method": method, "client_id": client_id, **row}
                for row in result.records
            )
        return scores, predictions

    def _run_metadata(self, num_clients, seed, runtime_seconds):
        return {
            "num_clients": num_clients,
            "setting": self.cfg["data"].get("setting", "unknown"),
            "seed": seed,
            "LLMRequests": self.backend.stats["requests"],
            "LLMAPICalls": self.backend.stats["api_calls"],
            "LLMAPIAttempts": self.backend.stats["api_attempts"],
            "CacheHits": self.backend.stats["cache_hits"],
            "InputTokens": self.backend.stats["input_tokens"],
            "OutputTokens": self.backend.stats["output_tokens"],
            "CacheHitRate": (
                self.backend.stats["cache_hits"] / self.backend.stats["requests"]
                if self.backend.stats["requests"] else 0.0
            ),
            "RuntimeSeconds": runtime_seconds,
            "LocalSteps": int(self.cfg.get("optimization", {}).get("max_steps", 1)),
            "BatchSize": int(self.cfg.get("optimization", {}).get("batch_size", 1)),
            "TokenCounter": self.token_counter_name,
        }

    def _save(self, clients, local_results, cards, matrix, baselines, prompts,
              summary_rows, client_rows, predictions, selected, rejected, gate_records):
        write_json(self.run_dir / "upstream_reference.json", {
            "repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT,
            "extension": "Scoped-FedTextGrad routes textual-gradient rules by client utility under token budgets",
        })
        write_json(self.run_dir / "clients.json", [{
            "client_id": client.client_id, "task": client.task, "group": client.group,
            "train_ids": [x.id for x in client.train],
            "val_ids": [x.id for x in client.val],
            "test_ids": [x.id for x in client.test],
        } for client in clients])
        write_json(self.run_dir / "candidate_rules.json", [card.to_dict() for card in cards])
        write_json(self.run_dir / "local_optimization.json", {
            key: asdict(value) for key, value in local_results.items()
        })
        write_json(self.run_dir / "validation_baselines.json", baselines)
        write_json(self.run_dir / "selected_rules_per_client.json", selected)
        write_json(self.run_dir / "rejected_rules.json", rejected)
        write_json(self.run_dir / "safe_gate_decisions.json", gate_records)
        write_json(self.run_dir / "final_client_prompts.json", prompts)
        write_json(self.run_dir / "predictions.json", predictions)
        write_json(self.run_dir / "llm_usage.json", self.backend.stats)
        write_json(self.run_dir / "token_budget_protocol.json", {
            "counter": self.token_counter_name,
            "budgets": [budget_label(value) for value in self._budgets()],
            "constraint": "tokens(base_prompt + selected shared rules + local rules) <= budget",
        })
        utility_rows = []
        for client, values in zip(clients, matrix):
            for card, utility in zip(cards, values):
                utility_rows.append({
                    "client_id": client.client_id,
                    "client_task": client.task,
                    "rule_id": card.card_id,
                    "source_client": card.source_client,
                    "source_task": card.task,
                    "failure_mode": card.failure_mode,
                    "proposed_rule": card.proposed_rule,
                    "local_delta": card.local_delta,
                    "token_length": card.to_dict()["token_length"],
                    "utility": utility,
                })
        # Emit both names: the pipeline specification names
        # utility_matrix_rules.csv while the required-output checklist names
        # rule_utility_matrix.csv.
        write_csv(self.run_dir / "rule_utility_matrix.csv", utility_rows)
        write_csv(self.run_dir / "utility_matrix_rules.csv", utility_rows)
        write_csv(self.run_dir / "main_results_table.csv", summary_rows)
        write_csv(self.run_dir / "per_client_scores.csv", client_rows)
