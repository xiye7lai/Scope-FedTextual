from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from src.data.client_split import build_clients
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import summarize_methods
from src.llm.backend import build_backend
from src.textgrad.local_optimizer import LocalTextGradOptimizer
from src.textgrad.official_compat import (
    FORMATTING_INSTRUCTION,
    OFFICIAL_INITIAL_PROMPT,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
)
from src.utils.io import ensure_dir, write_csv, write_json, write_text

from .aggregation import aggregate_prompts, method_label


LOGGER = logging.getLogger(__name__)
INITIAL_PROMPT = OFFICIAL_INITIAL_PROMPT


class FedTextGradRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        run_cfg = cfg["run"]
        self.run_dir = ensure_dir(Path(run_cfg.get("output_dir", "results")) / run_cfg["name"])
        llm_cfg = dict(cfg["llm"])
        llm_cfg["_dry_run"] = bool(run_cfg.get("dry_run", False))
        self.backend = build_backend(llm_cfg, self.run_dir)
        self.evaluator = Evaluator(
            self.backend, cfg.get("evaluation", {}).get("concurrency", 1),
        )
        opt_cfg = cfg.get("optimization", {})
        self.initial_prompt = str(opt_cfg.get("initial_prompt", INITIAL_PROMPT))
        self.formatting_instruction = str(
            opt_cfg.get("formatting_instruction", FORMATTING_INSTRUCTION)
        )

    def run(self) -> dict:
        started = time.perf_counter()
        seed = int(self.cfg["run"].get("seed", 0))
        write_json(self.run_dir / "run_status.json", {"status": "running", "seed": seed})
        write_json(self.run_dir / "resolved_config.json", self.cfg)
        clients = build_clients(self.cfg["data"], seed)
        optimizer = self._optimizer()
        def optimize_client(client):
            LOGGER.info("Local optimization: %s (%s)", client.client_id, client.task)
            result = optimizer.optimize(client, self.initial_prompt)
            write_json(self.run_dir / "checkpoints" / f"local_{client.client_id}.json", asdict(result))
            return client.client_id, result

        local_concurrency = max(
            1, int(self.cfg.get("evaluation", {}).get("local_client_concurrency", 1)),
        )
        if local_concurrency > 1:
            with ThreadPoolExecutor(max_workers=local_concurrency) as executor:
                pairs = list(executor.map(optimize_client, clients))
        else:
            pairs = [optimize_client(client) for client in clients]
        local_results = dict(pairs)
        cards = [card for client in clients for card in local_results[client.client_id].cards]

        prompts: dict[str, dict[str, str]] = {
            "Initial Prompt": {c.client_id: self.initial_prompt for c in clients},
            "Local TextGrad": {c.client_id: local_results[c.client_id].optimized_prompt for c in clients},
        }
        aggregate_artifacts = {}
        round_history = {}
        for method in self.cfg.get("aggregation", {}).get("methods", ["summary"]):
            label = method_label(method)
            global_prompt = self.initial_prompt
            method_rounds = []
            fed_cfg = self.cfg.get("federation", {})
            communication_rounds = int(fed_cfg.get("rounds", 1))
            fraction = float(fed_cfg.get("client_fraction", 1.0))
            for round_idx in range(communication_rounds):
                sample_size = max(1, min(len(clients), round(fraction * len(clients))))
                selected = random.Random(seed + round_idx * 1009).sample(clients, sample_size)
                round_local = {}
                for client in selected:
                    if round_idx == 0 and global_prompt == self.initial_prompt and len(selected) == len(clients):
                        result = local_results[client.client_id]
                    else:
                        result = self._optimizer().optimize(client, global_prompt)
                    round_local[client.client_id] = result
                local_prompts = [round_local[client.client_id].optimized_prompt for client in selected]
                global_prompt = aggregate_prompts(
                    local_prompts, method, self.backend, self.formatting_instruction,
                )
                method_rounds.append({
                    "round": round_idx,
                    "selected_clients": [client.client_id for client in selected],
                    "local_prompts": {cid: result.optimized_prompt for cid, result in round_local.items()},
                    "local_updates": {cid: asdict(result) for cid, result in round_local.items()},
                    "aggregated_global_prompt": global_prompt,
                })
                safe_label = label.lower().replace(" ", "_").replace("-", "_")
                write_json(
                    self.run_dir / "checkpoints" / f"{safe_label}_round_{round_idx}.json",
                    method_rounds[-1],
                )
            prompts[label] = {c.client_id: global_prompt for c in clients}
            aggregate_artifacts[label] = global_prompt
            round_history[label] = method_rounds

        if self.cfg.get("baselines", {}).get("oracle_centralized", False):
            pooled = type(clients[0])(
                "centralized", "pooled", "pooled",
                [example for client in clients for example in client.train],
                [example for client in clients for example in client.val],
                [example for client in clients for example in client.test],
            )
            oracle_result = self._optimizer().optimize(pooled, self.initial_prompt)
            prompts["Oracle Centralized TextGrad"] = {c.client_id: oracle_result.optimized_prompt for c in clients}
            round_history["Oracle Centralized TextGrad"] = [asdict(oracle_result)]

        scores = {method: {} for method in prompts}
        prediction_records = []
        eval_cfg = self.cfg.get("evaluation", {})
        jobs = [(method, by_client, client) for method, by_client in prompts.items() for client in clients]

        def evaluate_job(job):
            method, by_client, client = job
            result = self.evaluator.evaluate(
                by_client[client.client_id], client.test,
                concurrency=int(eval_cfg.get("test_concurrency_per_job", eval_cfg.get("concurrency", 1))),
            )
            return method, client.client_id, result

        job_concurrency = max(1, int(eval_cfg.get("method_client_concurrency", 1)))
        if job_concurrency > 1:
            with ThreadPoolExecutor(max_workers=job_concurrency) as executor:
                evaluated = list(executor.map(evaluate_job, jobs))
        else:
            evaluated = [evaluate_job(job) for job in jobs]
        for method, client_id, result in evaluated:
            scores[method][client_id] = result.accuracy
            prediction_records.extend(
                {"method": method, "client_id": client_id, **row} for row in result.records
            )
        for method in prompts:
            write_json(
                self.run_dir / "checkpoints" / f"predictions_{_safe_name(method)}.json",
                [row for row in prediction_records if row["method"] == method],
            )

        summary_rows, client_rows = summarize_methods(scores, prompts)
        runtime_seconds = time.perf_counter() - started
        for row in summary_rows:
            row.update({
                "num_clients": len(clients),
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
            })
        self._save(clients, local_results, cards, aggregate_artifacts, round_history, prompts, summary_rows, client_rows, prediction_records)
        write_json(self.run_dir / "run_status.json", {"status": "complete", "seed": seed, "backend_stats": self.backend.stats})
        return {"run_dir": str(self.run_dir), "summary": summary_rows, "backend_stats": self.backend.stats}
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

    def _save(self, clients, local_results, cards, aggregates, round_history, prompts, summary_rows, client_rows, prediction_records):
        write_json(self.run_dir / "resolved_config.json", self.cfg)
        write_json(self.run_dir / "upstream_reference.json", {
            "repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT,
            "compatibility": "full local prompts; textual feedback + TGD update; communication rounds; official aggregation templates",
        })
        write_json(self.run_dir / "clients.json", [{
            "client_id": c.client_id, "task": c.task, "group": c.group,
            "train_ids": [x.id for x in c.train], "val_ids": [x.id for x in c.val], "test_ids": [x.id for x in c.test],
        } for c in clients])
        write_json(self.run_dir / "gradient_cards.json", [card.to_dict() for card in cards])
        write_json(self.run_dir / "local_optimization.json", {key: asdict(value) for key, value in local_results.items()})
        write_json(self.run_dir / "aggregated_prompts.json", aggregates)
        write_json(self.run_dir / "federated_rounds.json", round_history)
        write_json(self.run_dir / "prompts.json", prompts)
        write_json(self.run_dir / "llm_usage.json", self.backend.stats)
        write_json(self.run_dir / "summary.json", summary_rows)
        write_csv(self.run_dir / "main_results_table.csv", summary_rows)
        write_csv(self.run_dir / "per_client_scores.csv", client_rows)
        write_json(self.run_dir / "predictions.json", prediction_records)
        for label, prompt in aggregates.items():
            safe = label.lower().replace(" ", "_").replace("-", "_")
            write_text(self.run_dir / "prompts" / f"{safe}.txt", prompt)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_")
