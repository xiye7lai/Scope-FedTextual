from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm import LLMCallBudgetExceeded
from src.scoped_fedtextgrad import ScopedFedTextGradRunner
from src.utils.config import load_config
from src.utils.io import write_json
from src.utils.logging import configure_logging


def configure_client_sample_sizes(cfg: dict, num_clients: int, fixed: bool = False) -> None:
    """Resolve per-client split sizes without changing legacy grid semantics.

    Client-count scaling experiments use ``fixed=True`` so that changing K does
    not also change the amount of train/validation/test data available to each
    client.  Existing experiment configs continue to use the original
    capacity-aware sizing and minimum-evaluation-size guard.
    """
    if fixed:
        sizes = {
            split: int(cfg["data"][f"{split}_per_client"])
            for split in ("train", "val", "test")
        }
        if any(value <= 0 for value in sizes.values()):
            raise ValueError("fixed client sample sizes must be positive")
        cfg["data"]["max_samples_per_client"] = sizes
        return

    per_task = (num_clients + len(cfg["data"]["tasks"]) - 1) // len(cfg["data"]["tasks"])
    cfg["data"]["train_per_client"] = min(20, 50 // per_task)
    cfg["data"]["val_per_client"] = min(20, 100 // per_task)
    cfg["data"]["test_per_client"] = min(50, 100 // per_task)
    cfg["data"]["max_samples_per_client"] = {
        split: cfg["data"][f"{split}_per_client"]
        for split in ("train", "val", "test")
    }
    if cfg["data"]["val_per_client"] < 20 or cfg["data"]["test_per_client"] < 30:
        raise ValueError(f"insufficient evaluation split for num_clients={num_clients}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Scoped-FedTextGrad exact-v2 grid")
    parser.add_argument("--config", default="configs/paper_grid.yaml")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    base = load_config(args.config)
    grid = base["grid"]
    root = Path(base["run"]["output_dir"])
    if args.dry_run:
        root = root.parent / f"{root.name}_dry_run"
        base["run"]["output_dir"] = str(root)
    manifest_path = root / "grid_manifest.json"
    manifest = {
        "config": str(Path(args.config).resolve()),
        "expected_runs": len(grid["settings"]) * len(grid["clients"]) * len(grid["local_steps"]) * len(grid["batch_sizes"]) * len(grid["seeds"]),
        "runs": [],
    }
    launched = 0
    stop = False
    for setting in grid["settings"]:
        for clients in grid["clients"]:
            for local_steps in grid["local_steps"]:
                for batch_size in grid["batch_sizes"]:
                    for seed in grid["seeds"]:
                        run_name = f"scoped_{setting}_k{clients}_e{local_steps}_b{batch_size}_s{seed}"
                        run_dir = root / run_name
                        status_path = run_dir / "run_status.json"
                        if status_path.exists() and not args.rerun_complete:
                            status = json.loads(status_path.read_text(encoding="utf-8"))
                            if status.get("status") == "complete":
                                manifest["runs"].append({"run_name": run_name, "status": "complete", "skipped": True})
                                continue
                        if args.max_runs is not None and launched >= args.max_runs:
                            stop = True
                            break
                        cfg = copy.deepcopy(base)
                        cfg.pop("grid", None)
                        cfg["run"].update({"name": run_name, "seed": seed, "dry_run": args.dry_run})
                        if args.dry_run:
                            cfg["llm"]["require_exact_tokenizer"] = False
                            cfg["llm"]["shared_cache_dir"] = str(root / "mock_cache")
                        cfg["llm"]["seed"] = seed
                        cfg["data"].update({"setting": setting, "num_clients": clients})
                        configure_client_sample_sizes(
                            cfg, clients,
                            fixed=bool(grid.get("fixed_client_sample_sizes", False)),
                        )
                        cfg["optimization"].update({
                            "local_epochs": local_steps, "max_steps": local_steps,
                            "batch_size": batch_size,
                        })
                        runner = ScopedFedTextGradRunner(cfg)
                        record = {
                            "run_name": run_name, "setting": setting, "clients": clients,
                            "local_steps": local_steps, "batch_size": batch_size, "seed": seed,
                        }
                        try:
                            result = runner.run()
                            record.update({"status": "complete", "backend_stats": result["backend_stats"]})
                        except LLMCallBudgetExceeded as exc:
                            record.update({"status": "budget_exhausted", "error": str(exc), "backend_stats": runner.backend.stats})
                            write_json(run_dir / "run_status.json", record)
                        except Exception as exc:
                            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "backend_stats": runner.backend.stats})
                            write_json(run_dir / "run_status.json", record)
                        manifest["runs"].append(record)
                        write_json(manifest_path, manifest)
                        launched += 1
                        if record["status"] != "complete" and not args.continue_on_error:
                            stop = True
                            break
                    if stop: break
                if stop: break
            if stop: break
        if stop: break
    write_json(manifest_path, manifest)
    print(json.dumps({"expected_runs": manifest["expected_runs"], "recorded": len(manifest["runs"]), "launched": launched}, indent=2))
    if any(row["status"] in {"failed", "budget_exhausted"} for row in manifest["runs"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
