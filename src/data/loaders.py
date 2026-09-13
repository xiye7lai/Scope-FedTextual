from __future__ import annotations

import json
import random
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    answer: str
    task: str
    difficulty: float = 0.5
    split: str = ""


def _object_counting(n: int = 120) -> list[Example]:
    nouns = ["apple", "coin", "book", "marble", "flower", "key"]
    out = []
    for i in range(n):
        counts = [1 + ((i * (j + 2) + j) % 5) for j in range(2 + i % 3)]
        chunks = [f"{count} {nouns[(i + j) % len(nouns)]}{'' if count == 1 else 's'}" for j, count in enumerate(counts)]
        question = "I have " + ", ".join(chunks[:-1]) + " and " + chunks[-1] + ". How many objects do I have in total?"
        out.append(Example(f"oc-{i:03d}", question, str(sum(counts)), "object_counting", len(counts) / 5))
    return out


def _multistep_arithmetic(n: int = 120) -> list[Example]:
    out = []
    for i in range(n):
        a, b, c = 2 + i % 13, 1 + (i * 3) % 9, 2 + (i * 5) % 7
        if i % 3 == 0:
            question, answer = f"Compute ({a} + {b}) * {c}.", (a + b) * c
        elif i % 3 == 1:
            question, answer = f"Compute {a} * {c} - {b}.", a * c - b
        else:
            question, answer = f"Compute {a + b + c} - {b} * {c} using standard order of operations.", a + b + c - b * c
        out.append(Example(f"ma-{i:03d}", question, str(answer), "multistep_arithmetic", 0.35 + 0.15 * (i % 3)))
    return out


def _gsm8k(n: int = 120) -> list[Example]:
    names = ["Mina", "Owen", "Liu", "Sara", "Noah"]
    out = []
    for i in range(n):
        name = names[i % len(names)]
        boxes, each, extra = 2 + i % 7, 3 + (i * 2) % 8, 1 + (i * 5) % 6
        if i % 2 == 0:
            question = f"{name} packs {boxes} boxes with {each} pencils in each box, then buys {extra} more pencils. How many pencils does {name} have?"
            answer = boxes * each + extra
        else:
            total = boxes * each + extra
            question = f"{name} had {total} stickers and gave {extra} away. The rest were shared equally among {boxes} friends. How many stickers did each friend get?"
            answer = (total - extra) // boxes
        out.append(Example(f"gsm-{i:03d}", question, str(answer), "gsm8k", 0.55 + 0.1 * (i % 2)))
    return out


BUILTINS = {
    "object_counting": _object_counting,
    "multistep_arithmetic": _multistep_arithmetic,
    "gsm8k": _gsm8k,
}

BBH_TASK_FILES = {
    "object_counting": "object_counting",
    "multistep_arithmetic": "multistep_arithmetic_two",
    "logical_deduction": "logical_deduction_three_objects",
    "tracking_shuffled_objects": "tracking_shuffled_objects_three_objects",
    "date_understanding": "date_understanding",
    "word_sorting": "word_sorting",
    "formal_fallacies": "formal_fallacies",
    "web_of_lies": "web_of_lies",
    "boolean_expressions": "boolean_expressions",
    "causal_judgement": "causal_judgement",
    "disambiguation_qa": "disambiguation_qa",
    "dyck_languages": "dyck_languages",
    "geometric_shapes": "geometric_shapes",
    "hyperbaton": "hyperbaton",
    "movie_recommendation": "movie_recommendation",
    "navigate": "navigate",
    "penguins_in_a_table": "penguins_in_a_table",
    "reasoning_about_colored_objects": "reasoning_about_colored_objects",
    "ruin_names": "ruin_names",
    "salient_translation_error_detection": "salient_translation_error_detection",
    "snarks": "snarks",
    "sports_understanding": "sports_understanding",
    "temporal_sequences": "temporal_sequences",
}


def load_task_examples(task: str, cfg: dict, seed: int) -> list[Example]:
    source = cfg.get("source", "builtin")
    if source == "builtin":
        if task not in BUILTINS:
            raise ValueError(f"Unknown builtin task: {task}")
        values = BUILTINS[task]()
    elif source == "official_fedtextgrad":
        values = _load_official_fedtextgrad(task, cfg)
    elif source == "jsonl":
        path = Path(cfg["jsonl_paths"][task])
        values = []
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                row = json.loads(line)
                values.append(Example(
                    str(row.get("id", f"{task}-{idx}")), str(row["question"]),
                    str(row["answer"]), task, float(row.get("difficulty", 0.5)),
                ))
    else:
        raise ValueError(f"Unsupported data source: {source}")
    if source != "official_fedtextgrad":
        random.Random(seed).shuffle(values)
    return values


def _download_text(url: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "Scope-FedTextual/1.0"})
        last_error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    payload = response.read()
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_bytes(payload)
                temporary.replace(path)
                break
            except (OSError, TimeoutError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Failed to download official dataset after 3 attempts: {url}") from last_error
    return path.read_text(encoding="utf-8")


def _load_official_fedtextgrad(task: str, cfg: dict) -> list[Example]:
    """Match the dataset sources and split indices in ubc-tea/FedTextGrad."""
    cache = Path(cfg.get("cache_dir", ".cache/fedtextgrad_datasets"))
    if task in BBH_TASK_FILES:
        upstream_name = BBH_TASK_FILES[task]
        url = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{upstream_name}.json"
        payload = json.loads(_download_text(url, cache / "bbh" / f"{upstream_name}.json"))
        examples = payload["examples"]
        boundaries = [("train", 0, 50), ("val", 50, 150), ("test", 150, len(examples))]
        output = []
        for split, start, end in boundaries:
            for idx in range(start, end):
                row = examples[idx]
                output.append(Example(
                    id=f"{task}-{idx:03d}", question=str(row["input"]), answer=str(row["target"]),
                    task=task, difficulty=min(1.0, len(str(row["input"]).split()) / 100), split=split,
                ))
        return output

    if task == "gsm8k":
        base = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"
        train_text = _download_text(f"{base}/train.jsonl", cache / "gsm8k" / "train.jsonl")
        test_text = _download_text(f"{base}/test.jsonl", cache / "gsm8k" / "test.jsonl")
        train_rows = [json.loads(line) for line in train_text.splitlines() if line.strip()]
        test_rows = [json.loads(line) for line in test_text.splitlines() if line.strip()]
        random.Random(0).shuffle(train_rows)
        random.Random(0).shuffle(test_rows)
        selected = [
            ("train", train_rows[:50], 0),
            ("val", train_rows[200:300], 200),
            ("test", test_rows[300:400], 300),
        ]
        output = []
        for split, rows, offset in selected:
            for local_idx, row in enumerate(rows):
                final_answer = str(row["answer"]).strip().split("####")[-1].strip().replace(",", "")
                question = f"Question: {row['question']}"
                output.append(Example(
                    id=f"gsm8k-{split}-{offset + local_idx:04d}", question=question,
                    answer=final_answer, task=task,
                    difficulty=min(1.0, len(question.split()) / 100), split=split,
                ))
        return output
    raise ValueError(f"Task {task!r} is not supported by official_fedtextgrad source")
